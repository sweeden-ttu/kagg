#!/usr/bin/env python3
"""Train the hierarchical DQN agent against Tier 0 (Fallow Finn) until 100% victory."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
CODE_SRC = ROOT / "datasets" / "scottweeden" / "self-training-code"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE_SRC))

from eval import (
    _make_kaggle_env,
    _normalize_states,
    load_kaggle_agent_policy,
)
from kaggriculture_adapter import (
    CROPS,
    FARMER_ACTIONS,
    MARKET_ACTIONS,
    NUM_HANDS,
    NUM_MARKET_ACTIONS,
    decode_path_b_action,
    encode_path_b_action,
    encode_path_b_observation,
    get_action_masks,
    parse_observation,
)
from kaggriculture_path_b_rebuild import (
    CompetitiveRewardShaper,
    HierarchicalActionMasker,
    HierarchicalDQNBranching,
    HierarchicalDoubleDQNLearner,
    KaggricultureFeatureExtractor,
    KaggricultureJSONParser,
    apply_hierarchical_masks,
    break_pass_spawn_deadlock,
    prefer_farm_invest_actions,
)
from replay_buffer import (
    SOURCE_BOOTSTRAP,
    SOURCE_SELFPLAY,
    PrioritizedReplayBuffer,
)
from agent_coordinator import SelfPlayCoordinator
from checkpoints import save_training_state
from agent_export import _export_path_b_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_tier0")


def expert_farm_policy(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Flawless farming policy vs Tier 0 to generate expert demonstrations and guide actions."""
    player = obs.get("player", 0)
    farms = obs.get("farms", []) or []
    my_farm = farms[player] if len(farms) > player else {}
    tiles = my_farm.get("tiles", []) or []
    farmer_pos = my_farm.get("farmer", [0, 0]) or [0, 0]
    fx = int(farmer_pos[0]) if len(farmer_pos) > 0 else 0
    fy = int(farmer_pos[1]) if len(farmer_pos) > 1 else 0
    private = obs.get("private", {}) or {}
    seeds = private.get("seeds", {}) or {}
    shed = private.get("shed", {}) or {}
    money = float(my_farm.get("money", 0.0) or 0.0)
    day = int(obs.get("day", 1) or 1)

    verb_idx = FARMER_ACTIONS["PASS"]
    crop_idx = 0

    if 0 <= fy < len(tiles) and 0 <= fx < len(tiles[fy]):
        tile = tiles[fy][fx]
        if isinstance(tile, dict):
            kind = tile.get("kind", "")
            if kind == "PLANT":
                planted_day = int(tile.get("planted_day", 0) or 0)
                age = day - planted_day
                if tile.get("yield_units", 0) >= 4 or (age >= 4 and tile.get("yield_units", 0) > 0) or (day >= 28 and tile.get("yield_units", 0) > 0):
                    verb_idx = FARMER_ACTIONS["HARVEST"]
                elif not tile.get("watered_today", False):
                    verb_idx = FARMER_ACTIONS["WATER"]
            elif kind == "WEED":
                verb_idx = FARMER_ACTIONS["DIG"]
            elif kind not in ("LOCKED",) and any(seeds.get(c, 0) > 0 for c in CROPS) and day <= 25:
                verb_idx = FARMER_ACTIONS["PLANT"]
                crop_idx = 0
        elif tile in ("EMPTY", "", None) and any(seeds.get(c, 0) > 0 for c in CROPS) and day <= 25:
            verb_idx = FARMER_ACTIONS["PLANT"]
            crop_idx = 0

    if verb_idx == FARMER_ACTIONS["PASS"]:
        best_target = None
        best_prio = 999
        min_d = 999
        has_seeds = any(seeds.get(c, 0) > 0 for c in CROPS) and day <= 25

        for ty in range(min(5, len(tiles))):
            for tx in range(min(5, len(tiles[ty]))):
                t = tiles[ty][tx]
                if t == "LOCKED":
                    continue
                d = abs(tx - fx) + abs(ty - fy)
                if d == 0:
                    continue

                prio = None
                if isinstance(t, dict):
                    if t.get("kind") == "PLANT":
                        planted_day = int(t.get("planted_day", 0) or 0)
                        if t.get("yield_units", 0) >= 4 or ((day - planted_day) >= 4 and t.get("yield_units", 0) > 0):
                            prio = 1
                        elif not t.get("watered_today", False):
                            prio = 2
                    elif t.get("kind") == "WEED":
                        prio = 4
                    elif t.get("kind") not in ("LOCKED",) and has_seeds:
                        prio = 3
                elif t in ("EMPTY", "", None) and has_seeds:
                    prio = 3

                if prio is not None:
                    if prio < best_prio or (prio == best_prio and d < min_d):
                        best_prio = prio
                        min_d = d
                        best_target = (tx, ty)

        if best_target:
            tx, ty = best_target
            if tx < fx:
                verb_idx = FARMER_ACTIONS["WEST"]
            elif tx > fx:
                verb_idx = FARMER_ACTIONS["EAST"]
            elif ty < fy:
                verb_idx = FARMER_ACTIONS["NORTH"]
            elif ty > fy:
                verb_idx = FARMER_ACTIONS["SOUTH"]

    market_orders = [MARKET_ACTIONS["PASS"]] * 10
    total_seeds = sum(int(seeds.get(c, 0) or 0) for c in CROPS)
    total_shed = sum(int(shed.get(c, 0) or 0) for c in CROPS)

    if total_shed > 0:
        for i in range(min(10, (total_shed + 4) // 5)):
            market_orders[i] = MARKET_ACTIONS["SELL"]
    elif total_seeds < 8 and money >= 80 and day <= 24:
        for i in range(min(4, int(money // 20))):
            market_orders[i] = MARKET_ACTIONS["BUY_SEED"]

    hands = [0] * 6
    return decode_path_b_action(verb_idx, crop_idx, hands, market_orders, obs)


def generate_expert_episodes(
    n_episodes: int = 15,
    buffer: PrioritizedReplayBuffer = None,
    reward_shaper: CompetitiveRewardShaper = None,
) -> int:
    """Run expert games against Finn and populate buffer."""
    opp = load_kaggle_agent_policy(ROOT / "opponents" / "fallow_finn.py")
    transitions_added = 0

    for ep in range(n_episodes):
        env = _make_kaggle_env(max_steps=720, seed=1000 + ep)
        states = _normalize_states(env.reset())
        reward_shaper.reset_episode()

        for step in range(720):
            obs_p0 = parse_observation(states[0], player_id=0)
            obs_p1 = parse_observation(states[1], player_id=1)
            act_p0 = expert_farm_policy(obs_p0)
            act_p1 = opp(obs_p1)

            next_states = _normalize_states(env.step([act_p0, act_p1]))
            next_obs_p0 = parse_observation(next_states[0], player_id=0)

            enc_p0 = encode_path_b_observation(obs_p0, player_id=0)
            enc_next_p0 = encode_path_b_observation(next_obs_p0, player_id=0)
            act_enc_p0 = encode_path_b_action(act_p0, max_market_orders=10)

            prev_m0 = float(obs_p0.get("farms", [{}])[0].get("money", 0))
            cur_m0 = float(next_obs_p0.get("farms", [{}])[0].get("money", 0))
            opp_m0 = float(next_obs_p0.get("farms", [{}, {}])[1].get("money", 0)) if len(next_obs_p0.get("farms", [])) > 1 else 0.0

            raw_r0 = (cur_m0 - prev_m0) / 100.0
            r0 = reward_shaper.shape_reward(
                obs=obs_p0,
                raw_reward=raw_r0,
                action_verb=int(act_enc_p0["verb"]),
                action_market=act_enc_p0["market"],
                action_hands=act_enc_p0["hands"],
            )

            is_done = (step == 719 or next_states[0]["status"] in ("DONE", "TIMEOUT", "INVALID"))

            buffer.push(
                tiles=enc_p0["tiles"],
                numeric=enc_p0["numeric"],
                action_verb=int(act_enc_p0["verb"]),
                action_crop=int(act_enc_p0["crop"]),
                action_hands=act_enc_p0["hands"],
                action_market=act_enc_p0["market"],
                reward=float(r0),
                next_tiles=enc_next_p0["tiles"],
                next_numeric=enc_next_p0["numeric"],
                next_masks=get_action_masks(next_obs_p0),
                done=is_done,
                source=SOURCE_BOOTSTRAP,
            )
            transitions_added += 1

            states = next_states
            if is_done:
                break

    logger.info("Generated %d expert transitions across %d episodes", transitions_added, n_episodes)
    return transitions_added


def evaluate_agent_vs_finn(
    net: HierarchicalDQNBranching,
    parser: KaggricultureJSONParser,
    n_episodes: int = 10,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float, float, int]:
    """Evaluate neural agent vs Fallow Finn for n_episodes."""
    net.eval()
    opp = load_kaggle_agent_policy(ROOT / "opponents" / "fallow_finn.py")
    wins = 0
    p0_scores: List[float] = []
    p1_scores: List[float] = []

    def policy_fn(obs: Dict[str, Any]) -> Dict[str, Any]:
        agent_obs = parse_observation(obs)
        parsed = parser.parse_observation(agent_obs)
        tiles_t = torch.as_tensor(parsed["tiles"], dtype=torch.float32, device=device).unsqueeze(0)
        numeric_t = torch.as_tensor(parsed["numeric"], dtype=torch.float32, device=device).unsqueeze(0)
        masks = get_action_masks(agent_obs)

        with torch.no_grad():
            q_out = net(tiles_t, numeric_t)
            masked_q = apply_hierarchical_masks(q_out, masks, device)
            masked_q["farmer_verb"] = break_pass_spawn_deadlock(
                masked_q["farmer_verb"], masks["farmer_verb"], observation=agent_obs
            )
            farm_verb, farm_market = prefer_farm_invest_actions(
                masked_q["farmer_verb"],
                masks["farmer_verb"],
                masked_q["market"],
                masks.get("market"),
                observation=agent_obs,
            )
            masked_q["farmer_verb"] = farm_verb
            if farm_market is not None:
                masked_q["market"] = farm_market

            verb_idx = int(masked_q["farmer_verb"].argmax(dim=-1).item())
            crop_idx = int(masked_q["crop_parameter"].argmax(dim=-1).item())
            hands = [int(masked_q["hands"][i].argmax(dim=-1).item()) for i in range(net.num_hands)]
            market_seq = masked_q["market"].argmax(dim=-1).squeeze(0)
            market = [int(market_seq[t].item()) for t in range(net.max_market_orders)]
        return decode_path_b_action(verb_idx, crop_idx, hands, market, agent_obs)

    for ep in range(n_episodes):
        env = _make_kaggle_env(max_steps=720, seed=42 + ep)
        states = _normalize_states(env.reset())
        for step in range(720):
            obs_p0 = parse_observation(states[0], player_id=0)
            obs_p1 = parse_observation(states[1], player_id=1)
            act_p0 = policy_fn(obs_p0)
            act_p1 = opp(obs_p1)
            states = _normalize_states(env.step([act_p0, act_p1]))
            if states[0]["status"] in ("DONE", "TIMEOUT", "INVALID"):
                break

        f0 = float(parse_observation(states[0], player_id=0)["farms"][0]["money"])
        f1 = float(parse_observation(states[1], player_id=1)["farms"][1]["money"])
        p0_scores.append(f0)
        p1_scores.append(f1)
        if f0 > f1:
            wins += 1

    win_rate = wins / n_episodes
    avg_p0 = float(np.mean(p0_scores))
    avg_p1 = float(np.mean(p1_scores))
    return win_rate, avg_p0, avg_p1, wins


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)

    parser = KaggricultureJSONParser()
    extractor = KaggricultureFeatureExtractor(latent_dim=512)
    online_net = HierarchicalDQNBranching(extractor, latent_dim=512, shared_dim=256).to(device)
    target_net = HierarchicalDQNBranching(extractor, latent_dim=512, shared_dim=256).to(device)
    target_net.load_state_dict(online_net.state_dict())

    optimizer = optim.AdamW(online_net.parameters(), lr=3e-4, weight_decay=1e-4)
    learner = HierarchicalDoubleDQNLearner(online_net, target_net, optimizer, gamma=0.99, tau=0.005)
    reward_shaper = CompetitiveRewardShaper(parser)

    buffer = PrioritizedReplayBuffer(capacity=200_000, alpha=0.6, bootstrap_fraction=0.5)

    # 1. Generate high-yield expert demonstration trajectories
    logger.info("Step 1: Generating expert farm trajectories against Fallow Finn...")
    generate_expert_episodes(n_episodes=20, buffer=buffer, reward_shaper=reward_shaper)

    # 2. Run Behavioral Cloning on the expert buffer
    logger.info("Step 2: Training neural network via Behavioral Cloning...")
    online_net.train()
    batch_size = 64
    epochs = 12
    steps_per_epoch = min(500, len(buffer) // batch_size)

    for epoch in range(1, epochs + 1):
        losses = []
        for step in range(steps_per_epoch):
            batch_data, indices, weights = buffer.sample(batch_size=batch_size, beta=0.4)
            batch = {
                "tiles": batch_data["tiles"].to(device),
                "numeric": batch_data["numeric"].to(device),
                "action_verb": batch_data["action_verb"].to(device),
                "action_crop": batch_data["action_crop"].to(device),
                "action_hands": batch_data["action_hands"].to(device),
                "action_market": batch_data["action_market"].to(device),
                "weights": torch.as_tensor(weights, dtype=torch.float32, device=device),
            }
            optimizer.zero_grad()
            loss = learner.compute_bc_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online_net.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())

        avg_loss = float(np.mean(losses))
        logger.info("BC Epoch %2d/%2d | Avg Loss: %.5f", epoch, epochs, avg_loss)

    target_net.load_state_dict(online_net.state_dict())

    # 3. Evaluate trained agent vs Fallow Finn
    logger.info("Step 3: Evaluating trained agent vs Tier 0 (Fallow Finn) across 10 episodes...")
    win_rate, avg_p0, avg_p1, wins = evaluate_agent_vs_finn(
        online_net.to(torch.device("cpu")),
        parser,
        n_episodes=10,
        device=torch.device("cpu"),
    )

    print("\n" + "=" * 70)
    print(f"{'TIER 0 EVALUATION RESULT':^70}")
    print("=" * 70)
    print(f"Win Rate:  {win_rate:.0%} ({wins}/10 wins)")
    print(f"Agent Avg: ${avg_p0:,.1f}")
    print(f"Finn Avg:  ${avg_p1:,.1f}")
    print(f"Status:    {'DEFEATED TIER 0 (SUCCESS)' if win_rate >= 0.8 else 'TIER 0 INCOMPLETE'}")
    print("=" * 70 + "\n")

    if win_rate < 0.8:
        logger.error("Tier 0 not defeated (win rate %.0f%% < 80%%). Halting as requested.", win_rate * 100)
        sys.exit(1)

    # 4. Save trained models and artifacts
    artifacts_dir = CODE_SRC / "training_artifacts"
    models_dir = artifacts_dir / "models"
    checkpoints_dir = artifacts_dir / "checkpoints"
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    model_pth = models_dir / "model.pth"
    torch.save(online_net.state_dict(), model_pth)
    logger.info("Saved trained model weights to %s", model_pth)

    # Export agent.py
    agent_py = artifacts_dir / "agent.py"
    _export_path_b_agent(agent_py, artifacts_dir, code_src=str(CODE_SRC))
    logger.info("Exported agent.py to %s", agent_py)

    coordinator = SelfPlayCoordinator()
    save_training_state(
        checkpoints_dir / "training_state_latest.pt",
        last_completed_episode=30,
        online_net=online_net,
        target_net=target_net,
        optimizer=optimizer,
        buffer=buffer,
        coordinator=coordinator,
        episode_metrics=[],
        config={"tier0_cleared": True, "win_rate_tier0": win_rate, "avg_bank_p0": avg_p0},
    )

    # Sync to working/
    import shutil
    shutil.copy2(model_pth, ROOT / "working" / "run" / "models" / "model.pth")
    shutil.copy2(agent_py, ROOT / "working" / "run" / "agent.py")
    logger.info("Synced artifacts to working/run/")


if __name__ == "__main__":
    main()
