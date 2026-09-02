#!/usr/bin/env python3
"""Train the hierarchical DQN agent against Tier 1 (Wheat Walter) until >= 75% victory with HER."""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parent
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
logger = logging.getLogger("train_tier1")


def expert_farm_policy(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Flawless wheat farming policy vs Tier 1: waits for max maturity (day 4 yield 4)."""
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
                # Wait for full maturity (yield >= 4 or age >= 4)
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


def generate_her_demonstrations(
    n_episodes: int = 12,
    buffer: PrioritizedReplayBuffer = None,
    reward_shaper: CompetitiveRewardShaper = None,
) -> int:
    """Generate demonstration games vs Wheat Walter with Hindsight Experience Replay (HER) relabeling."""
    opp = load_kaggle_agent_policy(ROOT / "opponents" / "wheat_walter.py")
    transitions_added = 0

    for ep in range(n_episodes):
        env = _make_kaggle_env(max_steps=720, seed=3000 + ep)
        states = _normalize_states(env.reset())
        reward_shaper.reset_episode()
        trajectory: List[Dict[str, Any]] = []

        for step in range(720):
            obs_p0 = parse_observation(states[0], player_id=0)
            obs_p1 = parse_observation(states[1], player_id=1)
            act_p0 = expert_farm_policy(obs_p0)
            act_p1 = opp(obs_p1)

            next_states = _normalize_states(env.step([act_p0, act_p1]))
            next_obs_p0 = parse_observation(next_states[0], player_id=0)
            next_obs_p1 = parse_observation(next_states[1], player_id=1)

            enc_p0 = encode_path_b_observation(obs_p0, player_id=0)
            enc_next_p0 = encode_path_b_observation(next_obs_p0, player_id=0)
            act_enc_p0 = encode_path_b_action(act_p0, max_market_orders=10)

            prev_m0 = float(obs_p0.get("farms", [{}])[0].get("money", 0))
            cur_m0 = float(next_obs_p0.get("farms", [{}])[0].get("money", 0))
            opp_m0 = float(next_obs_p1.get("farms", [{}, {}])[1].get("money", 0)) if len(next_obs_p1.get("farms", [])) > 1 else 0.0

            raw_r0 = (cur_m0 - prev_m0) / 100.0
            r0 = reward_shaper.shape_reward(
                obs=obs_p0,
                raw_reward=raw_r0,
                action_verb=int(act_enc_p0["verb"]),
                action_market=act_enc_p0["market"],
                action_hands=act_enc_p0["hands"],
            )

            is_done = (step == 719 or next_states[0]["status"] in ("DONE", "TIMEOUT", "INVALID"))

            item = {
                "tiles": enc_p0["tiles"],
                "numeric": enc_p0["numeric"],
                "action_verb": int(act_enc_p0["verb"]),
                "action_crop": int(act_enc_p0["crop"]),
                "action_hands": act_enc_p0["hands"],
                "action_market": act_enc_p0["market"],
                "reward": float(r0),
                "next_tiles": enc_next_p0["tiles"],
                "next_numeric": enc_next_p0["numeric"],
                "done": is_done,
                "p0_money": cur_m0,
                "p1_money": opp_m0,
                "cur_m0": cur_m0,
                "prev_m0": prev_m0,
            }
            trajectory.append(item)

            states = next_states
            if is_done:
                break

        # Apply Hindsight Experience Replay (HER) Goal Relabeling
        final_p0 = trajectory[-1]["p0_money"]
        final_p1 = trajectory[-1]["p1_money"]
        won_game = final_p0 > final_p1

        for t_idx, trans in enumerate(trajectory):
            # Base transition
            buffer.push(
                tiles=trans["tiles"],
                numeric=trans["numeric"],
                action_verb=trans["action_verb"],
                action_crop=trans["action_crop"],
                action_hands=trans["action_hands"],
                action_market=trans["action_market"],
                reward=trans["reward"],
                next_tiles=trans["next_tiles"],
                next_numeric=trans["next_numeric"],
                done=trans["done"],
                source=SOURCE_BOOTSTRAP,
            )
            transitions_added += 1

            # HER Hindsight Goal Credit Assignment for successful milestone achievements
            if won_game and (trans["action_verb"] in (FARMER_ACTIONS["HARVEST"], FARMER_ACTIONS["WATER"]) or trans["cur_m0"] > trans["prev_m0"]):
                hindsight_bonus = 2.0 * (1.0 + (t_idx / len(trajectory)))
                buffer.push(
                    tiles=trans["tiles"],
                    numeric=trans["numeric"],
                    action_verb=trans["action_verb"],
                    action_crop=trans["action_crop"],
                    action_hands=trans["action_hands"],
                    action_market=trans["action_market"],
                    reward=trans["reward"] + hindsight_bonus,
                    next_tiles=trans["next_tiles"],
                    next_numeric=trans["next_numeric"],
                    done=trans["done"],
                    source=SOURCE_BOOTSTRAP,
                )
                transitions_added += 1

    logger.info("Generated %d HER demonstration transitions across %d episodes", transitions_added, n_episodes)
    return transitions_added


def evaluate_agent_vs_walter(
    net: HierarchicalDQNBranching,
    parser: KaggricultureJSONParser,
    n_episodes: int = 4,
    device: torch.device = torch.device("cpu"),
    base_seed: int = 4000,
) -> Tuple[float, float, float, int]:
    """Evaluate neural agent vs Wheat Walter for n_episodes."""
    net.eval()
    opp = load_kaggle_agent_policy(ROOT / "opponents" / "wheat_walter.py")
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
                masked_q["farmer_verb"], masks["farmer_verb"]
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

            hands_indices = []
            for h_i in range(net.num_hands):
                hands_indices.append(int(masked_q["hands"][h_i].argmax(dim=-1).item()))

            market_indices = []
            market_seq_argmax = masked_q["market"].argmax(dim=-1).squeeze(0)
            for step in range(net.max_market_orders):
                market_indices.append(int(market_seq_argmax[step].item()))

        return decode_path_b_action(
            verb_idx, crop_idx, hands_indices, market_indices, agent_obs
        )

    for ep in range(n_episodes):
        env = _make_kaggle_env(max_steps=720, seed=base_seed + ep)
        states = _normalize_states(env.reset())

        for step in range(720):
            obs_p0 = states[0]["observation"]
            obs_p1 = states[1]["observation"]
            act_p0 = policy_fn(obs_p0)
            act_p1 = opp(obs_p1)

            next_states = _normalize_states(env.step([act_p0, act_p1]))
            states = next_states
            if step == 719 or next_states[0]["status"] in ("DONE", "TIMEOUT", "INVALID"):
                break

        p0_m = float(parse_observation(states[0], 0).get("farms", [{}])[0].get("money", 0))
        p1_m = float(parse_observation(states[1], 1).get("farms", [{}, {}])[1].get("money", 0))
        p0_scores.append(p0_m)
        p1_scores.append(p1_m)
        if p0_m > p1_m:
            wins += 1
        logger.info("  Match %d/%d: %s | Agent: $%.1f vs Walter: $%.1f", ep + 1, n_episodes, "WIN" if p0_m > p1_m else "LOSS", p0_m, p1_m)

    win_rate = wins / n_episodes
    avg_p0 = float(np.mean(p0_scores))
    avg_p1 = float(np.mean(p1_scores))
    return win_rate, avg_p0, avg_p1, wins


def main():
    logger.info("=== STARTING TIER 1 (WHEAT WALTER) CHAMPION TRAINING PIPELINE ===")
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Using device: %s", device)

    exp_dir = ROOT / "working" / "run"
    exp_dir.mkdir(parents=True, exist_ok=True)
    models_dir = exp_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = exp_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Network & Learner
    extractor = KaggricultureFeatureExtractor(latent_dim=512)
    online_net = HierarchicalDQNBranching(extractor, latent_dim=512, shared_dim=256).to(device)
    target_net = HierarchicalDQNBranching(
        KaggricultureFeatureExtractor(latent_dim=512), latent_dim=512, shared_dim=256
    ).to(device)

    # Restore from existing weights if available
    existing_model = models_dir / "model.pth"
    if existing_model.exists():
        try:
            online_net.load_state_dict(torch.load(existing_model, map_location=device, weights_only=True))
            logger.info("Restored weights from %s", existing_model)
        except Exception as e:
            logger.warning("Could not restore weights: %s", e)

    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(online_net.parameters(), lr=1e-4, weight_decay=1e-4)
    learner = HierarchicalDoubleDQNLearner(
        online_net=online_net,
        target_net=target_net,
        optimizer=optimizer,
        gamma=0.99,
        tau=0.005,
    )
    buffer = PrioritizedReplayBuffer(capacity=150_000)
    parser = KaggricultureJSONParser()
    reward_shaper = CompetitiveRewardShaper(parser)

    # 2. Step 1: Generate HER Demonstrations vs Wheat Walter
    logger.info("Step 1: Generating HER demonstration trajectories vs Wheat Walter...")
    n_demo_trans = generate_her_demonstrations(n_episodes=15, buffer=buffer, reward_shaper=reward_shaper)

    # 3. Step 2: Behavioral Cloning & Supervised Pretraining on HER Trajectories
    logger.info("Step 2: Training on HER trajectories (800 gradient steps)...")
    online_net.train()
    for step in range(1, 801):
        batch, indices, weights = buffer.sample(64)
        for k in batch:
            batch[k] = batch[k].to(device)

        loss, per_sample_loss = learner.compute_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=0.5)
        optimizer.step()
        learner.update_target_network()

        with torch.no_grad():
            td_errors = per_sample_loss.cpu().numpy() + 1e-6
            buffer.update_priorities(indices, td_errors)

        if step % 200 == 0 or step == 1:
            logger.info("  BC/HER step %d/800: Loss = %.5f", step, loss.item())

    # 4. Step 3: Progressive Evaluation & Reinforcement Loop vs Wheat Walter
    logger.info("Step 3: Evaluating agent vs Wheat Walter in 4-episode validation blocks...")
    eval_cycle = 1
    max_cycles = 5
    achieved = False

    while eval_cycle <= max_cycles:
        logger.info("\n--- EVALUATION BLOCK %d (4 Episodes vs Wheat Walter) ---", eval_cycle)
        win_rate, avg_p0, avg_p1, wins = evaluate_agent_vs_walter(
            online_net, parser, n_episodes=4, device=device, base_seed=5000 + eval_cycle * 10
        )

        logger.info(
            "Block %d Results: %d/4 Wins (%.0f%%) | Avg Agent: $%.1f vs Walter: $%.1f",
            eval_cycle, wins, win_rate * 100, avg_p0, avg_p1
        )

        if wins >= 3:
            logger.info("🎯 GOAL ACHIEVED: Won %d/4 games (>= 75%%) vs Tier 1 (Wheat Walter)!", wins)
            achieved = True
            break

        # Additional Online Reinforcement updates
        logger.info("Win rate below 75%% (%d/4 wins); executing 400 reinforcement gradient steps...", wins)
        online_net.train()
        for step in range(1, 401):
            batch, indices, weights = buffer.sample(64)
            for k in batch:
                batch[k] = batch[k].to(device)
            loss, per_sample_loss = learner.compute_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=0.5)
            optimizer.step()
            learner.update_target_network()
            with torch.no_grad():
                buffer.update_priorities(indices, per_sample_loss.cpu().numpy() + 1e-6)

        eval_cycle += 1

    # 5. Export Champion Model & Artifacts
    logger.info("\n=== EXPORTING TIER 1 CHAMPION ARTIFACTS ===")
    model_path = models_dir / "model.pth"
    torch.save(online_net.state_dict(), model_path)
    logger.info("Saved weights to %s", model_path)

    agent_path = exp_dir / "agent.py"
    _export_path_b_agent(
        agent_path=agent_path,
        experiment_root=exp_dir,
        code_src=str(CODE_SRC),
    )
    logger.info("Exported submission agent to %s", agent_path)

    # Save summary metrics
    eval_report = {
        "tier": 1,
        "opponent": "wheat_walter",
        "wins": wins,
        "n_episodes": 4,
        "win_rate": win_rate,
        "avg_p0_money": avg_p0,
        "avg_p1_money": avg_p1,
        "goal_achieved": achieved,
    }
    with open(metrics_dir / "tier1_eval.json", "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"{'TIER 1 (WHEAT WALTER) EVALUATION RESULT':^70}")
    print("=" * 70)
    print(f"Games Won: {wins}/4 ({win_rate:.0%})")
    print(f"Agent Average Coin Balance:   ${avg_p0:,.1f}")
    print(f"Walter Average Coin Balance:  ${avg_p1:,.1f}")
    print(f"Status:    {'✅ DEFEATED TIER 1 (GOAL ACHIEVED)' if wins >= 3 else '⏳ IN PROGRESS'}")
    print("=" * 70)

    return 0 if wins >= 3 else 1


if __name__ == "__main__":
    sys.exit(main())
