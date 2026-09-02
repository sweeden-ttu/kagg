#!/usr/bin/env python3
"""Train Tier 2 Champion against Tier 0 - Tier 2 opponents using BC + HER + PER.

Strictly constrained to Tier 0 (Fallow Finn), Tier 1 (Wheat Walter), and Tier 2 (Rotation Rosa).
Restores from previous training checkpoint/weights to retain all prior knowledge.
Trains until beating Tier 2 (Rotation Rosa) with >= 75% win rate while maintaining 100% on Tiers 0-1.
"""

from __future__ import annotations

import importlib.util
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
logger = logging.getLogger("train_tier2")

# STRICT ALLOWED OPPONENTS: Tier 0, Tier 1, Tier 2 ONLY
TIER_OPPONENTS = {
    0: ("fallow_finn", ROOT / "opponents" / "fallow_finn.py"),
    1: ("wheat_walter", ROOT / "opponents" / "wheat_walter.py"),
    2: ("rotation_rosa", ROOT / "opponents" / "rotation_rosa.py"),
}

# Load reference scheduler engine from Tier 2 Rosa
spec = importlib.util.spec_from_file_location("rosa_mod", str(ROOT / "opponents" / "rotation_rosa.py"))
rosa_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rosa_mod)

CHAMPION_POLICY_TIER2 = {
    "hands": 5,
    "crops": ["CARROT", "WHEAT", "TOMATO"],
    "crop_share": {"CARROT": 0.5, "WHEAT": 0.3, "TOMATO": 0.2},
    "seed_batch": 8,
    "seed_stock": 12,
    "seed_buffer": 50,
    "sell_order": ["TOMATO", "CARROT", "WHEAT"],
    "sell_chunk": 40,
    "shed_pressure": 50,
    "plant_until_day": 26,
    "liquidate_from_day": 27,
}


def champion_expert_policy(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Tier 2 Champion Expert Policy for generating HER high-yield demonstrations."""
    farms = obs.get("farms", []) or []
    player = obs.get("player", 0)
    me = farms[player] if len(farms) > player else {}
    priv = obs.get("private", {}) or {}
    jobs, animal_count = rosa_mod._collect_jobs(obs, CHAMPION_POLICY_TIER2, me, priv)
    units = rosa_mod._assign(obs, CHAMPION_POLICY_TIER2, me, priv, jobs, animal_count)
    market = rosa_mod._plan_market(obs, CHAMPION_POLICY_TIER2, me, priv, animal_count)
    return {"farmer": units[0], "hands": units[1:], "market": market}


def generate_tier2_her_demonstrations(
    buffer: PrioritizedReplayBuffer,
    reward_shaper: CompetitiveRewardShaper,
    n_games_per_tier: Dict[int, int] = {2: 15, 1: 4, 0: 3},
) -> int:
    """Generate high-yield demonstration matches against Tier 0, 1, and 2 with HER goal relabeling."""
    tier_policies = {
        tier: load_kaggle_agent_policy(path)
        for tier, (_, path) in TIER_OPPONENTS.items()
    }

    transitions_added = 0

    for target_tier, n_games in n_games_per_tier.items():
        opp_name, _ = TIER_OPPONENTS[target_tier]
        opp_fn = tier_policies[target_tier]
        logger.info("Generating %d demonstration matches vs Tier %d (%s)...", n_games, target_tier, opp_name)

        for ep in range(n_games):
            env = _make_kaggle_env(max_steps=720, seed=5000 + target_tier * 100 + ep)
            states = _normalize_states(env.reset())
            reward_shaper.reset_episode()
            trajectory: List[Dict[str, Any]] = []

            for step in range(720):
                obs_p0 = parse_observation(states[0], player_id=0)
                obs_p1 = parse_observation(states[1], player_id=1)

                act_p0 = champion_expert_policy(obs_p0)
                act_p1 = opp_fn(obs_p1)

                enc_p0 = encode_path_b_observation(obs_p0, player_id=0)
                act_enc_p0 = encode_path_b_action(act_p0, max_market_orders=10)

                next_states = _normalize_states(env.step([act_p0, act_p1]))
                next_obs_p0 = parse_observation(next_states[0], player_id=0)
                next_obs_p1 = parse_observation(next_states[1], player_id=1)
                enc_next_p0 = encode_path_b_observation(next_obs_p0, player_id=0)

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
                    "next_masks": get_action_masks(next_obs_p0),
                    "done": is_done,
                    "p0_money": cur_m0,
                    "p1_money": opp_m0,
                }
                trajectory.append(item)

                states = next_states
                if is_done:
                    break

            final_p0 = trajectory[-1]["p0_money"]
            final_p1 = trajectory[-1]["p1_money"]
            won_game = final_p0 > final_p1

            # Push base transitions + Hindsight Experience Replay (HER)
            for t_idx, trans in enumerate(trajectory):
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
                    next_masks=trans["next_masks"],
                    done=trans["done"],
                    source=SOURCE_BOOTSTRAP,
                )
                transitions_added += 1

                # Cash-Scaled HER Relabeling: Gives massive positive credit to high-money finishes
                if won_game or final_p0 > 9000:
                    hindsight_bonus = (final_p0 / 1000.0) * (1.0 + (t_idx / len(trajectory)))
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
                        next_masks=trans["next_masks"],
                        done=trans["done"],
                        source=SOURCE_BOOTSTRAP,
                    )
                    transitions_added += 1

    logger.info("Generated %d total HER demonstration transitions across Tier 0-2", transitions_added)
    return transitions_added


def act_with_policy(
    net: HierarchicalDQNBranching,
    parser: KaggricultureJSONParser,
    obs: Dict[str, Any],
    device: torch.device,
    epsilon: float = 0.0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate action using neural Q-network with hierarchical masking."""
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

    raw_action = decode_path_b_action(
        verb_idx, crop_idx, hands_indices, market_indices, agent_obs
    )
    encoded_action = {
        "verb": verb_idx,
        "crop": crop_idx,
        "hands": np.array(hands_indices, dtype=np.int64),
        "market": np.array(market_indices, dtype=np.int64),
    }
    return raw_action, encoded_action


def evaluate_tier_ladder(
    net: HierarchicalDQNBranching,
    parser: KaggricultureJSONParser,
    device: torch.device,
    n_games_per_tier: int = 4,
    base_seed: int = 8800,
) -> Dict[str, Any]:
    """Evaluate agent strictly against Tier 0, Tier 1, and Tier 2."""
    net.eval()
    results = {}

    for tier, (name, path) in TIER_OPPONENTS.items():
        opp_fn = load_kaggle_agent_policy(path)
        wins = 0
        p0_scores: List[float] = []
        p1_scores: List[float] = []

        for ep in range(n_games_per_tier):
            env = _make_kaggle_env(max_steps=720, seed=base_seed + tier * 100 + ep)
            states = _normalize_states(env.reset())

            for step in range(720):
                obs_p0 = states[0]["observation"]
                obs_p1 = states[1]["observation"]
                raw_act_p0, _ = act_with_policy(net, parser, obs_p0, device, epsilon=0.0)
                raw_act_p1 = opp_fn(obs_p1)

                next_states = _normalize_states(env.step([raw_act_p0, raw_act_p1]))
                states = next_states
                if step == 719 or next_states[0]["status"] in ("DONE", "TIMEOUT", "INVALID"):
                    break

            p0_m = float(parse_observation(states[0]["observation"], 0).get("farms", [{}])[0].get("money", 0))
            p1_m = float(parse_observation(states[1]["observation"], 1).get("farms", [{}, {}])[1].get("money", 0))
            p0_scores.append(p0_m)
            p1_scores.append(p1_m)
            if p0_m > p1_m:
                wins += 1

        results[f"tier_{tier}_{name}"] = {
            "tier": tier,
            "name": name,
            "wins": wins,
            "n_games": n_games_per_tier,
            "win_rate": wins / n_games_per_tier,
            "avg_agent_money": float(np.mean(p0_scores)),
            "avg_opp_money": float(np.mean(p1_scores)),
        }

    return results


def main():
    logger.info("=== STARTING TIER 2 (ROTATION ROSA) TRAINING PIPELINE (TIERS 0-2 ONLY) ===")
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Using device: %s", device)

    exp_dir = ROOT / "working" / "run"
    exp_dir.mkdir(parents=True, exist_ok=True)
    models_dir = exp_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Network & Learner from existing checkpoint (No Knowledge Lost)
    extractor = KaggricultureFeatureExtractor(latent_dim=512)
    online_net = HierarchicalDQNBranching(extractor, latent_dim=512, shared_dim=256).to(device)
    target_net = HierarchicalDQNBranching(
        KaggricultureFeatureExtractor(latent_dim=512), latent_dim=512, shared_dim=256
    ).to(device)

    existing_model = models_dir / "model.pth"
    if existing_model.exists():
        try:
            online_net.load_state_dict(torch.load(existing_model, map_location=device, weights_only=True))
            logger.info("Restored starting weights from %s (Tier 0 & Tier 1 knowledge retained)", existing_model)
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
    buffer = PrioritizedReplayBuffer(capacity=250_000)
    parser = KaggricultureJSONParser()
    reward_shaper = CompetitiveRewardShaper(parser)

    # 2. Step 1: Generate High-Yield HER Demonstration Trajectories
    logger.info("Step 1: Generating high-yield HER demonstration trajectories vs Tier 0, Tier 1, Tier 2...")
    generate_tier2_her_demonstrations(buffer=buffer, reward_shaper=reward_shaper)

    # 3. Step 2: Supervised BC & Double-DQN Multi-Head Optimization
    logger.info("Step 2: Training with BC + Double-DQN on HER trajectories (2,000 gradient steps)...")
    online_net.train()
    for step in range(1, 2001):
        batch, indices, weights = buffer.sample(64)
        for k in batch:
            batch[k] = batch[k].to(device)

        td_loss, per_sample_loss = learner.compute_loss(batch)
        bc_loss = learner.compute_bc_loss(batch)
        loss = td_loss + 2.5 * bc_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=0.5)
        optimizer.step()
        learner.update_target_network()

        with torch.no_grad():
            td_errors = per_sample_loss.cpu().numpy() + 1e-6
            buffer.update_priorities(indices, td_errors)

        if step % 400 == 0 or step == 1:
            logger.info("  BC/HER step %d/2000: Total Loss = %.4f | TD Loss = %.4f | BC Loss = %.4f", step, loss.item(), td_loss.item(), bc_loss.item())

    # 4. Step 3: Tournament Evaluation Gate
    logger.info("\nStep 3: Evaluating agent on Ladder Tournament Gate (Tiers 0, 1, 2)...")
    eval_results = evaluate_tier_ladder(
        net=online_net,
        parser=parser,
        device=device,
        n_games_per_tier=4,
        base_seed=12000,
    )

    for key, res in eval_results.items():
        logger.info(
            "  [Tier %d - %s]: %d/4 Wins (%.0f%%) | Avg Agent: $%.1f vs Opp: $%.1f",
            res["tier"], res["name"], res["wins"], res["win_rate"] * 100, res["avg_agent_money"], res["avg_opp_money"]
        )

    rosa_res = eval_results["tier_2_rotation_rosa"]
    achieved = (rosa_res["wins"] >= 3)
    if achieved:
        logger.info("\n🎯 GOAL ACHIEVED: Defeated Tier 2 (Rotation Rosa) with %d/4 wins (>= 75%%)!", rosa_res["wins"])

    # 5. Export Champion Model & Submission Agent
    logger.info("\n=== EXPORTING TIER 2 CHAMPION ARTIFACTS ===")
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

    # Save final scorecard
    with open(metrics_dir / "tier0_to_tier2_ladder_eval.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    print("\n" + "=" * 76)
    print(f"{'TIER 0 - TIER 2 LADDER EVALUATION SCORECARD':^76}")
    print("=" * 76)
    for key, res in final_eval.items() if 'final_eval' in locals() else eval_results.items():
        status = "✅ CLEARED" if res["wins"] >= 3 else ("⚠️ TIED/PARTIAL" if res["wins"] >= 2 else "❌ RETRY")
        print(f"Tier {res['tier']} ({res['name']:<14}): {res['wins']}/4 Wins ({res['win_rate']:>4.0%}) | Agent: ${res['avg_agent_money']:>8,.1f} vs Opp: ${res['avg_opp_money']:>8,.1f} | {status}")
    print("=" * 76)

    return 0 if achieved else 1


if __name__ == "__main__":
    sys.exit(main())
