#!/usr/bin/env python3
"""Run ladder opponent matches, analyze failure modes, and seed the replay buffer."""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CODE_SRC = ROOT / "datasets" / "scottweeden" / "self-training-code"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE_SRC))

from eval import (
    _make_kaggle_env,
    _normalize_states,
    discover_reference_opponents,
    load_experiment_agent_policy,
    load_kaggle_agent_policy,
)
from kaggriculture_adapter import (
    CROPS,
    NUM_HANDS,
    NUM_MARKET_ACTIONS,
    encode_path_b_action,
    encode_path_b_observation,
    parse_observation,
)
from kaggriculture_path_b_rebuild import (
    CompetitiveRewardShaper,
    HierarchicalDQNBranching,
    HierarchicalDoubleDQNLearner,
    KaggricultureFeatureExtractor,
    KaggricultureJSONParser,
)
from replay_buffer import (
    SOURCE_BOOTSTRAP,
    SOURCE_SELFPLAY,
    PrioritizedReplayBuffer,
)
from agent_coordinator import SelfPlayCoordinator
from checkpoints import (
    load_training_state,
    save_training_state,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("seed_and_analyze")


def run_detailed_match(
    env: Any,
    policy_p0: Any,
    policy_p1: Any,
    opp_name: str,
    max_steps: int = 720,
) -> Dict[str, Any]:
    """Run one episode and record detailed step telemetry."""
    states = _normalize_states(env.reset())
    obs_p0 = parse_observation(states[0], player_id=0)
    obs_p1 = parse_observation(states[1], player_id=1)

    steps_data: List[Dict[str, Any]] = []
    p0_actions: List[Dict[str, Any]] = []
    p1_actions: List[Dict[str, Any]] = []
    p0_wealth_history: List[float] = []
    p1_wealth_history: List[float] = []

    p0_action_counts = Counter()
    p1_action_counts = Counter()
    p0_market_counts = Counter()
    p1_market_counts = Counter()

    p0_redundant_plants = 0
    p0_pass_turns = 0
    p0_first_seed_turn = None
    p0_first_harvest_turn = None
    p0_first_sell_turn = None

    steps = 0
    done = False

    while not done and steps < max_steps:
        raw_state_0 = states[0]
        raw_state_1 = states[1]

        act_p0 = policy_p0(obs_p0)
        act_p1 = policy_p1(obs_p1)

        p0_actions.append(act_p0)
        p1_actions.append(act_p1)

        m0 = float(obs_p0.get("farms", [{}])[0].get("money", 0))
        m1 = float(obs_p1.get("farms", [{}, {}])[1].get("money", 0))
        p0_wealth_history.append(m0)
        p1_wealth_history.append(m1)

        # Classify P0 actions
        f0 = act_p0.get("farmer", ["PASS"])
        f0_verb = f0[0] if isinstance(f0, list) and len(f0) > 0 else str(f0)
        p0_action_counts[f0_verb] += 1
        if f0_verb == "PASS":
            p0_pass_turns += 1
        elif f0_verb == "PLANT":
            # check if tile was already planted
            my_farm = obs_p0.get("farms", [{}])[0]
            fx, fy = my_farm.get("farmer", [0, 0])
            tiles = my_farm.get("tiles", [])
            if 0 <= fy < len(tiles) and 0 <= fx < len(tiles[fy]):
                tile = tiles[fy][fx]
                if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                    p0_redundant_plants += 1
        elif f0_verb == "HARVEST":
            if p0_first_harvest_turn is None:
                p0_first_harvest_turn = steps

        for m_order in act_p0.get("market", []):
            if isinstance(m_order, list) and m_order:
                verb = m_order[0]
                p0_market_counts[verb] += 1
                if verb == "BUY_SEED" and p0_first_seed_turn is None:
                    p0_first_seed_turn = steps
                elif verb == "SELL" and p0_first_sell_turn is None:
                    p0_first_sell_turn = steps

        # Classify P1 actions
        f1 = act_p1.get("farmer", ["PASS"])
        f1_verb = f1[0] if isinstance(f1, list) and len(f1) > 0 else str(f1)
        p1_action_counts[f1_verb] += 1
        for m_order in act_p1.get("market", []):
            if isinstance(m_order, list) and m_order:
                p1_market_counts[m_order[0]] += 1

        steps_data.append({
            "step": steps,
            "obs_p0": obs_p0,
            "obs_p1": obs_p1,
            "raw_state_0": raw_state_0,
            "raw_state_1": raw_state_1,
            "act_p0": act_p0,
            "act_p1": act_p1,
        })

        states = _normalize_states(env.step([act_p0, act_p1]))
        obs_p0 = parse_observation(states[0], player_id=0)
        obs_p1 = parse_observation(states[1], player_id=1)
        status = states[0].get("status", "ACTIVE")
        done = status in ("DONE", "TIMEOUT", "INVALID")
        steps += 1

    # Record final state
    steps_data.append({
        "step": steps,
        "obs_p0": obs_p0,
        "obs_p1": obs_p1,
        "raw_state_0": states[0],
        "raw_state_1": states[1],
        "act_p0": {"farmer": ["PASS"], "hands": [], "market": []},
        "act_p1": {"farmer": ["PASS"], "hands": [], "market": []},
    })
    final_m0 = float(obs_p0.get("farms", [{}])[0].get("money", 0))
    final_m1 = float(obs_p1.get("farms", [{}, {}])[1].get("money", 0))
    p0_wealth_history.append(final_m0)
    p1_wealth_history.append(final_m1)

    winner = "P0 (Agent)" if final_m0 > final_m1 else ("P1 (Opponent)" if final_m1 > final_m0 else "Tie")

    return {
        "opp_name": opp_name,
        "steps": steps,
        "final_money_p0": final_m0,
        "final_money_p1": final_m1,
        "winner": winner,
        "p0_won": final_m0 > final_m1,
        "p0_wealth_history": p0_wealth_history,
        "p1_wealth_history": p1_wealth_history,
        "p0_action_counts": dict(p0_action_counts),
        "p1_action_counts": dict(p1_action_counts),
        "p0_market_counts": dict(p0_market_counts),
        "p1_market_counts": dict(p1_market_counts),
        "p0_redundant_plants": p0_redundant_plants,
        "p0_pass_turns": p0_pass_turns,
        "p0_first_seed_turn": p0_first_seed_turn,
        "p0_first_harvest_turn": p0_first_harvest_turn,
        "p0_first_sell_turn": p0_first_sell_turn,
        "steps_data": steps_data,
    }


def extract_transitions_from_game(
    game: Dict[str, Any],
    reward_shaper: CompetitiveRewardShaper,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert steps_data into Path B transition dicts for both P0 and P1."""
    steps_data = game["steps_data"]
    p0_transitions: List[Dict[str, Any]] = []
    p1_transitions: List[Dict[str, Any]] = []

    reward_shaper.reset_episode()

    for idx in range(len(steps_data) - 1):
        cur = steps_data[idx]
        nxt = steps_data[idx + 1]
        is_done = (idx == len(steps_data) - 2)

        # P0 perspective
        obs_p0 = cur["obs_p0"]
        next_obs_p0 = nxt["obs_p0"]
        act_p0 = cur["act_p0"]

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

        p0_transitions.append({
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
        })

        # P1 perspective (Opponent Expert Trajectory)
        obs_p1 = cur["obs_p1"]
        next_obs_p1 = nxt["obs_p1"]
        act_p1 = cur["act_p1"]

        enc_p1 = encode_path_b_observation(obs_p1, player_id=1)
        enc_next_p1 = encode_path_b_observation(next_obs_p1, player_id=1)
        act_enc_p1 = encode_path_b_action(act_p1, max_market_orders=10)

        prev_m1 = float(obs_p1.get("farms", [{}, {}])[1].get("money", 0)) if len(obs_p1.get("farms", [])) > 1 else 0.0
        cur_m1 = float(next_obs_p1.get("farms", [{}, {}])[1].get("money", 0)) if len(next_obs_p1.get("farms", [])) > 1 else 0.0
        opp_m1 = float(next_obs_p1.get("farms", [{}])[0].get("money", 0))

        # Shape reward for P1
        r1 = (cur_m1 - prev_m1) / 100.0
        if cur_m1 > opp_m1:
            r1 += 0.5
        elif cur_m1 < opp_m1:
            r1 -= 0.5

        p1_transitions.append({
            "tiles": enc_p1["tiles"],
            "numeric": enc_p1["numeric"],
            "action_verb": int(act_enc_p1["verb"]),
            "action_crop": int(act_enc_p1["crop"]),
            "action_hands": act_enc_p1["hands"],
            "action_market": act_enc_p1["market"],
            "reward": float(r1),
            "next_tiles": enc_next_p1["tiles"],
            "next_numeric": enc_next_p1["numeric"],
            "next_masks": get_action_masks(next_obs_p1),
            "done": is_done,
        })

    return p0_transitions, p1_transitions


def main() -> None:
    agent_artifacts_dir = CODE_SRC / "training_artifacts"
    opponents_dir = ROOT / "opponents"
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading agent policy from %s...", agent_artifacts_dir)
    policy_p0 = load_experiment_agent_policy(agent_artifacts_dir, repo_root=ROOT)

    logger.info("Discovering reference opponents in %s...", opponents_dir)
    opponents = discover_reference_opponents(opponents_dir)
    logger.info("Found %d reference opponents: %s", len(opponents), [name for name, _ in opponents])

    reward_shaper = CompetitiveRewardShaper(KaggricultureJSONParser())

    all_games: List[Dict[str, Any]] = []
    all_p0_transitions: List[Dict[str, Any]] = []
    all_p1_expert_transitions: List[Dict[str, Any]] = []

    summary_rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 80)
    print(f"{'OPPONENT MATCH EVALUATION & TELEMETRY COLLECTION':^80}")
    print("=" * 80)

    for opp_idx, (opp_name, opp_fn) in enumerate(opponents):
        logger.info("[%d/%d] Playing 720-step match vs %s...", opp_idx + 1, len(opponents), opp_name)
        env = _make_kaggle_env(max_steps=720, seed=42 + opp_idx, turns_per_day=24)
        game_res = run_detailed_match(env, policy_p0, opp_fn, opp_name, max_steps=720)
        all_games.append(game_res)

        p0_trans, p1_trans = extract_transitions_from_game(game_res, reward_shaper)
        all_p0_transitions.extend(p0_trans)
        all_p1_expert_transitions.extend(p1_trans)

        summary_rows.append({
            "opponent": opp_name,
            "p0_money": game_res["final_money_p0"],
            "p1_money": game_res["final_money_p1"],
            "winner": game_res["winner"],
            "p0_pass_turns": game_res["p0_pass_turns"],
            "p0_redundant_plants": game_res["p0_redundant_plants"],
            "p0_first_seed": game_res["p0_first_seed_turn"],
            "p0_first_harvest": game_res["p0_first_harvest_turn"],
            "p0_first_sell": game_res["p0_first_sell_turn"],
            "p0_actions": game_res["p0_action_counts"],
            "p1_actions": game_res["p1_action_counts"],
            "p0_market": game_res["p0_market_counts"],
            "p1_market": game_res["p1_market_counts"],
        })

        print(
            f"vs {opp_name:16s} | Agent: ${game_res['final_money_p0']:>7.0f} vs Opp: ${game_res['final_money_p1']:>7.0f} | "
            f"Winner: {game_res['winner']:<14s} | P0 Pass: {game_res['p0_pass_turns']:>3d}/720 | "
            f"P0 Redundant Plant: {game_res['p0_redundant_plants']:>3d} | 1st Seed: {str(game_res['p0_first_seed_turn']):>4s}"
        )

    print("=" * 80 + "\n")

    # ── SEED REPLAY BUFFER ────────────────────────────────────────────────
    logger.info("Initializing / Loading Replay Buffer to seed...")
    ckpt_path = agent_artifacts_dir / "checkpoints" / "training_state_latest.pt"

    device = torch.device("cpu")
    parser = KaggricultureJSONParser()
    extractor = KaggricultureFeatureExtractor(latent_dim=512)
    online_net = HierarchicalDQNBranching(extractor, latent_dim=512, shared_dim=256)
    target_net = HierarchicalDQNBranching(extractor, latent_dim=512, shared_dim=256)
    optimizer = torch.optim.Adam(online_net.parameters(), lr=1e-4)
    coordinator = SelfPlayCoordinator()
    buffer = PrioritizedReplayBuffer(capacity=200_000, alpha=0.6, bootstrap_fraction=0.5)

    last_completed_ep = 25
    episode_metrics = []
    config = {}

    if ckpt_path.exists():
        try:
            last_completed_ep, episode_metrics, config = load_training_state(
                ckpt_path,
                device=device,
                online_net=online_net,
                target_net=target_net,
                optimizer=optimizer,
                buffer=buffer,
                coordinator=coordinator,
            )
            logger.info("Loaded existing checkpoint (ep %d, buffer size %d)", last_completed_ep, len(buffer))
        except Exception as exc:
            logger.warning("Could not load full training state (%s); starting fresh buffer", exc)
            model_pth = agent_artifacts_dir / "models" / "model.pth"
            if model_pth.exists():
                online_net.load_state_dict(torch.load(model_pth, map_location=device))
                target_net.load_state_dict(online_net.state_dict())

    initial_bootstrap_size = buffer.bootstrap_size
    initial_selfplay_size = buffer.selfplay_size

    # Push expert opponent transitions into SOURCE_BOOTSTRAP partition
    logger.info("Seeding %d expert opponent transitions into SOURCE_BOOTSTRAP...", len(all_p1_expert_transitions))
    for t in all_p1_expert_transitions:
        buffer.push(
            tiles=t["tiles"],
            numeric=t["numeric"],
            action_verb=t["action_verb"],
            action_crop=t["action_crop"],
            action_hands=t["action_hands"],
            action_market=t["action_market"],
            reward=t["reward"],
            next_tiles=t["next_tiles"],
            next_numeric=t["next_numeric"],
            done=t["done"],
            source=SOURCE_BOOTSTRAP,
        )

    # Push agent transitions into SOURCE_SELFPLAY partition
    logger.info("Seeding %d agent gameplay transitions into SOURCE_SELFPLAY...", len(all_p0_transitions))
    for t in all_p0_transitions:
        buffer.push(
            tiles=t["tiles"],
            numeric=t["numeric"],
            action_verb=t["action_verb"],
            action_crop=t["action_crop"],
            action_hands=t["action_hands"],
            action_market=t["action_market"],
            reward=t["reward"],
            next_tiles=t["next_tiles"],
            next_numeric=t["next_numeric"],
            done=t["done"],
            source=SOURCE_SELFPLAY,
        )

    logger.info(
        "Replay Buffer seeded: Bootstrap %d -> %d | Selfplay %d -> %d | Total: %d transitions",
        initial_bootstrap_size,
        buffer.bootstrap_size,
        initial_selfplay_size,
        buffer.selfplay_size,
        len(buffer),
    )

    # Test sampling from seeded buffer
    sample_batch, sample_indices, sample_weights = buffer.sample(batch_size=64, beta=0.4)
    logger.info(
        "Buffer sample test OK: batch tiles shape %s, numeric shape %s, %d indices sampled",
        sample_batch["tiles"].shape,
        sample_batch["numeric"].shape,
        len(sample_indices),
    )

    # Save seeded checkpoint
    seeded_ckpt_path = agent_artifacts_dir / "checkpoints" / "training_state_latest.pt"
    backup_ckpt_path = agent_artifacts_dir / "checkpoints" / "training_state_seeded_backup.pt"
    config["last_completed_episode"] = last_completed_ep
    config["opponent_matches_seeded"] = len(all_games)
    config["seeded_transitions_count"] = len(all_p1_expert_transitions) + len(all_p0_transitions)

    save_training_state(
        seeded_ckpt_path,
        last_completed_episode=last_completed_ep,
        online_net=online_net,
        target_net=target_net,
        optimizer=optimizer,
        buffer=buffer,
        coordinator=coordinator,
        episode_metrics=episode_metrics,
        config=config,
    )
    save_training_state(
        backup_ckpt_path,
        last_completed_episode=last_completed_ep,
        online_net=online_net,
        target_net=target_net,
        optimizer=optimizer,
        buffer=buffer,
        coordinator=coordinator,
        episode_metrics=episode_metrics,
        config=config,
    )
    logger.info("Saved updated training state to %s and %s", seeded_ckpt_path, backup_ckpt_path)

    # Save reports
    analysis_report = {
        "n_opponents": len(opponents),
        "total_transitions_seeded": len(all_p1_expert_transitions) + len(all_p0_transitions),
        "bootstrap_transitions_seeded": len(all_p1_expert_transitions),
        "selfplay_transitions_seeded": len(all_p0_transitions),
        "buffer_final_stats": {
            "total_size": len(buffer),
            "bootstrap_size": buffer.bootstrap_size,
            "selfplay_size": buffer.selfplay_size,
            "capacity": buffer.capacity,
        },
        "match_summaries": summary_rows,
    }

    report_path = results_dir / "opponent_failure_analysis.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(analysis_report, fh, indent=2)
    logger.info("Analysis report saved to %s", report_path)


if __name__ == "__main__":
    main()
