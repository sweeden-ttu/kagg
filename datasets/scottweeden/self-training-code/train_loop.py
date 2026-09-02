"""Core self-play training loop.

Extracted from the monolithic kaggriculture_self_play_training.py.
Provides run_self_play_training() — the episode-level training loop.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment import create_competitive_env
from kaggriculture_path_b_rebuild import (
    HierarchicalActionMasker,
    apply_hierarchical_masks,
    break_pass_spawn_deadlock,
    prefer_farm_invest_actions,
)
from kaggriculture_adapter import (
    COMPETITION_TURNS_PER_DAY,
    EPISODE_STEPS,
    decode_path_b_action,
    parse_observation,
)
from checkpoints import _training_state_path, save_training_state
from training_metrics import save_episode_metrics

logger = logging.getLogger(__name__)


def run_self_play_training(
    online_net: nn.Module,
    target_net: nn.Module,
    optimizer: optim.Optimizer,
    learner,  # HierarchicalDoubleDQNLearner
    reward_shaper,  # CompetitiveRewardShaper
    config: Dict[str, Any],
    dirs: Dict[str, Path],
    progress,  # TrainingProgressRecorder
    buffer,  # PrioritizedReplayBuffer
    coordinator,  # SelfPlayCoordinator
    parser,  # KaggricultureJSONParser
    seed: int,
    total_episodes: int,
    learning_start_episodes: int,
    batch_size: int,
    checkpoint_interval: int,
    use_kaggle_env: bool,
    max_episode_steps: int,
    turns_per_cycle: int,
    n_eval_episodes: int,
    device,  # torch.device
    device_name: str,
    verbose: bool,
) -> List[Dict[str, float]]:
    """Run the core self-play training loop.

    Parameters are the same components that train_self_play() creates.
    Returns the accumulated episode_metrics list.
    """
    # Exploration parameter decay.
    eps_start = 0.12  # BC was already run before this is called
    eps_end = 0.03
    eps_decay_steps = max(1, total_episodes - learning_start_episodes)
    logger.info(
        "Self-play eps_start=%.2f (BC already completed)",
        eps_start,
    )

    # Initialize competitive self-play environment (official Kaggle simulator required).
    env = create_competitive_env(
        use_kaggle=use_kaggle_env,
        max_steps=max_episode_steps if use_kaggle_env else min(50, max_episode_steps),
        seed=seed,
        turns_per_cycle=turns_per_cycle,
    )
    if verbose:
        env_name = type(env).__name__
        logger.debug(
            "Self-play env: %s max_steps=%d turns_per_cycle=%d use_kaggle=%s",
            env_name,
            max_episode_steps if use_kaggle_env else min(50, max_episode_steps),
            turns_per_cycle,
            use_kaggle_env,
        )

    logger.info(
        "--- BEGINNING KAGGRICULTURE SELF-PLAY PIPELINE (episodes %d → %d) ---",
        1,
        total_episodes,
    )

    episode_metrics: List[Dict[str, float]] = []

    for ep in range(1, total_episodes + 1):
        # 1. Selection of Self-Play opponent agent
        opp_path = coordinator.select_opponent()
        opp_agent_fn = coordinator.get_agent_policy_fn(opp_path, online_net, device)

        # Calculate active Epsilon for current episode exploration
        if ep <= learning_start_episodes:
            eps = eps_start
        else:
            steps_into_decay = ep - learning_start_episodes
            eps = max(eps_end, eps_start - steps_into_decay * (eps_start - eps_end) / eps_decay_steps)

        if verbose:
            logger.debug(
                "=== Episode %d/%d | opponent=%s | eps=%.3f | buffer=%d ===",
                ep,
                total_episodes,
                opp_path or "online-self",
                eps,
                len(buffer),
            )

        # 2. Reset Environment
        obs_p0 = env.reset()
        obs_p1 = env._get_obs(player=1)
        reward_shaper.reset_episode()
        done = False

        ep_shaped_reward = 0.0
        ep_raw_reward = 0.0
        ep_loss_sum = 0.0
        ep_gradient_updates = 0
        loss_history = []
        step_num = 0

        # Run Episode step loop
        while not done:
            # Player 0 (Online Agent) Decision Making
            parsed_p0 = parser.parse_observation(obs_p0)

            # Format inputs as PyTorch Tensors
            tiles_t = torch.as_tensor(parsed_p0["tiles"], dtype=torch.float32, device=device).unsqueeze(0)
            numeric_t = torch.as_tensor(parsed_p0["numeric"], dtype=torch.float32, device=device).unsqueeze(0)

            # Apply dynamic action masks to prevent invalid commands
            masks = HierarchicalActionMasker.get_dynamic_masks(obs_p0)

            if verbose and step_num == 0:
                logger.debug(
                    "Ep %d step 0 obs: tiles=%s numeric=%s valid_verbs=%d valid_crops=%d valid_market=%d",
                    ep,
                    tuple(tiles_t.shape),
                    tuple(numeric_t.shape),
                    int(masks["farmer_verb"].sum()) if "farmer_verb" in masks else -1,
                    int(masks["crop_parameter"].sum()) if "crop_parameter" in masks else -1,
                    int(masks["market"].sum()) if "market" in masks else -1,
                )

            # Epsilon-Greedy choice over hierarchical streams
            if random.random() < eps:
                # Select random actions matching dynamic mask indices
                v_valid_idxs = np.where(masks["farmer_verb"])[0]
                verb_idx = random.choice(v_valid_idxs) if len(v_valid_idxs) > 0 else 0

                c_valid_idxs = np.where(masks["crop_parameter"])[0]
                crop_idx = random.choice(c_valid_idxs) if len(c_valid_idxs) > 0 else 0

                hands_indices = [random.randint(0, 14) for _ in range(online_net.num_hands)]
                market_indices = []
                m_valid_idxs = np.where(masks["market"])[0]
                for _ in range(online_net.max_market_orders):
                    market_indices.append(random.choice(m_valid_idxs) if len(m_valid_idxs) > 0 else 0)
            else:
                # Action selection must happen in EVAL mode to avoid BatchNorm size 1 issues
                online_net.eval()
                with torch.no_grad():
                    q_out = online_net(tiles_t, numeric_t)
                    masked_q = apply_hierarchical_masks(q_out, masks, device)
                    masked_q["farmer_verb"] = break_pass_spawn_deadlock(
                        masked_q["farmer_verb"], masks["farmer_verb"]
                    )
                    farm_verb, farm_market = prefer_farm_invest_actions(
                        masked_q["farmer_verb"],
                        masks["farmer_verb"],
                        masked_q["market"],
                        masks.get("market"),
                        observation=obs_p0,
                    )
                    masked_q["farmer_verb"] = farm_verb
                    if farm_market is not None:
                        masked_q["market"] = farm_market

                    verb_idx = int(masked_q["farmer_verb"].argmax(dim=-1).item())
                    crop_idx = int(masked_q["crop_parameter"].argmax(dim=-1).item())

                    hands_indices = []
                    for h_i in range(online_net.num_hands):
                        hands_indices.append(int(masked_q["hands"][h_i].argmax(dim=-1).item()))

                    market_indices = []
                    market_seq_argmax = masked_q["market"].argmax(dim=-1).squeeze(0) # (max_market_orders,)
                    for step_i in range(online_net.max_market_orders):
                        market_indices.append(int(market_seq_argmax[step_i].item()))

            # Translate decision to Kaggriculture environment Command Dictionary
            act_p0 = decode_path_b_action(
                verb_idx, crop_idx, hands_indices, market_indices, obs_p0
            )

            # Player 1 (Opponent Agent) chooses policy
            act_p1 = opp_agent_fn(obs_p1)

            # 3. Environment Step Execution
            (next_obs_p0, next_obs_p1), rewards, done, _ = env.step([act_p0, act_p1])

            raw_reward_p0 = rewards[0]
            shaped_reward_p0 = reward_shaper.shape_reward(
                obs_p0,
                raw_reward_p0,
                action_verb=verb_idx,
                action_market=market_indices,
                action_hands=hands_indices,
            )

            ep_raw_reward += raw_reward_p0
            ep_shaped_reward += shaped_reward_p0

            # Format next state observations
            parsed_next_p0 = parser.parse_observation(next_obs_p0)

            # 4. Save Transition into Prioritized Experience Replay Buffer
            buffer.push(
                tiles=parsed_p0["tiles"],
                numeric=parsed_p0["numeric"],
                action_verb=verb_idx,
                action_crop=crop_idx,
                action_hands=np.array(hands_indices, dtype=np.int64),
                action_market=np.array(market_indices, dtype=np.int64),
                reward=shaped_reward_p0,
                next_tiles=parsed_next_p0["tiles"],
                next_numeric=parsed_next_p0["numeric"],
                done=done
            )

            # Shift state reference
            obs_p0 = next_obs_p0
            obs_p1 = next_obs_p1

            if verbose and (step_num < 3 or step_num % 100 == 0):
                logger.debug(
                    "Ep %d step %d: verb=%d crop=%d raw_r=%.3f shaped_r=%.3f done=%s buffer=%d learn=%s",
                    ep,
                    step_num,
                    verb_idx,
                    crop_idx,
                    raw_reward_p0,
                    shaped_reward_p0,
                    done,
                    len(buffer),
                    ep > learning_start_episodes and len(buffer) >= batch_size,
                )

            step_num += 1

            # 5. Optimize Model on batches from PER Buffer (switched to TRAIN mode)
            if ep > learning_start_episodes and len(buffer) >= batch_size:
                online_net.train() # Enable training mode for BatchNorm updates
                batch, indices, weights = buffer.sample(batch_size)
                # Move batch to device
                for k in batch:
                    batch[k] = batch[k].to(device)

                loss, per_sample_loss = learner.compute_loss(batch)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=0.5)
                optimizer.step()
                learner.update_target_network()
                loss_history.append(loss.item())
                ep_loss_sum += float(loss.item())
                ep_gradient_updates += 1

                if verbose and (step_num <= 3 or step_num % 100 == 0):
                    logger.debug(
                        "Ep %d step %d DQN update: loss=%.5f batch=%d priorities_updated=%d",
                        ep,
                        step_num,
                        loss.item(),
                        batch_size,
                        len(indices),
                    )

                with torch.no_grad():
                    td_errors = per_sample_loss.cpu().numpy() + 1e-6
                    buffer.update_priorities(indices, td_errors)
                    if ep_gradient_updates == 1:
                        logger.info(
                            "PER: TD-error priority reweighting applied "
                            "(n=%d td_mean=%.5f td_max=%.5f)",
                            len(indices),
                            float(np.mean(td_errors)),
                            float(np.max(td_errors)),
                        )

        # Performance Monitoring
        avg_loss = np.mean(loss_history) if loss_history else 0.0
        logger.info(
            "Episode %02d/%02d | Epsilon: %.3f | Buffer Size: %d | Raw Reward: %.2f | "
            "Shaped Reward: %.2f | Avg Loss: %.5f",
            ep, total_episodes, eps, len(buffer), ep_raw_reward, ep_shaped_reward, avg_loss,
        )

        # Record match results vs reference ladder opponents for curriculum promotion
        if opp_path and opp_path.endswith(".py"):
            farms_p0 = obs_p0.get("farms", []) if isinstance(obs_p0, dict) else []
            farms_p1 = obs_p1.get("farms", []) if isinstance(obs_p1, dict) else []
            p0_m = float(farms_p0[0].get("money", 0.0)) if len(farms_p0) > 0 else 0.0
            p1_m = float(farms_p1[1].get("money", 0.0)) if len(farms_p1) > 1 else 0.0
            won = p0_m > p1_m
            coordinator.record_match_result(opp_path, won, p0_m, p1_m)
            opp_name = Path(opp_path).name
            cur_tier_slug = (
                coordinator.ladder_opponents[coordinator.current_tier_idx][0]
                if coordinator.ladder_opponents else "N/A"
            )
            logger.info(
                "Curriculum match vs %s: %s (P0 $%.1f vs P1 $%.1f) | Active Target: %s",
                opp_name,
                "WIN" if won else ("TIE" if p0_m == p1_m else "LOSS"),
                p0_m,
                p1_m,
                cur_tier_slug,
            )
        ep_row = {
            "episode": ep,
            "epsilon": eps,
            "buffer_size": len(buffer),
            "bootstrap_buffer_size": getattr(buffer, "bootstrap_size", len(buffer)),
            "selfplay_buffer_size": getattr(buffer, "selfplay_size", 0),
            "steps": step_num,
            "raw_reward": ep_raw_reward,
            "shaped_reward": ep_shaped_reward,
            "avg_loss": avg_loss,
            "loss_sum": ep_loss_sum,
            "gradient_updates": ep_gradient_updates,
        }
        episode_metrics.append(ep_row)
        progress.record_self_play_episode(ep_row)
        save_episode_metrics(dirs["root"], episode_metrics)

        # Periodically save models, append to Self-Play pool, and persist full training state
        if ep % checkpoint_interval == 0 or ep == total_episodes:
            online_net.eval()
            coordinator.save_checkpoint(online_net, episode=ep)
            save_training_state(
                _training_state_path(dirs["root"]),
                last_completed_episode=ep,
                online_net=online_net,
                target_net=target_net,
                optimizer=optimizer,
                buffer=buffer,
                coordinator=coordinator,
                episode_metrics=episode_metrics,
                config={**config, "last_completed_episode": ep},
            )

    with open(dirs["root"] / "config.json", "w", encoding="utf-8") as fh:
        json.dump({**config, "last_completed_episode": total_episodes}, fh, indent=2)

    final_model_path = dirs["models"] / "model.pth"
    torch.save(online_net.state_dict(), final_model_path)
    logger.info("Final model saved to: %s", final_model_path)

    return episode_metrics
