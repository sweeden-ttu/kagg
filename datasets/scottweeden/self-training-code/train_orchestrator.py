"""Kaggriculture self-play training orchestration.

This module replaces the monolithic train_self_play() from the original
kaggriculture_self_play_training.py. It coordinates:
  1. Experiment directory setup
  2. Model component creation
  3. Resume logic
  4. Bootstrap data loading
  5. BC pretrain
  6. Self-play training loop
  7. Post-training evaluation
  8. Agent export
  9. Metrics updates
  10. Optional publishing

All implementation logic is delegated to separate SRP modules.
"""

from __future__ import annotations

import json
import logging
import sys
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import from SRP modules
from environment import create_competitive_env
from replay_buffer import PrioritizedReplayBuffer
from agent_coordinator import SelfPlayCoordinator, setup_experiment_dirs
from checkpoints import (
    _load_state_dict,
    _training_state_path,
    _load_episode_metrics,
    _episode_from_checkpoint_name,
    resolve_resume_path,
    save_training_state,
    _coerce_cpu_byte_rng_state,
    _restore_rng_state,
    load_training_state,
    load_weights_checkpoint,
)
from _resolve_code_src import _resolve_code_src
from agent_export import _export_path_b_agent
from train_loop import run_self_play_training

# Import from external modules on PYTHONPATH
from kaggriculture_path_b_rebuild import (
    KaggricultureJSONParser,
    KaggricultureFeatureExtractor,
    HierarchicalDQNBranching,
    HierarchicalDoubleDQNLearner,
    CompetitiveRewardShaper,
)
from kaggriculture_adapter import resolve_training_device
from path_b_bootstrap import (
    bootstrap_path_b_replay_buffer,
    incremental_daily_bootstrap_bc,
    resolve_bootstrap_episode_files,
    run_bc_pretrain,
    stream_bootstrap_bc_pretrain,
)
from training_metrics import (
    TrainingProgressRecorder,
    merge_corpus_trends,
    save_episode_metrics,
    save_progress,
)
from eval_policy import (
    evaluate_ladder,
    resolve_opponents_dir,
    save_eval_report,
    win_rate_eval_from_ladder,
)

logger = logging.getLogger(__name__)


def train_self_play(
    total_episodes: int = 15,
    learning_start_episodes: int = 2,
    batch_size: int = 32,
    checkpoint_interval: int = 5,
    experiment_dir: str = "experiments/self_play",
    seed: int = 42,
    device_name: str = "auto",
    use_kaggle_env: bool = True,
    max_episode_steps: int = 720,
    turns_per_cycle: int = 24,
    n_eval_episodes: int = 10,
    resume: Optional[str] = None,
    bootstrap_episodes: Optional[int] = None,
    bootstrap_transitions: Optional[int] = 50_000,
    data_dir: str = "working/kaggle_episodes",
    download_bootstrap: bool = False,
    bc_epochs: int = 15,
    bc_batch_size: int = 64,
    bc_steps_per_epoch: Optional[int] = None,
    buffer_capacity: int = 10_000,
    metadata_path: Optional[str] = None,
    bootstrap_top_per_day: Optional[int] = 20,
    bootstrap_passes: int = 1,
    bc_epochs_per_pass: int = 2,
    verbose: bool = False,
    code_src: Optional[str] = None,
    bootstrap_mode: str = "daily_incremental",
    bootstrap_days_per_run: int = 3,
    publish_code_dataset: bool = False,
    opponents_dir: Optional[str] = None,
    ladder_eval_episodes: int = 0,
    ladder_win_rate_target: float = 0.75,
    min_self_play_episodes: int = 0,
):
    """Coordinate the full Kaggriculture self-play training pipeline.

    When ``resume`` is set, training continues from the last completed episode
    up to ``total_episodes`` (cumulative target, not additional episodes).
    If resume already meets ``total_episodes``, ``min_self_play_episodes`` extends
    the target so at least that many new self-play episodes still run.
    """
    resuming = resume is not None
    if resuming:
        exp_root, resume_file, resume_kind = resolve_resume_path(resume)
        experiment_dir = str(exp_root)
    else:
        resume_file = None
        resume_kind = ""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dirs = setup_experiment_dirs(Path(experiment_dir))
    config = {
        "experiment_dir": str(dirs["root"]),
        "seed": seed,
        "device": device_name,
        "total_episodes": total_episodes,
        "learning_start_episodes": learning_start_episodes,
        "batch_size": batch_size,
        "checkpoint_interval": checkpoint_interval,
        "use_kaggle_env": use_kaggle_env,
        "kinematic_phase_a": 3,
        "kinematic_phase_b": 4,
        "kinematic_phase_c": 6,
        "turns_per_cycle": turns_per_cycle,
        "cycles_per_episode": max(1, max_episode_steps // max(1, turns_per_cycle)),
        "max_episode_steps": max_episode_steps,
        "n_eval_episodes": n_eval_episodes,
        "bootstrap_episodes": bootstrap_episodes,
        "bootstrap_transitions": bootstrap_transitions,
        "data_dir": data_dir,
        "download_bootstrap": download_bootstrap,
        "bc_epochs": bc_epochs,
        "bc_batch_size": bc_batch_size,
        "bc_steps_per_epoch": bc_steps_per_epoch,
        "buffer_capacity": buffer_capacity,
        "metadata_path": metadata_path,
        "bootstrap_top_per_day": bootstrap_top_per_day,
        "bootstrap_passes": bootstrap_passes,
        "bc_epochs_per_pass": bc_epochs_per_pass,
        "verbose": verbose,
        "code_src": code_src,
        "bootstrap_mode": bootstrap_mode,
        "bootstrap_days_per_run": bootstrap_days_per_run,
        "publish_code_dataset": publish_code_dataset,
        "opponents_dir": opponents_dir,
        "ladder_eval_episodes": ladder_eval_episodes,
        "ladder_win_rate_target": ladder_win_rate_target,
        "min_self_play_episodes": min_self_play_episodes,
        "timestamp": datetime.now().isoformat(),
        "resumed_from": str(resume_file) if resuming else None,
    }

    log_path = dirs["logs"] / "self_play_training.log"
    log_mode = "a" if resuming else "w"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode=log_mode),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)
    if verbose:
        for module_name in (
            __name__,
            "path_b_bootstrap",
            "episode_catalog",
        ):
            logging.getLogger(module_name).setLevel(logging.DEBUG)
    logger.info("Experiment directory: %s", dirs["root"])
    progress = TrainingProgressRecorder(dirs["root"], resumed=resuming)
    if metadata_path and Path(metadata_path).exists():
        with open(metadata_path, encoding="utf-8") as fh:
            merge_corpus_trends(progress.state, json.load(fh))
        save_progress(dirs["root"], progress.state)
    if verbose:
        logger.debug("Verbose debug logging enabled")
        logger.debug("Training config: %s", json.dumps(config, indent=2, default=str))

    if device_name == "auto":
        device = resolve_training_device("auto")
    else:
        device = resolve_training_device(device_name)
    logger.info("Self-play pipeline running on device: %s", device)

    # Initialize components
    parser = KaggricultureJSONParser()
    extractor_online = KaggricultureFeatureExtractor(latent_dim=512)
    extractor_target = KaggricultureFeatureExtractor(latent_dim=512)

    online_net = HierarchicalDQNBranching(extractor_online, latent_dim=512, shared_dim=256).to(device)
    target_net = HierarchicalDQNBranching(extractor_target, latent_dim=512, shared_dim=256).to(device)

    if verbose:
        n_params = sum(p.numel() for p in online_net.parameters())
        logger.debug(
            "Model init: HierarchicalDQNBranching params=%d (%.2fM) buffer_capacity=%d",
            n_params,
            n_params / 1e6,
            buffer_capacity,
        )

    optimizer = optim.Adam(online_net.parameters(), lr=1e-4)
    learner = HierarchicalDoubleDQNLearner(
        online_net=online_net,
        target_net=target_net,
        optimizer=optimizer,
        gamma=0.995,
        tau=0.005
    )

    reward_shaper = CompetitiveRewardShaper(parser)
    config["reward_phase_gates"] = reward_shaper.phase_gates()
    logger.info("Reward phase gates: %s", config["reward_phase_gates"])
    with open(dirs["root"] / "config.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)
    coordinator = SelfPlayCoordinator(checkpoint_dir=str(dirs["checkpoints"]))

    start_episode = 0
    episode_metrics: List[Dict[str, float]] = []

    if resuming:
        assert resume_file is not None
        if resume_kind == "full":
            start_episode, episode_metrics, saved_config = load_training_state(
                resume_file, device, online_net, target_net, optimizer, buffer, coordinator
            )
            for key in ("seed", "use_kaggle_env", "max_episode_steps", "turns_per_cycle", "learning_start_episodes"):
                if key in saved_config:
                    saved_val = saved_config[key]
                    cli_val = config[key]
                    if saved_val != cli_val:
                        logger.warning(
                            "Resume config mismatch for %s: saved=%s cli=%s (using CLI value)",
                            key, saved_val, cli_val,
                        )
            logger.info(
                "Resumed full training state from %s (completed episode %d, buffer=%d, pool=%d)",
                resume_file, start_episode, len(buffer), len(coordinator.opponent_pool),
            )
        else:
            start_episode, episode_metrics = load_weights_checkpoint(
                resume_file, device, online_net, target_net, learner, coordinator, dirs["root"]
            )
            logger.warning(
                "Resumed weights only from %s (episode %d). Replay buffer and optimizer reset.",
                resume_file, start_episode,
            )
        config["last_completed_episode"] = start_episode
    else:
        online_net.eval()
        coordinator.save_checkpoint(online_net, episode=0)

    if start_episode >= total_episodes and min_self_play_episodes > 0:
        extended = start_episode + min_self_play_episodes
        logger.info(
            "Resume at episode %d already meets target %d; extending to %d (+ %d min self-play)",
            start_episode,
            total_episodes,
            extended,
            min_self_play_episodes,
        )
        total_episodes = extended
        config["total_episodes"] = total_episodes

    skip_bootstrap = (
        resuming
        and resume_kind == "full"
        and len(buffer) > 0
        and bootstrap_mode != "daily_incremental"
    )
    bootstrap_count = 0
    stream_stats: Dict[str, Any] = {}
    bc_loss_history: List[float] = []

    # ── Bootstrap phase ────────────────────────────────────────────────
    if bootstrap_episodes != 0 and not skip_bootstrap:
        if verbose:
            logger.debug(
                "Bootstrap phase: mode=%s episodes=%s passes=%d transitions=%s metadata=%s",
                bootstrap_mode,
                bootstrap_episodes,
                bootstrap_passes,
                bootstrap_transitions,
                metadata_path,
            )
        if bootstrap_mode == "daily_incremental":
            if not metadata_path:
                logger.warning("daily_incremental bootstrap requires metadata_path; skipping")
            else:
                stream_stats = incremental_daily_bootstrap_bc(
                    learner,
                    buffer,
                    device,
                    metadata_path=metadata_path,
                    experiment_root=dirs["root"],
                    days_per_run=bootstrap_days_per_run,
                    bc_epochs_per_day=bc_epochs_per_pass,
                    bc_batch_size=bc_batch_size,
                    bc_steps_per_epoch=bc_steps_per_epoch,
                    max_market_orders=online_net.max_market_orders,
                    random_seed=seed,
                    top_per_day=bootstrap_top_per_day,
                    verbose=verbose,
                )
                bootstrap_count = int(stream_stats.get("total_transitions_loaded", 0))
                bc_loss_history = list(stream_stats.get("epoch_losses", []))
                if stream_stats.get("new_days"):
                    config["bootstrap_days_this_run"] = stream_stats["new_days"]
                    config["bootstrapped_dates"] = stream_stats.get("bootstrapped_dates", [])
                # daily_incremental historically ignored bootstrap_passes; when >1,
                # run explicit corpus buffer BC passes so the knob is exercised.
                if bootstrap_passes > 1 and len(buffer) > 0 and bc_epochs_per_pass > 0:
                    logger.info(
                        "daily_incremental: %d corpus BC pass(es) on buffer (%d transitions)",
                        bootstrap_passes,
                        len(buffer),
                    )
                    corpus_pass_losses: List[float] = []
                    for pass_idx in range(1, bootstrap_passes + 1):
                        pass_losses = run_bc_pretrain(
                            learner,
                            buffer,
                            device,
                            epochs=bc_epochs_per_pass,
                            batch_size=bc_batch_size,
                            max_steps_per_epoch=bc_steps_per_epoch,
                            verbose=verbose,
                        )
                        corpus_pass_losses.extend(pass_losses)
                        logger.info(
                            "Bootstrap corpus pass %d/%d | epochs=%d | final_loss=%s",
                            pass_idx,
                            bootstrap_passes,
                            len(pass_losses),
                            f"{pass_losses[-1]:.5f}" if pass_losses else "n/a",
                        )
                    bc_loss_history.extend(corpus_pass_losses)
                    stream_stats["corpus_pass_losses"] = corpus_pass_losses
                    stream_stats["bootstrap_passes"] = bootstrap_passes
        elif bootstrap_passes > 1:
            episode_files = resolve_bootstrap_episode_files(
                data_dir=data_dir,
                max_episodes=bootstrap_episodes,
                download=download_bootstrap,
                metadata_path=metadata_path,
                top_per_day=bootstrap_top_per_day,
            )
            if not episode_files:
                logger.warning("Streaming bootstrap skipped: no episode files resolved")
            else:
                per_pass_cap = (
                    bootstrap_transitions if bootstrap_transitions is not None else buffer_capacity
                )
                if verbose:
                    logger.debug(
                        "Streaming bootstrap (by day): %d episode files, %d day passes",
                        len(episode_files),
                        bootstrap_passes,
                    )
                stream_stats = stream_bootstrap_bc_pretrain(
                    learner,
                    buffer,
                    device,
                    episode_files,
                    bootstrap_passes=bootstrap_passes,
                    max_transitions_per_pass=per_pass_cap,
                    bc_epochs_per_pass=bc_epochs_per_pass,
                    bc_batch_size=bc_batch_size,
                    bc_steps_per_epoch=bc_steps_per_epoch,
                    max_market_orders=online_net.max_market_orders,
                    random_seed=seed,
                    metadata_path=metadata_path,
                    experiment_root=dirs["root"],
                    verbose=verbose,
                )
                bootstrap_count = int(stream_stats.get("total_transitions_loaded", 0))
                bc_loss_history = list(stream_stats.get("epoch_losses", []))
        else:
            bootstrap_count = bootstrap_path_b_replay_buffer(
                buffer,
                data_dir=data_dir,
                max_episodes=bootstrap_episodes,
                max_transitions=bootstrap_transitions,
                max_market_orders=online_net.max_market_orders,
                random_seed=seed,
                download=download_bootstrap,
                metadata_path=metadata_path,
                top_per_day=bootstrap_top_per_day,
            )
        config["bootstrap_transitions_loaded"] = bootstrap_count
        logger.info("Replay buffer size after bootstrap: %d", len(buffer))
        bootstrap_meta = None
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, encoding="utf-8") as fh:
                bootstrap_meta = json.load(fh)
        progress.record_bootstrap_result(
            {
                **stream_stats,
                "bootstrap_transitions_loaded": bootstrap_count,
                "new_days": stream_stats.get("new_days") or config.get("bootstrap_days_this_run") or [],
            },
            bootstrap_meta,
        )
        save_progress(dirs["root"], progress.state)
    elif skip_bootstrap:
        logger.info(
            "Skipping bootstrap on full resume (buffer already has %d transitions)",
            len(buffer),
        )

    # ── BC pretrain ────────────────────────────────────────────────────
    if bc_epochs > 0 and bootstrap_passes <= 1:
        if len(buffer) == 0:
            logger.warning("BC pretrain requested but replay buffer is empty; skipping BC")
        else:
            bc_loss_history = run_bc_pretrain(
                learner,
                buffer,
                device,
                epochs=bc_epochs,
                batch_size=bc_batch_size,
                max_steps_per_epoch=bc_steps_per_epoch,
                verbose=verbose,
            )

    # ── BC metrics ─────────────────────────────────────────────────────
    if bc_loss_history:
        bc_metrics_path = dirs["metrics"] / "bc_pretrain.json"
        with open(bc_metrics_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "epochs": len(bc_loss_history),
                    "batch_size": bc_batch_size,
                    "bc_steps_per_epoch": bc_steps_per_epoch,
                    "buffer_size": len(buffer),
                    "buffer_capacity": buffer_capacity,
                    "bootstrap_transitions": bootstrap_count,
                    "metadata_path": metadata_path,
                    "bootstrap_passes": bootstrap_passes,
                    "bc_epochs_per_pass": bc_epochs_per_pass,
                    "bootstrap_mode": bootstrap_mode,
                    "bootstrap_days_per_run": bootstrap_days_per_run,
                    "bootstrapped_dates": stream_stats.get("bootstrapped_dates"),
                    "stream_stats": stream_stats,
                    "epoch_losses": bc_loss_history,
                    "final_loss": bc_loss_history[-1] if bc_loss_history else None,
                    "cumulative_bc_loss": float(sum(bc_loss_history)),
                },
                fh,
                indent=2,
            )
        logger.info("BC pretrain metrics saved to: %s", bc_metrics_path)

    # ── Save post-bootstrap state ──────────────────────────────────────
    new_bootstrap_days = stream_stats.get("new_days") or []
    if bootstrap_count > 0 or new_bootstrap_days:
        dirs["models"].mkdir(parents=True, exist_ok=True)
        torch.save(online_net.state_dict(), dirs["models"] / "model.pth")
        save_training_state(
            _training_state_path(dirs["root"]),
            last_completed_episode=start_episode,
            online_net=online_net,
            target_net=target_net,
            optimizer=optimizer,
            buffer=buffer,
            coordinator=coordinator,
            episode_metrics=episode_metrics,
            config={**config, "last_completed_episode": start_episode},
        )
        with open(dirs["root"] / "config.json", "w", encoding="utf-8") as fh:
            json.dump({**config, "last_completed_episode": start_episode}, fh, indent=2)

    # ── Publish code dataset (optional) ────────────────────────────────
    if publish_code_dataset and new_bootstrap_days:
        try:
            from kaggriculture_dataset_publish import publish_training_artifacts_to_code_dataset

            publish_summary = publish_training_artifacts_to_code_dataset(dirs["root"])
            config["code_dataset_publish"] = publish_summary
            logger.info(
                "Published code dataset after bootstrapping days %s",
                new_bootstrap_days,
            )
        except Exception as exc:
            logger.error("Code dataset publish failed: %s", exc)

    if start_episode >= total_episodes:
        logger.info(
            "Already completed %d episodes (target %d). Running export/eval only.",
            start_episode, total_episodes,
        )

    with open(dirs["root"] / "config.json", "w", encoding="utf-8") as fh:
        json.dump({**config, "last_completed_episode": start_episode}, fh, indent=2)

    # ── Exploration parameter decay ────────────────────────────────────
    if not bc_loss_history:
        bc_metrics_path = dirs["metrics"] / "bc_pretrain.json"
        if bc_metrics_path.exists():
            try:
                with open(bc_metrics_path, encoding="utf-8") as fh:
                    prior_bc = json.load(fh)
                prior_losses = prior_bc.get("epoch_losses") or []
                if prior_losses:
                    bc_loss_history = list(prior_losses)
                    logger.info(
                        "Using prior BC metrics (%d epoch losses) for low-ε self-play",
                        len(bc_loss_history),
                    )
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read prior BC metrics: %s", exc)

    # ── Run self-play training loop ────────────────────────────────────
    ep_metrics = run_self_play_training(
        online_net=online_net,
        target_net=target_net,
        optimizer=optimizer,
        learner=learner,
        reward_shaper=reward_shaper,
        config=config,
        dirs=dirs,
        progress=progress,
        buffer=buffer,
        coordinator=coordinator,
        parser=parser,
        seed=seed,
        total_episodes=total_episodes,
        learning_start_episodes=learning_start_episodes,
        batch_size=batch_size,
        checkpoint_interval=checkpoint_interval,
        use_kaggle_env=use_kaggle_env,
        max_episode_steps=max_episode_steps,
        turns_per_cycle=turns_per_cycle,
        n_eval_episodes=n_eval_episodes,
        device=device,
        device_name=device_name,
        verbose=verbose,
    )
    episode_metrics.extend(ep_metrics)

    # ── Post-training: ladder evaluation ───────────────────────────────
    from kaggriculture_adapter import COMPETITION_TURNS_PER_DAY, EPISODE_STEPS

    def _path_b_policy(obs):
        """Policy function for post-training evaluation."""
        from kaggriculture_path_b_rebuild import (
            HierarchicalActionMasker,
            apply_hierarchical_masks,
            break_pass_spawn_deadlock,
            prefer_farm_invest_actions,
        )
        from kaggriculture_adapter import decode_path_b_action

        online_net.eval()
        parsed = parser.parse_observation(obs)
        tiles_t = torch.as_tensor(parsed["tiles"], dtype=torch.float32, device=device).unsqueeze(0)
        numeric_t = torch.as_tensor(parsed["numeric"], dtype=torch.float32, device=device).unsqueeze(0)
        masks = HierarchicalActionMasker.get_dynamic_masks(obs)
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
                observation=obs,
            )
            masked_q["farmer_verb"] = farm_verb
            if farm_market is not None:
                masked_q["market"] = farm_market
            verb_idx = int(masked_q["farmer_verb"].argmax(dim=-1).item())
            crop_idx = int(masked_q["crop_parameter"].argmax(dim=-1).item())
            hands_indices = [
                int(masked_q["hands"][i].argmax(dim=-1).item())
                for i in range(online_net.num_hands)
            ]
            market_seq = masked_q["market"].argmax(dim=-1).squeeze(0)
            market_indices = [int(market_seq[t].item()) for t in range(online_net.max_market_orders)]
        return decode_path_b_action(verb_idx, crop_idx, hands_indices, market_indices, obs)

    episodes_per_opponent = (
        ladder_eval_episodes if ladder_eval_episodes > 0 else n_eval_episodes
    )
    if episodes_per_opponent > 0:
        opp_root = resolve_opponents_dir(opponents_dir, code_src=code_src)
        if opp_root is None:
            logger.warning(
                "League eval requested (%d ep/opponent) but opponents/ not found "
                "(opponents_dir=%r); skipping (no random baseline fallback)",
                episodes_per_opponent,
                opponents_dir,
            )
        else:
            try:
                ladder_report = evaluate_ladder(
                    _path_b_policy,
                    opponents_dir=str(opp_root),
                    code_src=code_src,
                    n_episodes=episodes_per_opponent,
                    max_steps=EPISODE_STEPS,
                    base_seed=seed + 2000,
                    win_rate_target=ladder_win_rate_target,
                    turns_per_day=COMPETITION_TURNS_PER_DAY,
                )
                ladder_path = dirs["metrics"] / "ladder_eval.json"
                with open(ladder_path, "w", encoding="utf-8") as fh:
                    json.dump(ladder_report, fh, indent=2)
                progress.record_eval("ladder", ladder_report)

                wr_summary = win_rate_eval_from_ladder(ladder_report)
                save_eval_report(wr_summary, dirs["metrics"] / "win_rate_eval.json")
                progress.record_eval("win_rate", wr_summary)
                save_progress(dirs["root"], progress.state)

                logger.info(
                    "League win-rate summary: %.2f (%d/%d ep), cleared=%s/%s, beats_all=%s",
                    wr_summary["win_rate"],
                    wr_summary["wins"],
                    wr_summary["n_episodes"],
                    wr_summary.get("opponents_cleared"),
                    wr_summary.get("n_opponents"),
                    wr_summary.get("beats_all_opponents"),
                )
                logger.info(
                    "Ladder eval (%d opponents, %d ep each, target %.0f%%): beats_all=%s → %s",
                    len(ladder_report.get("results", {})),
                    episodes_per_opponent,
                    ladder_win_rate_target * 100,
                    ladder_report.get("beats_all_opponents"),
                    ladder_path,
                )
                for slug, row in ladder_report.get("results", {}).items():
                    cleared = row.get("cleared", False)
                    mark = "PASS" if cleared else "FAIL"
                    logger.info(
                        "  [%s] vs %-16s win=%.0f%% (%d/%d) money %.0f vs %.0f",
                        mark,
                        slug,
                        row.get("win_rate", 0) * 100,
                        row.get("wins", 0),
                        row.get("n_episodes", 0),
                        row.get("avg_p0_money", 0),
                        row.get("avg_p1_money", 0),
                    )
            except Exception as exc:
                logger.warning("Ladder / league win-rate eval skipped: %s", exc)

    # ── Export agent ───────────────────────────────────────────────────
    agent_path = dirs["root"] / "agent.py"
    _export_path_b_agent(agent_path, dirs["root"], code_src=code_src)
    logger.info("Agent export saved to: %s", agent_path)

    # ── Save final metrics ─────────────────────────────────────────────
    metrics_path = save_episode_metrics(dirs["root"], episode_metrics)
    logger.info("Episode metrics saved to: %s", metrics_path)

    config["last_completed_episode"] = total_episodes
    progress_path = progress.finalize_run(config)
    logger.info("Cumulative training progress saved to: %s", progress_path)

    # ── Optional: update trend plots ───────────────────────────────────
    try:
        from visualize import update_experiment_plots

        plot_summary = update_experiment_plots(dirs["root"])
        logger.info(
            "Trend plots updated (%d files) → %s",
            len(plot_summary.get("plots_written") or []),
            plot_summary.get("plots_dir"),
        )
    except Exception as exc:
        logger.warning("Trend plot update skipped: %s", exc)

    logger.info("--- SELF-PLAY TRAINING LOOP COMPLETED ---")
