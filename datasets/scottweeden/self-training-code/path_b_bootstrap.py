"""Bootstrap Path B self-play replay buffer from Kaggle episode JSONs + BC pretrain."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch

from kaggriculture_adapter import encode_path_b_action, encode_path_b_observation
from dataset_loader import EPISODE_FILE_PATTERN, KaggleEpisodesLoader
from datetime import date as date_cls, timedelta
from episode_catalog import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    dates_in_metadata,
    is_kaggle_runtime,
    pick_next_bootstrap_days,
    resolve_episode_json_path,
    resolve_episode_paths_for_dates,
    resolve_episode_paths_from_metadata,
)

if TYPE_CHECKING:
    from kaggriculture_path_b_rebuild import HierarchicalDoubleDQNLearner

logger = logging.getLogger(__name__)

DAILY_DATASET_PREFIX = "kaggriculture-episodes-"
BOOTSTRAP_STATE_FILENAME = "bootstrap_state.json"


def _episode_date_from_path(path: Path) -> str:
    """Extract YYYY-MM-DD from ``.../kaggriculture-episodes-YYYY-MM-DD/{id}.json``."""
    parent = path.parent.name
    if parent.startswith(DAILY_DATASET_PREFIX):
        return parent[len(DAILY_DATASET_PREFIX):]
    return ""


def order_episode_files_by_date(
    episode_files: List[Path],
    metadata_path: Optional[str] = None,
) -> List[Path]:
    """Order episodes oldest-date first, highest ``avg_score`` first within each day."""
    date_by_id: Dict[str, str] = {}
    score_by_id: Dict[str, float] = {}
    if metadata_path and Path(metadata_path).exists():
        with open(metadata_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        for ep in meta.get("episodes", []):
            ep_id = str(ep["episode_id"])
            date_by_id[ep_id] = str(ep.get("date", ""))
            score_by_id[ep_id] = float(ep.get("avg_score", 0))

    def sort_key(path: Path) -> Tuple[str, float, str]:
        ep_id = path.stem
        date = date_by_id.get(ep_id) or _episode_date_from_path(path)
        score = score_by_id.get(ep_id, 0.0)
        return (date, -score, ep_id)

    return sorted(episode_files, key=sort_key)


def parse_path_b_episode_transitions(
    episode_data: Dict[str, Any],
    max_market_orders: int = 10,
) -> List[Dict[str, Any]]:
    """Parse one Kaggle episode JSON into Path B replay-buffer transitions."""
    steps = episode_data.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        return []

    transitions: List[Dict[str, Any]] = []

    for step_idx in range(len(steps) - 1):
        current_turn = steps[step_idx]
        next_turn = steps[step_idx + 1]
        if not isinstance(current_turn, list) or not isinstance(next_turn, list):
            continue

        for player_id in range(min(len(current_turn), len(next_turn))):
            current = current_turn[player_id]
            nxt = next_turn[player_id]
            if not isinstance(current, dict) or not isinstance(nxt, dict):
                continue

            observation = current.get("observation")
            action_raw = current.get("action")
            next_observation = nxt.get("observation")
            if observation is None or action_raw is None or next_observation is None:
                continue

            try:
                obs_with_player = dict(observation)
                obs_with_player["player"] = player_id
                next_with_player = dict(next_observation)
                next_with_player["player"] = player_id

                parsed = encode_path_b_observation(obs_with_player, player_id)
                parsed_next = encode_path_b_observation(next_with_player, player_id)
                action = encode_path_b_action(action_raw, max_market_orders=max_market_orders)

                reward = float(nxt.get("reward", 0.0))
                status = nxt.get("status", "ACTIVE")
                done = status in ("DONE", "TIMEOUT", "INVALID")
            except Exception:
                continue

            transitions.append(
                {
                    "tiles": parsed["tiles"],
                    "numeric": parsed["numeric"],
                    "action_verb": action["verb"],
                    "action_crop": action["crop"],
                    "action_hands": action["hands"],
                    "action_market": action["market"],
                    "reward": reward,
                    "next_tiles": parsed_next["tiles"],
                    "next_numeric": parsed_next["numeric"],
                    "done": done,
                }
            )

    return transitions


def _push_transition(buffer: Any, transition: Dict[str, Any]) -> None:
    buffer.push(
        tiles=transition["tiles"],
        numeric=transition["numeric"],
        action_verb=transition["action_verb"],
        action_crop=transition["action_crop"],
        action_hands=transition["action_hands"],
        action_market=transition["action_market"],
        reward=transition["reward"],
        next_tiles=transition["next_tiles"],
        next_numeric=transition["next_numeric"],
        done=transition["done"],
    )


def resolve_bootstrap_episode_files(
    data_dir: str = "./data/kaggle_episodes",
    max_episodes: Optional[int] = 100,
    download: bool = False,
    metadata_path: Optional[str] = None,
    episode_ids: Optional[List[str]] = None,
    top_per_day: Optional[int] = 20,
) -> List[Path]:
    """Resolve episode JSON paths from metadata or local ``episodes/`` dir."""
    data_path = Path(data_dir)
    episodes_dir = None if is_kaggle_runtime() else data_path / "episodes"
    episode_files: List[Path] = []

    if episode_ids:
        for ep_id in episode_ids:
            fpath = episodes_dir / f"{ep_id}.json"
            if fpath.exists():
                episode_files.append(fpath)
    elif metadata_path and Path(metadata_path).exists():
        with open(metadata_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        episode_files = resolve_episode_paths_from_metadata(
            meta,
            top_per_day=top_per_day,
            max_episodes=max_episodes,
            local_episodes_dir=episodes_dir,
        )
        if not episode_files:
            ranked = sorted(
                meta.get("episodes", []),
                key=lambda r: r.get("avg_score", 0),
                reverse=True,
            )
            rows = ranked[:max_episodes] if max_episodes is not None else ranked
            for row in rows:
                resolved = resolve_episode_json_path(
                    row["date"],
                    row["episode_id"],
                    local_episodes_dir=episodes_dir,
                )
                if resolved is not None:
                    episode_files.append(resolved)
    else:
        loader = KaggleEpisodesLoader(data_dir=data_dir)
        if download:
            loader.download_if_needed()
        episode_files = loader.get_episode_files()
        if not episode_files:
            episode_files = sorted(
                fpath
                for fpath in episodes_dir.glob("*.json")
                if EPISODE_FILE_PATTERN.match(fpath.name)
            )

    episode_files = order_episode_files_by_date(episode_files, metadata_path=metadata_path)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "resolve_bootstrap_episode_files: count=%d first=%s last=%s kaggle=%s",
            len(episode_files),
            episode_files[0] if episode_files else None,
            episode_files[-1] if episode_files else None,
            is_kaggle_runtime(),
        )

    return episode_files


def bootstrap_pass_fill_buffer(
    buffer: Any,
    episode_files: List[Path],
    max_transitions: int,
    start_episode_idx: int = 0,
    max_market_orders: int = 10,
    clear_buffer: bool = True,
    verbose: bool = False,
) -> Tuple[int, int, int]:
    """Stream transitions from date-ordered corpus starting at ``start_episode_idx``.

    Returns ``(transitions_loaded, episodes_read, next_episode_idx)``.
    """
    if clear_buffer and hasattr(buffer, "clear"):
        buffer.clear()

    loaded = 0
    episodes_read = 0
    idx = start_episode_idx
    remaining = len(episode_files) - start_episode_idx

    if verbose:
        logger.debug(
            "bootstrap_pass_fill_buffer: start_idx=%d remaining=%d cap=%d clear=%s",
            start_episode_idx,
            remaining,
            max_transitions,
            clear_buffer,
        )

    while idx < len(episode_files) and loaded < max_transitions:
        fpath = episode_files[idx]
        idx += 1
        try:
            with open(fpath, encoding="utf-8") as fh:
                episode_data = json.load(fh)
            transitions = parse_path_b_episode_transitions(
                episode_data, max_market_orders=max_market_orders
            )
            episodes_read += 1
            for transition in transitions:
                if loaded >= max_transitions:
                    break
                _push_transition(buffer, transition)
                loaded += 1

            if verbose and (episodes_read <= 3 or episodes_read % 100 == 0):
                logger.debug(
                    "Bootstrap read ep %d: %s | date=%s | %d transitions | buffer=%d/%d",
                    episodes_read,
                    fpath.name,
                    _episode_date_from_path(fpath),
                    len(transitions),
                    loaded,
                    max_transitions,
                )
            if verbose and episodes_read == 1 and transitions:
                sample = transitions[0]
                logger.debug(
                    "Sample transition shapes: tiles=%s numeric=%s | actions verb=%s crop=%s "
                    "hands=%s market=%s | reward=%.4f done=%s",
                    np.shape(sample.get("tiles")),
                    np.shape(sample.get("numeric")),
                    sample.get("action_verb"),
                    sample.get("action_crop"),
                    sample.get("action_hands"),
                    sample.get("action_market"),
                    float(sample.get("reward", 0.0)),
                    sample.get("done"),
                )
        except Exception as exc:
            logger.warning("Failed to bootstrap from %s: %s", fpath, exc)

    if verbose:
        logger.debug(
            "bootstrap_pass_fill_buffer done: loaded=%d episodes_read=%d next_idx=%d/%d",
            loaded,
            episodes_read,
            idx,
            len(episode_files),
        )

    return loaded, episodes_read, idx


def bootstrap_path_b_replay_buffer(
    buffer: Any,
    data_dir: str = "./data/kaggle_episodes",
    max_episodes: Optional[int] = 100,
    max_transitions: Optional[int] = 50_000,
    max_market_orders: int = 10,
    random_seed: int = 42,
    download: bool = False,
    metadata_path: Optional[str] = None,
    episode_ids: Optional[List[str]] = None,
    top_per_day: Optional[int] = 20,
) -> int:
    """Load Kaggle episode JSONs into a Path B replay buffer (single sequential pass)."""
    episode_files = resolve_bootstrap_episode_files(
        data_dir=data_dir,
        max_episodes=max_episodes,
        download=download,
        metadata_path=metadata_path,
        episode_ids=episode_ids,
        top_per_day=top_per_day,
    )

    if not episode_files:
        logger.warning(
            "No episode JSON files found. Expected Kaggle input paths like "
            "/kaggle/input/kaggriculture-episodes-YYYY-MM-DD/{id}.json "
            "or local copies under %s/episodes/.",
            data_dir,
        )
        return 0

    cap = max_transitions if max_transitions is not None else buffer.capacity
    ordered = order_episode_files_by_date(episode_files, metadata_path=metadata_path)
    logger.info(
        "Bootstrapping Path B buffer from %d episode files (cap=%d; oldest first: %s)",
        len(ordered),
        cap,
        ordered[0],
    )

    loaded, episodes_read, _ = bootstrap_pass_fill_buffer(
        buffer,
        ordered,
        max_transitions=cap,
        start_episode_idx=0,
        max_market_orders=max_market_orders,
        clear_buffer=False,
    )
    logger.info(
        "Bootstrap complete: %d transitions from %d episodes (buffer size=%d)",
        loaded,
        episodes_read,
        len(buffer),
    )
    return loaded


def group_episode_files_by_date(
    episode_files: List[Path],
    metadata_path: Optional[str] = None,
) -> List[Tuple[str, List[Path]]]:
    """Group episode paths by calendar date, oldest first."""
    ordered = order_episode_files_by_date(episode_files, metadata_path=metadata_path)
    by_date: Dict[str, List[Path]] = {}
    for fpath in ordered:
        day = _episode_date_from_path(fpath) or "unknown"
        by_date.setdefault(day, []).append(fpath)
    return [(day, by_date[day]) for day in sorted(by_date.keys())]


def bootstrap_state_path(experiment_root: Path) -> Path:
    return experiment_root / "metrics" / BOOTSTRAP_STATE_FILENAME


def bootstrap_metadata_start_date(
    bootstrapped_dates: Optional[List[str]],
    default_start: str = DEFAULT_START_DATE,
) -> str:
    """First date to index/download after the last bootstrapped day."""
    if not bootstrapped_dates:
        return default_start
    resume = (date_cls.fromisoformat(max(bootstrapped_dates)) + timedelta(days=1)).isoformat()
    return max(resume, default_start)


def plan_next_bootstrap_days_from_state(
    bootstrapped_dates: Optional[List[str]],
    n_days: int,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> List[str]:
    """Next chronological bootstrap days using saved state (no metadata required)."""
    excluded = set(bootstrapped_dates or [])
    window_start = bootstrap_metadata_start_date(sorted(excluded), default_start=start_date)
    available = [d for d in _dates_between(window_start, end_date) if d not in excluded]
    return available[: max(0, n_days)]


def _dates_between(start_date: str, end_date: str) -> List[str]:
    start = date_cls.fromisoformat(start_date)
    end = date_cls.fromisoformat(end_date)
    days: List[str] = []
    current = start
    while current <= end:
        days.append(current.isoformat())
        current += timedelta(days=1)
    return days


def _bootstrapped_dates_from_config(config_path: Path) -> List[str]:
    if not config_path.exists():
        return []
    with open(config_path, encoding="utf-8") as fh:
        return list(json.load(fh).get("bootstrapped_dates", []))


def merge_bootstrap_state_from_code_dataset(
    code_src: Path,
    experiment_dir: Path,
) -> Dict[str, Any]:
    """Union bootstrapped_dates from local run + code-dataset training_artifacts."""
    experiment_dir = Path(experiment_dir)
    code_src = Path(code_src)
    local_state = load_bootstrap_state(experiment_dir)

    code_state_path = code_src / "training_artifacts" / "metrics" / BOOTSTRAP_STATE_FILENAME
    code_state: Dict[str, Any] = {"bootstrapped_dates": []}
    if code_state_path.exists():
        with open(code_state_path, encoding="utf-8") as fh:
            code_state = json.load(fh)

    merged_dates = sorted(
        set(local_state.get("bootstrapped_dates", []))
        | set(code_state.get("bootstrapped_dates", []))
        | set(_bootstrapped_dates_from_config(experiment_dir / "config.json"))
        | set(_bootstrapped_dates_from_config(code_src / "training_artifacts" / "config.json"))
    )

    state = dict(local_state)
    state["bootstrapped_dates"] = merged_dates
    state["total_transitions"] = max(
        int(local_state.get("total_transitions", 0)),
        int(code_state.get("total_transitions", 0)),
    )
    if merged_dates or local_state.get("runs") or code_state.get("runs"):
        save_bootstrap_state(experiment_dir, state)
    return state


def load_bootstrap_state(experiment_root: Path) -> Dict[str, Any]:
    path = bootstrap_state_path(experiment_root)
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {"bootstrapped_dates": [], "runs": [], "total_transitions": 0}


def save_bootstrap_state(experiment_root: Path, state: Dict[str, Any]) -> Path:
    path = bootstrap_state_path(experiment_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    dates = sorted(set(state.get("bootstrapped_dates", [])))
    state["bootstrapped_dates"] = dates
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    logger.info(
        "Bootstrap state saved: %d days bootstrapped → %s",
        len(dates),
        path,
    )
    return path


def is_day_bootstrapped(experiment_root: Path, day: str) -> bool:
    """Return True if ``day`` is already listed in bootstrap_state.json."""
    state = load_bootstrap_state(experiment_root)
    return day in set(state.get("bootstrapped_dates", []))


def mark_day_bootstrapped(
    experiment_root: Path,
    state: Dict[str, Any],
    day: str,
    *,
    day_stats: Optional[Dict[str, Any]] = None,
) -> bool:
    """Record ``day`` in bootstrapped_dates and persist immediately."""
    dates = list(state.get("bootstrapped_dates", []))
    if day in dates:
        logger.warning("Skip bootstrap day %s: already in bootstrapped_dates", day)
        return False

    dates.append(day)
    state["bootstrapped_dates"] = sorted(set(dates))
    if day_stats:
        records = state.setdefault("day_records", {})
        records[day] = day_stats
    save_bootstrap_state(experiment_root, state)
    logger.info(
        "Marked day bootstrapped: %s (%d days total)",
        day,
        len(state["bootstrapped_dates"]),
    )
    return True


def _collate_bc_batch(transitions: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    tiles_b, numeric_b, act_v_b, act_c_b, act_h_b, act_m_b, r_b, n_tiles_b, n_num_b, d_b = zip(
        *[
            (
                t["tiles"],
                t["numeric"],
                t["action_verb"],
                t["action_crop"],
                t["action_hands"],
                t["action_market"],
                t["reward"],
                t["next_tiles"],
                t["next_numeric"],
                t["done"],
            )
            for t in transitions
        ]
    )
    return {
        "tiles": torch.as_tensor(np.array(tiles_b), dtype=torch.float32),
        "numeric": torch.as_tensor(np.array(numeric_b), dtype=torch.float32),
        "action_verb": torch.as_tensor(act_v_b, dtype=torch.long),
        "action_crop": torch.as_tensor(act_c_b, dtype=torch.long),
        "action_hands": torch.as_tensor(np.array(act_h_b), dtype=torch.long),
        "action_market": torch.as_tensor(np.array(act_m_b), dtype=torch.long),
        "reward": torch.as_tensor(r_b, dtype=torch.float32),
        "next_tiles": torch.as_tensor(np.array(n_tiles_b), dtype=torch.float32),
        "next_numeric": torch.as_tensor(np.array(n_num_b), dtype=torch.float32),
        "done": torch.as_tensor(d_b, dtype=torch.float32),
    }


def run_bc_pretrain_over_episode_files(
    learner: HierarchicalDoubleDQNLearner,
    device: torch.device,
    episode_files: List[Path],
    epochs: int = 1,
    batch_size: int = 64,
    max_steps_per_epoch: Optional[int] = None,
    max_market_orders: int = 10,
    random_seed: int = 42,
    verbose: bool = False,
) -> Tuple[List[float], int]:
    """Stream all transitions from episode files into BC (no replay-buffer cap).

    Returns ``(epoch_losses, transition_count)``.
    """
    if epochs <= 0 or not episode_files:
        return [], 0

    import random

    rng = random.Random(random_seed)
    transition_count = 0
    epoch_losses: List[float] = []

    learner.online.train()
    logger.info(
        "Streaming BC over %d episode files (epochs=%d, batch=%d, steps/epoch=%s)",
        len(episode_files),
        epochs,
        batch_size,
        max_steps_per_epoch if max_steps_per_epoch is not None else "all",
    )

    for epoch in range(1, epochs + 1):
        files = list(episode_files)
        rng.shuffle(files)
        pending: List[Dict[str, Any]] = []
        batch_losses: List[float] = []
        step = 0

        for fpath in files:
            try:
                with open(fpath, encoding="utf-8") as fh:
                    episode_data = json.load(fh)
                transitions = parse_path_b_episode_transitions(
                    episode_data, max_market_orders=max_market_orders
                )
            except Exception as exc:
                logger.warning("BC skip %s: %s", fpath, exc)
                continue

            transition_count += len(transitions)
            pending.extend(transitions)

            while len(pending) >= batch_size:
                batch = _collate_bc_batch(pending[:batch_size])
                pending = pending[batch_size:]
                for key in batch:
                    batch[key] = batch[key].to(device)

                loss = learner.compute_bc_loss(batch)
                learner.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(learner.online.parameters(), max_norm=1.0)
                learner.optimizer.step()
                loss_val = float(loss.item())
                batch_losses.append(loss_val)
                step += 1

                if verbose and (step <= 3 or step % 100 == 0):
                    logger.debug(
                        "BC stream epoch %d/%d step %d | loss=%.5f | file=%s",
                        epoch,
                        epochs,
                        step,
                        loss_val,
                        fpath.name,
                    )
                if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                    break

            if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                break

        avg_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        epoch_losses.append(avg_loss)
        logger.info(
            "BC stream epoch %d/%d | steps=%d | avg loss: %.5f",
            epoch,
            epochs,
            step,
            avg_loss,
        )

    learner.update_target_network()
    learner.online.eval()
    return epoch_losses, transition_count


def seed_buffer_from_episode_files(
    buffer: Any,
    episode_files: List[Path],
    max_transitions: Optional[int] = None,
    max_market_orders: int = 10,
    clear_buffer: bool = False,
) -> int:
    """Load transitions into replay buffer for self-play (optional cap)."""
    cap = max_transitions if max_transitions is not None else getattr(buffer, "capacity", 10_000)
    loaded, _, _ = bootstrap_pass_fill_buffer(
        buffer,
        episode_files,
        max_transitions=cap,
        start_episode_idx=0,
        max_market_orders=max_market_orders,
        clear_buffer=clear_buffer,
        verbose=False,
    )
    return loaded


def stream_bootstrap_bc_pretrain(
    learner: HierarchicalDoubleDQNLearner,
    buffer: Any,
    device: torch.device,
    episode_files: List[Path],
    bootstrap_passes: int = 10,
    max_transitions_per_pass: Optional[int] = None,
    bc_epochs_per_pass: int = 1,
    bc_batch_size: int = 64,
    bc_steps_per_epoch: Optional[int] = None,
    max_market_orders: int = 10,
    random_seed: int = 42,
    metadata_path: Optional[str] = None,
    experiment_root: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Multi-pass bootstrap: one **full calendar day** per pass, chronological order."""
    if bootstrap_passes <= 0 or not episode_files:
        return {
            "epoch_losses": [],
            "pass_losses": [],
            "total_transitions_loaded": 0,
            "total_episodes_read": 0,
            "bootstrap_passes": 0,
        }

    day_groups = group_episode_files_by_date(episode_files, metadata_path=metadata_path)
    if not day_groups:
        return {
            "epoch_losses": [],
            "pass_losses": [],
            "total_transitions_loaded": 0,
            "total_episodes_read": 0,
            "bootstrap_passes": 0,
        }

    state = load_bootstrap_state(experiment_root) if experiment_root else {"bootstrapped_dates": []}
    bootstrapped = set(state.get("bootstrapped_dates", []))
    pending_days = [(d, files) for d, files in day_groups if d not in bootstrapped]
    if experiment_root:
        days_this_run = pending_days[:bootstrap_passes]
    else:
        days_this_run = day_groups[:bootstrap_passes]

    if not days_this_run:
        logger.info(
            "Streaming bootstrap: all %d indexed days already bootstrapped",
            len(day_groups),
        )
        return {
            "epoch_losses": [],
            "pass_losses": [],
            "total_transitions_loaded": 0,
            "total_episodes_read": 0,
            "bootstrap_passes": 0,
            "bootstrapped_dates": sorted(bootstrapped),
        }

    per_day_buffer_cap = max_transitions_per_pass
    if per_day_buffer_cap is None and hasattr(buffer, "capacity"):
        per_day_buffer_cap = max(1, buffer.capacity // max(1, len(days_this_run)))

    all_losses: List[float] = []
    pass_losses: List[List[float]] = []
    total_transitions = 0
    total_episodes_read = 0

    logger.info(
        "Streaming bootstrap: %d day pass(es) (%s → %s); %d/%d corpus days left",
        len(days_this_run),
        days_this_run[0][0],
        days_this_run[-1][0],
        len(pending_days),
        len(day_groups),
    )

    for pass_idx, (date, day_files) in enumerate(days_this_run, 1):
        if experiment_root and is_day_bootstrapped(experiment_root, date):
            logger.warning(
                "Bootstrap pass %d/%d: skip day %s (already bootstrapped)",
                pass_idx,
                len(days_this_run),
                date,
            )
            continue

        logger.info(
            "Bootstrap pass %d/%d: day %s — %d episodes (full-day streaming BC)",
            pass_idx,
            len(days_this_run),
            date,
            len(day_files),
        )

        if hasattr(buffer, "clear") and pass_idx == 1:
            buffer.clear()

        day_losses: List[float] = []
        day_transitions = 0
        if bc_epochs_per_pass > 0:
            day_losses, day_transitions = run_bc_pretrain_over_episode_files(
                learner,
                device,
                day_files,
                epochs=bc_epochs_per_pass,
                batch_size=bc_batch_size,
                max_steps_per_epoch=bc_steps_per_epoch,
                max_market_orders=max_market_orders,
                random_seed=random_seed + pass_idx,
                verbose=verbose,
            )
            pass_losses.append(day_losses)
            all_losses.extend(day_losses)

        total_transitions += day_transitions
        total_episodes_read += len(day_files)

        seed_buffer_from_episode_files(
            buffer,
            day_files,
            max_transitions=per_day_buffer_cap,
            max_market_orders=max_market_orders,
            clear_buffer=False,
        )

        day_record = {
            "date": date,
            "episodes": len(day_files),
            "transitions": day_transitions,
            "bc_final_loss": day_losses[-1] if day_losses else None,
        }
        if experiment_root:
            mark_day_bootstrapped(experiment_root, state, date, day_stats=day_record)
        else:
            bootstrapped.add(date)

    bootstrapped_sorted = sorted(
        state.get("bootstrapped_dates", []) if experiment_root else bootstrapped
    )
    if experiment_root:
        state["total_transitions"] = int(state.get("total_transitions", 0)) + total_transitions
        state.setdefault("runs", []).append(
            {
                "mode": "streaming_by_day",
                "new_days": [d for d, _ in days_this_run],
                "transitions_this_run": total_transitions,
                "bc_final_loss": all_losses[-1] if all_losses else None,
            }
        )
        save_bootstrap_state(experiment_root, state)

    return {
        "epoch_losses": all_losses,
        "pass_losses": pass_losses,
        "total_transitions_loaded": total_transitions,
        "total_episodes_read": total_episodes_read,
        "bootstrap_passes": len(days_this_run),
        "bootstrapped_dates": bootstrapped_sorted,
        "new_days": [d for d, _ in days_this_run],
    }


def run_bc_pretrain(
    learner: HierarchicalDoubleDQNLearner,
    buffer: Any,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 64,
    max_steps_per_epoch: Optional[int] = None,
    verbose: bool = False,
) -> List[float]:
    """Behavioral cloning pretrain on bootstrapped (or any) replay buffer."""
    if epochs <= 0 or len(buffer) == 0:
        return []

    if not hasattr(buffer, "sample_uniform"):
        raise TypeError("Buffer must implement sample_uniform for BC pretraining")

    learner.online.train()
    epoch_losses: List[float] = []
    steps_per_epoch = max(1, len(buffer) // batch_size)
    if max_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps_per_epoch)

    logger.info(
        "Starting BC pretrain: %d epochs, batch_size=%d, buffer=%d, steps/epoch=%d",
        epochs, batch_size, len(buffer), steps_per_epoch,
    )

    for epoch in range(1, epochs + 1):
        batch_losses: List[float] = []
        for step_i in range(steps_per_epoch):
            batch = buffer.sample_uniform(batch_size)
            if not batch:
                break
            for key in batch:
                batch[key] = batch[key].to(device)

            loss = learner.compute_bc_loss(batch)
            learner.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(learner.online.parameters(), max_norm=1.0)
            learner.optimizer.step()
            loss_val = float(loss.item())
            batch_losses.append(loss_val)

            if verbose and (step_i < 3 or step_i % 50 == 0 or step_i == steps_per_epoch - 1):
                batch_shapes = {k: tuple(v.shape) for k, v in batch.items()}
                logger.debug(
                    "BC epoch %d/%d step %d/%d | loss=%.5f | batch shapes=%s",
                    epoch,
                    epochs,
                    step_i + 1,
                    steps_per_epoch,
                    loss_val,
                    batch_shapes,
                )

        avg_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        epoch_losses.append(avg_loss)
        logger.info("BC epoch %d/%d | avg loss: %.5f", epoch, epochs, avg_loss)

    learner.update_target_network()
    learner.online.eval()
    return epoch_losses


def incremental_daily_bootstrap_bc(
    learner: HierarchicalDoubleDQNLearner,
    buffer: Any,
    device: torch.device,
    metadata_path: str,
    experiment_root: Path,
    days_per_run: int = 3,
    bc_epochs_per_day: int = 1,
    bc_batch_size: int = 64,
    bc_steps_per_epoch: Optional[int] = None,
    max_market_orders: int = 10,
    random_seed: int = 42,
    buffer_seed_per_day: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Pick next ``days_per_run`` chronological days; BC on **all** episodes each day."""
    meta_path = Path(metadata_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata_path not found: {metadata_path}")

    with open(meta_path, encoding="utf-8") as fh:
        metadata = json.load(fh)

    state = load_bootstrap_state(experiment_root)
    bootstrapped = list(state.get("bootstrapped_dates", []))
    new_days = pick_next_bootstrap_days(
        metadata,
        n_days=days_per_run,
        exclude_dates=bootstrapped,
    )

    if not new_days:
        logger.info(
            "Incremental daily bootstrap: all %d indexed dates already bootstrapped",
            len(bootstrapped),
        )
        return {
            "epoch_losses": [],
            "new_days": [],
            "bootstrapped_dates": bootstrapped,
            "total_transitions_loaded": 0,
            "bootstrap_mode": "daily_incremental",
        }

    all_dates = dates_in_metadata(metadata)
    logger.info(
        "Incremental daily bootstrap: adding %d day(s) %s (%d/%d dates already done)",
        len(new_days),
        new_days,
        len(bootstrapped),
        len(all_dates),
    )

    all_losses: List[float] = []
    run_day_stats: List[Dict[str, Any]] = []
    total_transitions = 0
    episodes_dir = None if is_kaggle_runtime() else Path(metadata_path).parent / "episodes"

    per_day_buffer_cap = buffer_seed_per_day
    if per_day_buffer_cap is None and hasattr(buffer, "capacity"):
        per_day_buffer_cap = max(1, buffer.capacity // max(1, len(new_days)))

    for day in new_days:
        if is_day_bootstrapped(experiment_root, day):
            logger.warning("Skip bootstrap day %s: already bootstrapped", day)
            continue

        day_files = resolve_episode_paths_for_dates(
            metadata, [day], local_episodes_dir=episodes_dir
        )
        if not day_files:
            logger.warning("No episode files resolved for date %s", day)
            continue

        logger.info("Bootstrap day %s: %d episodes (all transitions, streaming BC)", day, len(day_files))

        day_losses, day_transitions = run_bc_pretrain_over_episode_files(
            learner,
            device,
            day_files,
            epochs=bc_epochs_per_day,
            batch_size=bc_batch_size,
            max_steps_per_epoch=bc_steps_per_epoch,
            max_market_orders=max_market_orders,
            random_seed=random_seed + hash(day) % 10_000,
            verbose=verbose,
        )
        all_losses.extend(day_losses)
        total_transitions += day_transitions

        seeded = seed_buffer_from_episode_files(
            buffer,
            day_files,
            max_transitions=per_day_buffer_cap,
            max_market_orders=max_market_orders,
            clear_buffer=False,
        )

        day_record = {
            "date": day,
            "episodes": len(day_files),
            "transitions": day_transitions,
            "buffer_seeded": seeded,
            "bc_epoch_losses": day_losses,
            "bc_final_loss": day_losses[-1] if day_losses else None,
        }
        run_day_stats.append(day_record)
        mark_day_bootstrapped(experiment_root, state, day, day_stats=day_record)

    bootstrapped = sorted(state.get("bootstrapped_dates", []))
    state["total_transitions"] = int(state.get("total_transitions", 0)) + total_transitions
    state.setdefault("runs", []).append(
        {
            "new_days": [s["date"] for s in run_day_stats],
            "day_stats": run_day_stats,
            "transitions_this_run": total_transitions,
            "bc_final_loss": all_losses[-1] if all_losses else None,
        }
    )
    save_bootstrap_state(experiment_root, state)

    return {
        "epoch_losses": all_losses,
        "new_days": new_days,
        "bootstrapped_dates": bootstrapped,
        "total_transitions_loaded": total_transitions,
        "day_stats": run_day_stats,
        "bootstrap_mode": "daily_incremental",
        "bootstrap_state_path": str(bootstrap_state_path(experiment_root)),
    }
