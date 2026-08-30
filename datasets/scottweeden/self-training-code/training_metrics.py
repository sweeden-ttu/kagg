"""Cumulative training progress metadata for measuring improvement over time."""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

SCHEMA_VERSION = 1
PROGRESS_FILENAME = "training_progress.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def progress_path(experiment_root: Path) -> Path:
    return Path(experiment_root) / "metrics" / PROGRESS_FILENAME


def empty_lifetime() -> Dict[str, Any]:
    return {
        "runs_completed": 0,
        "bootstrap_days": 0,
        "bootstrap_episodes": 0,
        "bootstrap_transitions": 0,
        "bootstrap_bc_steps": 0,
        "self_play_episodes": 0,
        "self_play_steps": 0,
        "gradient_updates": 0,
        "cumulative_raw_reward": 0.0,
        "cumulative_shaped_reward": 0.0,
        "cumulative_loss": 0.0,
        "win_rate_eval_runs": 0,
        "win_rate_eval_wins": 0,
        "win_rate_eval_episodes": 0,
        "ladder_eval_runs": 0,
        "ladder_eval_wins": 0,
        "ladder_eval_losses": 0,
        "ladder_eval_ties": 0,
        "ladder_eval_episodes": 0,
        "ladder_opponents_cleared_best": 0,
    }


def empty_progress() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": None,
        "lifetime": empty_lifetime(),
        "corpus_trends": {
            "by_date": {},
            "cumulative": {
                "days_indexed": 0,
                "episodes_indexed": 0,
                "total_size_bytes": 0,
                "avg_score_sum": 0.0,
                "min_score_sum": 0.0,
                "sum_score_sum": 0.0,
                "score_margin_sum": 0.0,
            },
        },
        "run_history": [],
        "self_play_episodes": [],
        "eval_history": [],
    }


def load_progress(experiment_root: Path) -> Dict[str, Any]:
    path = progress_path(experiment_root)
    if not path.exists():
        return empty_progress()
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("schema_version") != SCHEMA_VERSION:
        merged = empty_progress()
        merged.update({k: v for k, v in data.items() if k in merged})
        return merged
    return data


def save_progress(experiment_root: Path, state: Dict[str, Any]) -> Path:
    path = progress_path(experiment_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    state["schema_version"] = SCHEMA_VERSION
    state["updated_at"] = _utc_now()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    return path


def corpus_stats_for_episodes(episodes: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate min/avg/sum score and size_bytes for a list of episode index rows."""
    rows = list(episodes)
    if not rows:
        return {
            "episode_count": 0,
            "avg_score_mean": 0.0,
            "min_score_mean": 0.0,
            "sum_score_mean": 0.0,
            "score_margin_mean": 0.0,
            "size_bytes_total": 0,
            "size_bytes_mean": 0.0,
        }
    avg_scores = [float(r["avg_score"]) for r in rows]
    min_scores = [float(r["min_score"]) for r in rows]
    sum_scores = [float(r["sum_score"]) for r in rows]
    sizes = [int(float(r.get("size_bytes", 0))) for r in rows]
    margins = [a - m for a, m in zip(avg_scores, min_scores)]
    return {
        "episode_count": len(rows),
        "avg_score_mean": float(statistics.mean(avg_scores)),
        "min_score_mean": float(statistics.mean(min_scores)),
        "sum_score_mean": float(statistics.mean(sum_scores)),
        "score_margin_mean": float(statistics.mean(margins)),
        "size_bytes_total": int(sum(sizes)),
        "size_bytes_mean": float(statistics.mean(sizes)),
    }


def merge_corpus_trends(
    state: MutableMapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    dates: Optional[Iterable[str]] = None,
) -> None:
    """Merge corpus score/size trends for ``dates`` (or all indexed dates) into progress state."""
    trends = state.setdefault("corpus_trends", {"by_date": {}, "cumulative": {}})
    by_date: Dict[str, Any] = trends.setdefault("by_date", {})
    target_dates = set(dates) if dates is not None else None

    episodes_by_date: Dict[str, List[Mapping[str, Any]]] = {}
    for ep in metadata.get("episodes", []):
        day = str(ep.get("date", ""))
        if not day:
            continue
        if target_dates is not None and day not in target_dates:
            continue
        episodes_by_date.setdefault(day, []).append(ep)

    for day, rows in episodes_by_date.items():
        by_date[day] = corpus_stats_for_episodes(rows)

    cumulative = {
        "days_indexed": len(by_date),
        "episodes_indexed": 0,
        "total_size_bytes": 0,
        "avg_score_sum": 0.0,
        "min_score_sum": 0.0,
        "sum_score_sum": 0.0,
        "score_margin_sum": 0.0,
    }
    for day_stats in by_date.values():
        n = int(day_stats.get("episode_count", 0))
        cumulative["episodes_indexed"] += n
        cumulative["total_size_bytes"] += int(day_stats.get("size_bytes_total", 0))
        cumulative["avg_score_sum"] += float(day_stats.get("avg_score_mean", 0.0)) * n
        cumulative["min_score_sum"] += float(day_stats.get("min_score_mean", 0.0)) * n
        cumulative["sum_score_sum"] += float(day_stats.get("sum_score_mean", 0.0)) * n
        cumulative["score_margin_sum"] += float(day_stats.get("score_margin_mean", 0.0)) * n
    trends["cumulative"] = cumulative


def _sum_episode_metrics(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not episodes:
        return {
            "episodes": 0,
            "cumulative_raw_reward": 0.0,
            "cumulative_shaped_reward": 0.0,
            "gradient_updates": 0,
            "cumulative_loss": 0.0,
        }
    return {
        "episodes": len(episodes),
        "cumulative_raw_reward": float(sum(float(e.get("raw_reward", 0.0)) for e in episodes)),
        "cumulative_shaped_reward": float(sum(float(e.get("shaped_reward", 0.0)) for e in episodes)),
        "gradient_updates": int(sum(int(e.get("gradient_updates", 0)) for e in episodes)),
        "cumulative_loss": float(sum(float(e.get("loss_sum", 0.0)) for e in episodes)),
    }


def enrich_episode_metrics(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Add per-episode and run-level cumulative fields."""
    cumulative_raw = 0.0
    cumulative_shaped = 0.0
    cumulative_loss = 0.0
    cumulative_updates = 0
    enriched: List[Dict[str, Any]] = []

    for row in episodes:
        cumulative_raw += float(row.get("raw_reward", 0.0))
        cumulative_shaped += float(row.get("shaped_reward", 0.0))
        cumulative_loss += float(row.get("loss_sum", row.get("avg_loss", 0.0)))
        cumulative_updates += int(row.get("gradient_updates", 0))
        enriched.append(
            {
                **row,
                "cumulative_raw_reward": cumulative_raw,
                "cumulative_shaped_reward": cumulative_shaped,
                "cumulative_loss": cumulative_loss,
                "cumulative_gradient_updates": cumulative_updates,
            }
        )

    return {
        "totals": {
            **_sum_episode_metrics(enriched),
            "cumulative_raw_reward": cumulative_raw,
            "cumulative_shaped_reward": cumulative_shaped,
            "cumulative_loss": cumulative_loss,
            "gradient_updates": cumulative_updates,
        },
        "episodes": enriched,
    }


def save_episode_metrics(experiment_root: Path, episodes: List[Dict[str, Any]]) -> Path:
    path = Path(experiment_root) / "metrics" / "episode_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = enrich_episode_metrics(episodes)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


class TrainingProgressRecorder:
    """Record and persist cumulative metrics across bootstrap, self-play, and eval."""

    def __init__(
        self,
        experiment_root: Path,
        *,
        run_id: Optional[str] = None,
        resumed: bool = False,
    ) -> None:
        self.experiment_root = Path(experiment_root)
        self.state = load_progress(self.experiment_root)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.started_at = _utc_now()
        self.resumed = resumed
        self._run: Dict[str, Any] = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "resumed": resumed,
            "bootstrap_days": [],
            "bootstrap_episodes": 0,
            "bootstrap_transitions": 0,
            "bootstrap_bc_steps": 0,
            "self_play_episodes": 0,
            "self_play_steps": 0,
            "gradient_updates": 0,
            "cumulative_raw_reward": 0.0,
            "cumulative_shaped_reward": 0.0,
            "cumulative_loss": 0.0,
            "corpus_days": [],
        }
        self._self_play_rows: List[Dict[str, Any]] = list(self.state.get("self_play_episodes", []))

    def record_bootstrap_result(
        self,
        result: Mapping[str, Any],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        new_days = list(result.get("new_days") or result.get("bootstrap_days_this_run") or [])
        self._run["bootstrap_days"] = new_days
        self._run["bootstrap_transitions"] = int(
            result.get("total_transitions_loaded")
            or result.get("transitions_this_run")
            or result.get("bootstrap_transitions_loaded")
            or 0
        )
        day_stats = list(result.get("day_stats") or [])
        self._run["bootstrap_episodes"] = int(
            sum(int(d.get("episodes", 0)) for d in day_stats)
        )
        self._run["bootstrap_bc_steps"] = int(
            sum(len(d.get("bc_epoch_losses") or []) for d in day_stats)
        )
        if metadata and new_days:
            merge_corpus_trends(self.state, metadata, dates=new_days)
            self._run["corpus_days"] = [
                self.state["corpus_trends"]["by_date"].get(day, {"date": day})
                for day in new_days
            ]

    def record_self_play_episode(self, metrics: Mapping[str, Any]) -> Dict[str, Any]:
        row = dict(metrics)
        raw = float(row.get("raw_reward", 0.0))
        shaped = float(row.get("shaped_reward", 0.0))
        loss = float(row.get("loss_sum", row.get("avg_loss", 0.0)))
        updates = int(row.get("gradient_updates", 0))
        steps = int(row.get("steps", 0))

        lifetime = self.state["lifetime"]
        row["cumulative_raw_reward"] = (
            float(lifetime.get("cumulative_raw_reward", 0.0))
            + float(self._run.get("cumulative_raw_reward", 0.0))
            + raw
        )
        row["cumulative_shaped_reward"] = (
            float(lifetime.get("cumulative_shaped_reward", 0.0))
            + float(self._run.get("cumulative_shaped_reward", 0.0))
            + shaped
        )
        row["cumulative_loss"] = (
            float(lifetime.get("cumulative_loss", 0.0))
            + float(self._run.get("cumulative_loss", 0.0))
            + loss
        )
        row["cumulative_gradient_updates"] = (
            int(lifetime.get("gradient_updates", 0))
            + int(self._run.get("gradient_updates", 0))
            + updates
        )
        row["lifetime_episode"] = (
            int(lifetime.get("self_play_episodes", 0))
            + int(self._run.get("self_play_episodes", 0))
            + 1
        )
        row["recorded_at"] = _utc_now()

        self._run["self_play_episodes"] += 1
        self._run["self_play_steps"] += steps
        self._run["gradient_updates"] += updates
        self._run["cumulative_raw_reward"] += raw
        self._run["cumulative_shaped_reward"] += shaped
        self._run["cumulative_loss"] += loss

        self._self_play_rows.append(row)
        self.state["self_play_episodes"] = self._self_play_rows
        save_progress(self.experiment_root, self.state)
        return row

    def record_eval(self, kind: str, report: Mapping[str, Any]) -> None:
        entry = {
            "kind": kind,
            "recorded_at": _utc_now(),
            "run_id": self.run_id,
            **{k: v for k, v in report.items() if k != "episodes"},
        }
        history: List[Dict[str, Any]] = self.state.setdefault("eval_history", [])
        history.append(entry)
        self._run.setdefault("evals", []).append(entry)

        lifetime = self.state["lifetime"]
        if kind == "win_rate":
            lifetime["win_rate_eval_runs"] += 1
            lifetime["win_rate_eval_wins"] += int(report.get("wins", 0))
            lifetime["win_rate_eval_episodes"] += int(report.get("n_episodes", 0))
        elif kind == "ladder":
            lifetime["ladder_eval_runs"] += 1
            lifetime["ladder_eval_wins"] += int(report.get("wins_total", 0))
            lifetime["ladder_eval_losses"] += int(report.get("losses_total", 0))
            lifetime["ladder_eval_ties"] += int(report.get("ties_total", 0))
            lifetime["ladder_eval_episodes"] += int(report.get("n_episodes_total", 0))
            cleared = int(report.get("opponents_cleared", 0))
            lifetime["ladder_opponents_cleared_best"] = max(
                int(lifetime.get("ladder_opponents_cleared_best", 0)),
                cleared,
            )

    def finalize_run(self, config: Optional[Mapping[str, Any]] = None) -> Path:
        finished_at = _utc_now()
        self._run["finished_at"] = finished_at
        if config:
            self._run["training_mode"] = config.get("training_mode")
            self._run["total_episodes_target"] = config.get("total_episodes")
            self._run["last_completed_episode"] = config.get("last_completed_episode")

        lifetime = self.state["lifetime"]
        lifetime["runs_completed"] = int(lifetime.get("runs_completed", 0)) + 1
        lifetime["bootstrap_days"] = max(
            int(lifetime.get("bootstrap_days", 0)),
            len(set(self.state.get("corpus_trends", {}).get("by_date", {}))),
        )
        lifetime["bootstrap_episodes"] += int(self._run.get("bootstrap_episodes", 0))
        lifetime["bootstrap_transitions"] += int(self._run.get("bootstrap_transitions", 0))
        lifetime["bootstrap_bc_steps"] += int(self._run.get("bootstrap_bc_steps", 0))
        lifetime["self_play_episodes"] += int(self._run.get("self_play_episodes", 0))
        lifetime["self_play_steps"] += int(self._run.get("self_play_steps", 0))
        lifetime["gradient_updates"] += int(self._run.get("gradient_updates", 0))
        lifetime["cumulative_raw_reward"] += float(self._run.get("cumulative_raw_reward", 0.0))
        lifetime["cumulative_shaped_reward"] += float(self._run.get("cumulative_shaped_reward", 0.0))
        lifetime["cumulative_loss"] += float(self._run.get("cumulative_loss", 0.0))

        runs: List[Dict[str, Any]] = self.state.setdefault("run_history", [])
        runs.append(dict(self._run))
        return save_progress(self.experiment_root, self.state)
