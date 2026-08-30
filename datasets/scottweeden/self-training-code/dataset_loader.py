"""Dataset loader for bootstrapping RL from Kaggle episodes.

Downloads and loads episodes from the Kaggle Kaggriculture dataset:
https://www.kaggle.com/datasets/kaggle/kaggriculture-episodes-index

These episodes contain transitions from real player games, which we use
to pre-populate the replay buffer before starting self-play training.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from kaggriculture_adapter import (
    CROPS,
    FARMER_ACTIONS,
    MARKET_ACTIONS,
    encode_action as encode_kaggriculture_action,
    encode_observation as encode_kaggriculture_observation,
)

# Re-export for backward compatibility
encode_observation = encode_kaggriculture_observation
encode_action = encode_kaggriculture_action

logger = logging.getLogger(__name__)

EPISODE_FILE_PATTERN = re.compile(r"^\d+\.json$")


def parse_kaggriculture_episode(
    episode_data: Dict[str, Any],
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """Parse one Kaggle episode JSON into replay-buffer transitions."""
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
                state = encode_kaggriculture_observation(
                    observation, player_id, device=device
                )
                next_state = encode_kaggriculture_observation(
                    next_observation, player_id, device=device
                )
                action = encode_kaggriculture_action(action_raw)
                reward = float(nxt.get("reward", 0.0))
                status = nxt.get("status", "ACTIVE")
                done = status in ("DONE", "TIMEOUT", "INVALID")
            except Exception:
                continue

            transitions.append(
                {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "done": done,
                }
            )

    return transitions


class KaggleEpisodesLoader:
    """Load episodes from the Kaggle Kaggriculture dataset."""

    def __init__(
        self,
        data_dir: str = "./data/kaggle_episodes",
        dataset_name: str = "kaggle/kaggriculture-episodes-index",
        api_token_path: Optional[str] = None,
        force_download: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.force_download = force_download

        if api_token_path is None:
            self.api_token_path = Path.home() / ".kaggle" / "kaggle.json"
        else:
            self.api_token_path = Path(api_token_path)

        self.episode_dir = self.data_dir / "episodes"
        self.metadata_file = self.data_dir / "metadata.json"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_if_needed(self) -> None:
        """Download the dataset if not already cached."""
        if not self.force_download and self._is_cached():
            logger.info("Dataset already cached in %s", self.data_dir)
            return

        logger.info(
            "Downloading dataset %s to %s...",
            self.dataset_name,
            self.data_dir,
        )
        self._setup_kaggle_api()
        cmd = f"kaggle datasets download -d {self.dataset_name} -p {self.data_dir}"
        exit_code = os.system(cmd)
        if exit_code != 0:
            raise RuntimeError(
                "Failed to download dataset. Check your Kaggle API credentials. "
                f"Command: {cmd}"
            )
        self._extract_data()
        logger.info("Dataset downloaded and extracted.")

    def _setup_kaggle_api(self) -> None:
        if not self.api_token_path.exists():
            raise FileNotFoundError(
                f"Kaggle API token not found at {self.api_token_path}. "
                "Obtain one from https://www.kaggle.com/settings."
            )
        import shutil

        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        kaggle_json = kaggle_dir / "kaggle.json"
        shutil.copy2(self.api_token_path, kaggle_json)
        os.chmod(kaggle_json, 0o600)

    def _is_cached(self) -> bool:
        return self.episode_dir.exists() and any(self.episode_dir.glob("*.json"))

    def _extract_data(self) -> None:
        import tarfile
        import zipfile

        for fpath in self.data_dir.iterdir():
            if fpath.name.endswith(".zip"):
                logger.info("Extracting %s...", fpath.name)
                with zipfile.ZipFile(fpath, "r") as zf:
                    zf.extractall(self.data_dir)
                fpath.unlink()
            elif fpath.name.endswith((".tar.gz", ".tgz")):
                logger.info("Extracting %s...", fpath.name)
                with tarfile.open(fpath, "r:gz") as tf:
                    tf.extractall(self.data_dir)
                fpath.unlink()

    def get_episode_files(self) -> List[Path]:
        """Return sorted episode JSON files from the episodes directory."""
        files = [
            fpath
            for fpath in self.episode_dir.glob("*.json")
            if EPISODE_FILE_PATTERN.match(fpath.name)
        ]
        return sorted(files)

    def load_into_buffer(
        self,
        buffer: Any,
        max_episodes: int = 1000,
        max_transitions: int = 500_000,
        random_seed: int = 42,
    ) -> int:
        """Load episodes into a replay buffer."""
        np.random.seed(random_seed)

        episode_files = self.get_episode_files()
        if not episode_files:
            logger.warning(
                "No episode files found in %s. Run download_expert_replays first.",
                self.episode_dir,
            )
            return 0

        logger.info("Found %d episode files in %s.", len(episode_files), self.episode_dir)
        loaded = 0

        for episode_idx, fpath in enumerate(episode_files):
            if episode_idx >= max_episodes:
                break
            if loaded >= max_transitions:
                break

            try:
                file_loaded = 0
                transitions = self._load_episodes(fpath)
                for transition in transitions:
                    if loaded >= max_transitions:
                        break
                    buffer.store(**transition)
                    loaded += 1
                    file_loaded += 1

                logger.info(
                    "Loaded %d transitions from episode %d/%d (file: %s; total: %d)",
                    file_loaded,
                    episode_idx + 1,
                    len(episode_files),
                    fpath.name,
                    loaded,
                )
            except Exception as exc:
                logger.warning("Failed to load %s: %s", fpath, exc)
                continue

        logger.info("Total transitions loaded: %d", loaded)
        return loaded

    def _load_episodes(self, fpath: Path) -> List[Dict[str, Any]]:
        if fpath.suffix == ".csv":
            return self._load_csv_episodes(fpath)
        if fpath.suffix == ".json":
            return self._load_json_episodes(fpath)
        raise ValueError(f"Unsupported file format: {fpath.suffix}")

    def _load_csv_episodes(self, fpath: Path) -> List[Dict[str, Any]]:
        import csv

        transitions: List[Dict[str, Any]] = []
        with open(fpath, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    transitions.append(
                        {
                            "state": self._parse_observation(row.get("observation", "{}")),
                            "action": self._parse_action(row.get("action", "{}")),
                            "reward": float(row.get("reward", 0)),
                            "next_state": self._parse_observation(
                                row.get("next_observation", row.get("next_obs", "{}"))
                            ),
                            "done": row.get("done", "False").lower() == "true",
                        }
                    )
                except Exception:
                    continue
        return transitions

    def _load_json_episodes(self, fpath: Path) -> List[Dict[str, Any]]:
        with open(fpath, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        if isinstance(data, dict) and "steps" in data:
            return parse_kaggriculture_episode(data)

        transitions: List[Dict[str, Any]] = []
        if isinstance(data, list):
            for episode in data:
                if isinstance(episode, dict):
                    transitions.extend(parse_kaggriculture_episode(episode))
        elif isinstance(data, dict):
            for key in ("episodes", "data"):
                if key in data and isinstance(data[key], list):
                    for episode in data[key]:
                        if isinstance(episode, dict):
                            transitions.extend(parse_kaggriculture_episode(episode))
                    break
            else:
                transitions.extend(parse_kaggriculture_episode(data))
        return transitions

    def _parse_observation(self, obs_raw: Any) -> Dict[str, Any]:
        if isinstance(obs_raw, str):
            try:
                obs_raw = json.loads(obs_raw)
            except json.JSONDecodeError:
                return {}
        if not isinstance(obs_raw, dict):
            return {}
        return obs_raw

    def _parse_action(self, action_raw: Any) -> Dict[str, Any]:
        if isinstance(action_raw, str):
            try:
                action_raw = json.loads(action_raw)
            except json.JSONDecodeError:
                return {"farmer": 0, "hands": [0] * 6, "market": 0}
        if isinstance(action_raw, dict):
            if isinstance(action_raw.get("farmer"), list):
                return encode_kaggriculture_action(action_raw)
            return {
                "farmer": int(action_raw.get("farmer", 0)),
                "hands": action_raw.get("hands", [0] * 6),
                "market": int(action_raw.get("market", 0)),
            }
        return {"farmer": 0, "hands": [0] * 6, "market": 0}

    def get_statistics(self) -> Dict[str, Any]:
        episode_files = self.get_episode_files()
        total_episodes = 0
        total_transitions = 0
        total_reward = 0.0

        for fpath in episode_files[:100]:
            try:
                transitions = self._load_episodes(fpath)
                total_episodes += 1
                total_transitions += len(transitions)
                total_reward += sum(t.get("reward", 0.0) for t in transitions)
            except Exception:
                continue

        return {
            "episode_files_found": len(episode_files),
            "episodes_sampled": total_episodes,
            "total_transitions": total_transitions,
            "avg_reward": total_reward / total_transitions if total_transitions else 0.0,
            "avg_transitions_per_episode": (
                total_transitions / total_episodes if total_episodes else 0.0
            ),
        }
