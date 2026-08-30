import argparse
import json
import logging
import os
import re
import sys
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_SCRIPT_DIR = Path(__file__).resolve().parent
for _extra in (_SCRIPT_DIR, _SCRIPT_DIR / "artifacts", _SCRIPT_DIR / "scratch"):
    if _extra.exists() and str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

# Try to import from the rebuild artifact
try:
    from kaggriculture_path_b_rebuild import (
        KaggricultureJSONParser,
        KaggricultureFeatureExtractor,
        HierarchicalDQNBranching,
        HierarchicalActionMasker,
        HierarchicalDoubleDQNLearner,
        CompetitiveRewardShaper,
        apply_hierarchical_masks
    )
    from kaggriculture_adapter import (
        COMPETITION_TURNS_PER_DAY,
        CROPS,
        EPISODE_STEPS,
        decode_path_b_action,
        parse_observation,
        resolve_training_device,
    )
    from eval_policy import (
        evaluate_ladder,
        resolve_opponents_dir,
        save_eval_report,
        win_rate_eval_from_ladder,
    )
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
    print("Successfully imported Kaggriculture Path B components.")
except ImportError as exc:
    raise ImportError(
        "Failed to import Path B training modules. Ensure kaggriculture_adapter.py, "
        "kaggriculture_path_b_rebuild.py, path_b_bootstrap.py, and eval_policy.py are on PYTHONPATH."
    ) from exc

# =============================================================================
# 1. ENVIRONMENT (official kaggle-environments only)
# =============================================================================

def _normalize_env_states(result):
    if isinstance(result, list):
        return [s if isinstance(s, dict) else {
            "observation": getattr(s, "observation", {}),
            "status": getattr(s, "status", "ACTIVE"),
            "reward": getattr(s, "reward", 0),
        } for s in result]
    return [result if isinstance(result, dict) else {
        "observation": getattr(result, "observation", {}),
        "status": getattr(result, "status", "ACTIVE"),
        "reward": getattr(result, "reward", 0),
    }]


class KaggleCompetitiveEnv:
    """Two-player wrapper around official kaggle-environments."""

    def __init__(
        self,
        max_steps: int = 720,
        seed: int = 42,
        turns_per_cycle: int = 72,
    ):
        import kaggle_environments
        self.max_steps = max_steps
        self.turns_per_cycle = int(turns_per_cycle)
        self.env = kaggle_environments.make(
            "kaggriculture",
            configuration={
                "episodeSteps": max_steps,
                "turnsPerDay": self.turns_per_cycle,
                "seed": seed,
            },
            debug=False,
        )
        self._obs: List[Dict[str, Any]] = [{}, {}]
        self._prev_money = [0.0, 0.0]

    def reset(self) -> Dict[str, Any]:
        states = _normalize_env_states(self.env.reset())
        self._obs = [
            parse_observation(states[0], player_id=0),
            parse_observation(states[1], player_id=1),
        ]
        self._prev_money = [
            float(self._obs[0]["farms"][0].get("money", 0.0)) if self._obs[0].get("farms") else 0.0,
            float(self._obs[1]["farms"][1].get("money", 0.0)) if len(self._obs[1].get("farms", [])) > 1 else 0.0,
        ]
        return self._obs[0]

    def _get_obs(self, player: int) -> Dict[str, Any]:
        return self._obs[player]

    def step(self, actions: List[Dict[str, Any]]):
        states = _normalize_env_states(self.env.step(actions))
        self._obs = [
            parse_observation(states[0], player_id=0),
            parse_observation(states[1], player_id=1),
        ]
        rewards = []
        for p in range(2):
            money = float(self._obs[p]["farms"][p].get("money", 0.0)) if self._obs[p].get("farms") else 0.0
            rewards.append((money - self._prev_money[p]) / 100.0)
            self._prev_money[p] = money
        status = states[0].get("status", "ACTIVE")
        done = status in ("DONE", "TIMEOUT", "INVALID")
        return (self._obs[0], self._obs[1]), rewards, done, {}


def create_competitive_env(
    use_kaggle: bool = True,
    max_steps: int = 720,
    seed: int = 42,
    turns_per_cycle: int = 72,
):
    if not use_kaggle:
        raise RuntimeError(
            "Offline training requires the official Kaggle simulator (use_kaggle_env=True). "
            "Install kaggle-environments and attach the kaggriculture environment."
        )
    try:
        return KaggleCompetitiveEnv(
            max_steps=max_steps,
            seed=seed,
            turns_per_cycle=turns_per_cycle,
        )
    except ImportError as exc:
        raise ImportError(
            "kaggle-environments is required for self-play training. "
            "Install with: pip install kaggle-environments"
        ) from exc


SOURCE_BOOTSTRAP = "bootstrap"
SOURCE_SELFPLAY = "selfplay"


class PrioritizedReplayBuffer:
    """Dual-partition PER buffer: 50% past-gameplay bootstrap, 50% self-play."""

    def __init__(self, capacity: int = 50000, alpha: float = 0.6, bootstrap_fraction: float = 0.5):
        self.capacity = capacity
        self.alpha = alpha
        self.bootstrap_fraction = bootstrap_fraction
        self.bootstrap_capacity = max(1, int(capacity * bootstrap_fraction))
        self.selfplay_capacity = max(1, capacity - self.bootstrap_capacity)
        self._init_partition(SOURCE_BOOTSTRAP)
        self._init_partition(SOURCE_SELFPLAY)

    def _init_partition(self, source: str) -> None:
        cap = self.bootstrap_capacity if source == SOURCE_BOOTSTRAP else self.selfplay_capacity
        setattr(self, f"{source}_buffer", [])
        setattr(self, f"{source}_pos", 0)
        setattr(self, f"{source}_priorities", np.zeros((cap,), dtype=np.float32))

    def _partition(self, source: str):
        if source == SOURCE_BOOTSTRAP:
            return self.bootstrap_buffer, self.bootstrap_pos, self.bootstrap_priorities, self.bootstrap_capacity
        return self.selfplay_buffer, self.selfplay_pos, self.selfplay_priorities, self.selfplay_capacity

    def push(
        self,
        tiles: np.ndarray,
        numeric: np.ndarray,
        action_verb: int,
        action_crop: int,
        action_hands: np.ndarray,
        action_market: np.ndarray,
        reward: float,
        next_tiles: np.ndarray,
        next_numeric: np.ndarray,
        done: bool,
        source: str = SOURCE_SELFPLAY,
    ) -> None:
        if source not in (SOURCE_BOOTSTRAP, SOURCE_SELFPLAY):
            raise ValueError(f"source must be {SOURCE_BOOTSTRAP!r} or {SOURCE_SELFPLAY!r}, got {source!r}")

        buf, pos, prios, cap = self._partition(source)
        transition = (
            tiles, numeric, action_verb, action_crop, action_hands,
            action_market, reward, next_tiles, next_numeric, done,
        )
        max_prio = float(prios[: len(buf)].max()) if buf else 1.0
        if len(buf) < cap:
            buf.append(transition)
            idx = len(buf) - 1
        else:
            idx = pos
            buf[idx] = transition
            pos = (pos + 1) % cap
            if source == SOURCE_BOOTSTRAP:
                self.bootstrap_pos = pos
            else:
                self.selfplay_pos = pos
        prios[idx] = max_prio

    def _sample_partition(
        self,
        source: str,
        batch_size: int,
        beta: float,
    ) -> Tuple[List[tuple], np.ndarray, np.ndarray, np.ndarray]:
        buf, _, prios, _ = self._partition(source)
        if not buf or batch_size <= 0:
            return [], np.array([], dtype=int), np.array([], dtype=np.float32), np.array([], dtype=int)

        n = min(batch_size, len(buf))
        prios_slice = prios[: len(buf)].astype(np.float64)
        probs = prios_slice ** self.alpha
        probs /= probs.sum()
        local_indices = np.random.choice(len(buf), n, p=probs, replace=n > len(buf))
        samples = [buf[i] for i in local_indices]
        weights = (len(buf) * probs[local_indices]) ** (-beta)
        weights /= weights.max()
        global_indices = np.array([(0 if source == SOURCE_BOOTSTRAP else 1), *local_indices], dtype=object)
        # Encode global index as (source_flag, local_idx) for priority updates
        tagged_indices = np.array([(0 if source == SOURCE_BOOTSTRAP else 1, int(i)) for i in local_indices])
        return samples, tagged_indices, np.asarray(weights, dtype=np.float32), local_indices

    def _batch_from_samples(self, samples: List[tuple], weights: np.ndarray) -> Dict[str, torch.Tensor]:
        tiles_b, numeric_b, act_v_b, act_c_b, act_h_b, act_m_b, r_b, n_tiles_b, n_num_b, d_b = zip(*samples)
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
            "weights": torch.as_tensor(weights, dtype=torch.float32),
        }

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        n_bootstrap = batch_size // 2
        n_selfplay = batch_size - n_bootstrap
        b_samples, b_indices, b_weights, _ = self._sample_partition(SOURCE_BOOTSTRAP, n_bootstrap, beta)
        s_samples, s_indices, s_weights, _ = self._sample_partition(SOURCE_SELFPLAY, n_selfplay, beta)
        samples = b_samples + s_samples
        if not samples:
            return {}, np.array([]), np.array([])

        indices = np.concatenate([b_indices, s_indices]) if len(b_indices) and len(s_indices) else (
            b_indices if len(b_indices) else s_indices
        )
        weights = np.concatenate([b_weights, s_weights]) if len(b_weights) and len(s_weights) else (
            b_weights if len(b_weights) else s_weights
        )
        return self._batch_from_samples(samples, weights), indices, weights

    def sample_uniform(self, batch_size: int, source: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Uniform sample for BC — defaults to bootstrap partition."""
        src = source or SOURCE_BOOTSTRAP
        buf, _, _, _ = self._partition(src)
        if not buf:
            return {}
        n = min(batch_size, len(buf))
        indices = np.random.choice(len(buf), n, replace=False)
        samples = [buf[i] for i in indices]
        weights = np.ones(n, dtype=np.float32)
        return self._batch_from_samples(samples, weights)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        indices = np.asarray(indices)
        priorities = np.asarray(priorities)
        if indices.size == 0 or priorities.size == 0:
            return
        if indices.ndim == 1 and indices.dtype == object:
            rows = list(indices)
        elif indices.ndim == 2 and indices.shape[-1] == 2:
            rows = indices
        else:
            return
        for idx, prio in zip(rows, priorities.reshape(-1)):
            try:
                source_flag, local_idx = int(idx[0]), int(idx[1])
            except (TypeError, IndexError, ValueError):
                continue
            source = SOURCE_BOOTSTRAP if source_flag == 0 else SOURCE_SELFPLAY
            buf, _, prios, _ = self._partition(source)
            n = len(buf)
            if 0 <= local_idx < n:
                prios[local_idx] = max(float(prio), 1e-6)

    @property
    def bootstrap_size(self) -> int:
        return len(self.bootstrap_buffer)

    @property
    def selfplay_size(self) -> int:
        return len(self.selfplay_buffer)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "bootstrap_fraction": self.bootstrap_fraction,
            "bootstrap_capacity": self.bootstrap_capacity,
            "selfplay_capacity": self.selfplay_capacity,
            "bootstrap_pos": self.bootstrap_pos,
            "selfplay_pos": self.selfplay_pos,
            "bootstrap_buffer": self.bootstrap_buffer,
            "selfplay_buffer": self.selfplay_buffer,
            "bootstrap_priorities": self.bootstrap_priorities[: len(self.bootstrap_buffer)].copy(),
            "selfplay_priorities": self.selfplay_priorities[: len(self.selfplay_buffer)].copy(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "bootstrap_buffer" in state:
            self.capacity = int(state["capacity"])
            self.alpha = float(state["alpha"])
            self.bootstrap_fraction = float(state.get("bootstrap_fraction", 0.5))
            self.bootstrap_capacity = int(state.get("bootstrap_capacity", max(1, self.capacity // 2)))
            self.selfplay_capacity = int(state.get("selfplay_capacity", self.capacity - self.bootstrap_capacity))
            self.bootstrap_pos = int(state.get("bootstrap_pos", 0))
            self.selfplay_pos = int(state.get("selfplay_pos", 0))
            self.bootstrap_buffer = list(state.get("bootstrap_buffer", []))
            self.selfplay_buffer = list(state.get("selfplay_buffer", []))
            self.bootstrap_priorities = np.zeros((self.bootstrap_capacity,), dtype=np.float32)
            self.selfplay_priorities = np.zeros((self.selfplay_capacity,), dtype=np.float32)
            bp = state.get("bootstrap_priorities")
            sp = state.get("selfplay_priorities")
            if bp is not None and len(self.bootstrap_buffer):
                self.bootstrap_priorities[: len(self.bootstrap_buffer)] = np.asarray(bp, dtype=np.float32)
            if sp is not None and len(self.selfplay_buffer):
                self.selfplay_priorities[: len(self.selfplay_buffer)] = np.asarray(sp, dtype=np.float32)
            return

        # Legacy single-buffer checkpoint → assign to bootstrap partition
        self.capacity = int(state["capacity"])
        self.alpha = float(state["alpha"])
        self.bootstrap_fraction = 0.5
        self.bootstrap_capacity = max(1, self.capacity // 2)
        self.selfplay_capacity = max(1, self.capacity - self.bootstrap_capacity)
        legacy = list(state.get("buffer", []))
        self.bootstrap_buffer = legacy[: self.bootstrap_capacity]
        self.selfplay_buffer = legacy[self.bootstrap_capacity : self.bootstrap_capacity + self.selfplay_capacity]
        self.bootstrap_pos = len(self.bootstrap_buffer) % self.bootstrap_capacity
        self.selfplay_pos = len(self.selfplay_buffer) % self.selfplay_capacity
        self.bootstrap_priorities = np.zeros((self.bootstrap_capacity,), dtype=np.float32)
        self.selfplay_priorities = np.zeros((self.selfplay_capacity,), dtype=np.float32)
        legacy_p = state.get("priorities")
        if legacy_p is not None:
            legacy_p = np.asarray(legacy_p, dtype=np.float32)
            n_b = min(len(legacy_p), len(self.bootstrap_buffer))
            n_s = min(max(0, len(legacy_p) - n_b), len(self.selfplay_buffer))
            if n_b:
                self.bootstrap_priorities[:n_b] = legacy_p[:n_b]
            if n_s:
                self.selfplay_priorities[:n_s] = legacy_p[n_b : n_b + n_s]

    def __len__(self) -> int:
        return len(self.bootstrap_buffer) + len(self.selfplay_buffer)

    def clear(self, source: Optional[str] = None) -> None:
        if source is None:
            self._init_partition(SOURCE_BOOTSTRAP)
            self._init_partition(SOURCE_SELFPLAY)
        elif source == SOURCE_BOOTSTRAP:
            self._init_partition(SOURCE_BOOTSTRAP)
        elif source == SOURCE_SELFPLAY:
            self._init_partition(SOURCE_SELFPLAY)
        else:
            raise ValueError(f"unknown source: {source!r}")


# =============================================================================
# 3. SELF-PLAY TRAINER COORDINATOR
# =============================================================================

def setup_experiment_dirs(experiment_dir: Path) -> Dict[str, Path]:
    """Create experiment output layout matching train.py."""
    experiment_dir = Path(experiment_dir)
    subdirs = {
        "root": experiment_dir,
        "models": experiment_dir / "models",
        "checkpoints": experiment_dir / "checkpoints",
        "logs": experiment_dir / "logs",
        "metrics": experiment_dir / "metrics",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


class SelfPlayCoordinator:
    """
    Coordinates self-play training by managing an opponent pool of historical checkpoints
    and selecting checkpoints to play against the current online model.
    """
    def __init__(self,
                 latent_dim: int = 512,
                 shared_dim: int = 256,
                 checkpoint_dir: Optional[str] = None):
        self.latent_dim = latent_dim
        self.shared_dim = shared_dim
        if checkpoint_dir is None:
            checkpoint_dir = str(_SCRIPT_DIR / "experiments" / "self_play" / "checkpoints")
        self.checkpoint_dir = str(checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.opponent_pool = []

    def restore_opponent_pool(self, paths: Optional[List[str]] = None) -> None:
        """Rebuild opponent pool from saved paths or checkpoint directory."""
        if paths:
            self.opponent_pool = [p for p in paths if os.path.exists(p)]
            return
        pattern = re.compile(r"^checkpoint_ep_(\d+)\.pt$")
        discovered = []
        for name in os.listdir(self.checkpoint_dir):
            if pattern.match(name):
                discovered.append(os.path.join(self.checkpoint_dir, name))
        self.opponent_pool = sorted(discovered)

    def save_checkpoint(self, online_net: nn.Module, episode: int):
        path = os.path.join(self.checkpoint_dir, f"checkpoint_ep_{episode}.pt")
        torch.save(online_net.state_dict(), path)
        if path not in self.opponent_pool:
            self.opponent_pool.append(path)
        print(f"[Self-Play] Saved agent checkpoint to: {path}")
        print(f"[Self-Play] Opponent pool size: {len(self.opponent_pool)}")
        return path

    def select_opponent(self) -> str:
        """
        Returns path to a random historical checkpoint, or None to play against
        the active network weights (pure self-play).
        """
        if not self.opponent_pool:
            return None
        # 80% chance to select historical, 20% to select active network
        if random.random() < 0.8:
            return random.choice(self.opponent_pool)
        return None

    def get_agent_policy_fn(self, 
                            checkpoint_path: str, 
                            online_net: HierarchicalDQNBranching, 
                            device: torch.device):
        """
        Generates an agent execution policy function mapping observation to action.
        """
        parser = KaggricultureJSONParser()
        
        # Instantiate opponent network
        opp_extractor = KaggricultureFeatureExtractor(latent_dim=self.latent_dim)
        opp_net = HierarchicalDQNBranching(opp_extractor, latent_dim=self.latent_dim, shared_dim=self.shared_dim).to(device)
        
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            opp_net.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            # Use active online network weights
            opp_net.load_state_dict(online_net.state_dict())
        
        opp_net.eval()

        @torch.no_grad()
        def opponent_policy(obs: Dict[str, Any]) -> Dict[str, Any]:
            parsed = parser.parse_observation(obs)
            tiles_t = torch.as_tensor(parsed["tiles"], dtype=torch.float32, device=device).unsqueeze(0)
            numeric_t = torch.as_tensor(parsed["numeric"], dtype=torch.float32, device=device).unsqueeze(0)
            
            # Generate action Q-values (in eval mode)
            q_out = opp_net(tiles_t, numeric_t)
            
            # Apply Masks
            masks = HierarchicalActionMasker.get_dynamic_masks(obs)
            masked_q = apply_hierarchical_masks(q_out, masks, device)
            
            # Select argmax action values
            verb_idx = int(masked_q["farmer_verb"].argmax(dim=-1).item())
            crop_idx = int(masked_q["crop_parameter"].argmax(dim=-1).item())
            
            hands_indices = []
            for i in range(opp_net.num_hands):
                hands_indices.append(int(masked_q["hands"][i].argmax(dim=-1).item()))
                
            market_indices = []
            market_seq_argmax = masked_q["market"].argmax(dim=-1).squeeze(0) # (max_market_orders,)
            for step in range(opp_net.max_market_orders):
                market_indices.append(int(market_seq_argmax[step].item()))
                
            # Translate to game format
            return decode_path_b_action(
                verb_idx, crop_idx, hands_indices, market_indices, obs
            )
            
        return opponent_policy


TRAINING_STATE_FILENAME = "training_state_latest.pt"
_CHECKPOINT_EP_PATTERN = re.compile(r"checkpoint_ep_(\d+)\.pt$")


def _load_state_dict(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _training_state_path(experiment_root: Path) -> Path:
    return experiment_root / "checkpoints" / TRAINING_STATE_FILENAME


def _episode_from_checkpoint_name(path: Path) -> int:
    match = _CHECKPOINT_EP_PATTERN.match(path.name)
    if match:
        return int(match.group(1))
    return 0


def _load_episode_metrics(experiment_root: Path) -> List[Dict[str, float]]:
    metrics_path = experiment_root / "metrics" / "episode_metrics.json"
    if not metrics_path.exists():
        return []
    try:
        with open(metrics_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return list(data.get("episodes", []))
    except (json.JSONDecodeError, OSError):
        return []


def resolve_resume_path(resume: str) -> Tuple[Path, Path, str]:
    """Resolve --resume to (experiment_dir, checkpoint_file, kind).

    kind is ``full`` for training_state_latest.pt or ``weights`` for weight-only files.
    """
    path = Path(resume).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Resume path not found: {path}")

    if path.is_dir():
        exp_dir = path
        state_file = _training_state_path(exp_dir)
        if state_file.exists():
            return exp_dir, state_file, "full"
        weight_files = sorted(
            exp_dir.glob("checkpoints/checkpoint_ep_*.pt"),
            key=_episode_from_checkpoint_name,
        )
        if weight_files:
            return exp_dir, weight_files[-1], "weights"
        model_file = exp_dir / "models" / "model.pth"
        if model_file.exists():
            return exp_dir, model_file, "weights"
        raise FileNotFoundError(
            f"No resume checkpoint in {exp_dir}. Expected "
            f"checkpoints/{TRAINING_STATE_FILENAME} or checkpoints/checkpoint_ep_*.pt"
        )

    if path.name == TRAINING_STATE_FILENAME:
        exp_dir = path.parent.parent
        return exp_dir, path, "full"
    if _CHECKPOINT_EP_PATTERN.match(path.name):
        exp_dir = path.parent.parent
        return exp_dir, path, "weights"
    if path.name == "model.pth":
        exp_dir = path.parent.parent
        return exp_dir, path, "weights"

    raise ValueError(
        f"Unsupported resume file: {path}. Use an experiment directory, "
        f"{TRAINING_STATE_FILENAME}, checkpoint_ep_N.pt, or model.pth"
    )


def save_training_state(
    path: Path,
    *,
    last_completed_episode: int,
    online_net: nn.Module,
    target_net: nn.Module,
    optimizer: optim.Optimizer,
    buffer: PrioritizedReplayBuffer,
    coordinator: SelfPlayCoordinator,
    episode_metrics: List[Dict[str, float]],
    config: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng_state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()

    payload = {
        "version": 1,
        "last_completed_episode": last_completed_episode,
        "online_net": online_net.state_dict(),
        "target_net": target_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "buffer": buffer.state_dict(),
        "opponent_pool": list(coordinator.opponent_pool),
        "episode_metrics": episode_metrics,
        "config": config,
        "rng_state": rng_state,
        "saved_at": datetime.now().isoformat(),
    }
    torch.save(payload, path)


def _coerce_cpu_byte_rng_state(state: Any) -> torch.Tensor:
    """Restore torch RNG state saved across devices / PyTorch versions."""
    if isinstance(state, torch.Tensor):
        return state.detach().cpu().to(dtype=torch.uint8)
    return torch.as_tensor(state, dtype=torch.uint8)


def _restore_rng_state(rng_state: Dict[str, Any]) -> None:
    if not rng_state:
        return
    try:
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "torch" in rng_state:
            torch.set_rng_state(_coerce_cpu_byte_rng_state(rng_state["torch"]))
        if torch.cuda.is_available() and "torch_cuda" in rng_state:
            cuda_states = [
                _coerce_cpu_byte_rng_state(s) for s in rng_state["torch_cuda"]
            ]
            torch.cuda.set_rng_state_all(cuda_states)
    except (TypeError, ValueError) as exc:
        logger.warning("RNG state restore skipped (non-fatal): %s", exc)


def load_training_state(
    path: Path,
    device: torch.device,
    online_net: nn.Module,
    target_net: nn.Module,
    optimizer: optim.Optimizer,
    buffer: PrioritizedReplayBuffer,
    coordinator: SelfPlayCoordinator,
) -> Tuple[int, List[Dict[str, float]], Dict[str, Any]]:
    """Load full training state; returns (last_completed_episode, metrics, config)."""
    payload = torch.load(path, map_location=device, weights_only=False)
    online_net.load_state_dict(payload["online_net"])
    target_net.load_state_dict(payload["target_net"])
    optimizer.load_state_dict(payload["optimizer"])
    buffer.load_state_dict(payload["buffer"])
    coordinator.restore_opponent_pool(payload.get("opponent_pool"))
    _restore_rng_state(payload.get("rng_state", {}))

    return (
        int(payload["last_completed_episode"]),
        list(payload.get("episode_metrics", [])),
        dict(payload.get("config", {})),
    )


def load_weights_checkpoint(
    path: Path,
    device: torch.device,
    online_net: nn.Module,
    target_net: nn.Module,
    learner: HierarchicalDoubleDQNLearner,
    coordinator: SelfPlayCoordinator,
    experiment_root: Path,
) -> Tuple[int, List[Dict[str, float]]]:
    """Load weights-only checkpoint (legacy or model.pth)."""
    online_net.load_state_dict(_load_state_dict(path, device))
    target_net.load_state_dict(online_net.state_dict())
    learner.target.load_state_dict(online_net.state_dict())
    coordinator.restore_opponent_pool()

    last_episode = _episode_from_checkpoint_name(path)
    if last_episode == 0 and path.name == "model.pth":
        metrics = _load_episode_metrics(experiment_root)
        if metrics:
            last_episode = int(metrics[-1].get("episode", len(metrics)))
    return last_episode, _load_episode_metrics(experiment_root)


# =============================================================================
# 5. CORE TRAINING RUNNER
# =============================================================================

def train_self_play(total_episodes: int = 15,
                    learning_start_episodes: int = 2,
                    batch_size: int = 32,
                    checkpoint_interval: int = 5,
                    experiment_dir: str = "experiments/self_play",
                    seed: int = 42,
                    device_name: str = "auto",
                    use_kaggle_env: bool = False,
                    max_episode_steps: int = 720,
                    turns_per_cycle: int = 72,
                    n_eval_episodes: int = 5,
                    resume: Optional[str] = None,
                    bootstrap_episodes: Optional[int] = 0,
                    bootstrap_transitions: Optional[int] = 50_000,
                    data_dir: str = "./data/kaggle_episodes",
                    download_bootstrap: bool = False,
                    bc_epochs: int = 15,
                    bc_batch_size: int = 64,
                    bc_steps_per_epoch: Optional[int] = None,
                    buffer_capacity: int = 10_000,
                    metadata_path: Optional[str] = None,
                    bootstrap_top_per_day: Optional[int] = 20,
                    bootstrap_passes: int = 1,
                    bc_epochs_per_pass: int = 1,
                    verbose: bool = False,
                    code_src: Optional[str] = None,
                    bootstrap_mode: str = "streaming",
                    bootstrap_days_per_run: int = 3,
                    publish_code_dataset: bool = False,
                    opponents_dir: Optional[str] = None,
                    ladder_eval_episodes: int = 0,
                    ladder_win_rate_target: float = 0.5,
                    min_self_play_episodes: int = 0):
    """
    Coordinates self-play game loops, reward shaping, experience replay storage,
    and DDQN model optimization.

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

    # Exploration parameter decay. After BC, keep ε low so random
    # self-play does not wipe the cloned policy before DQN can refine it.
    eps_start = 0.25 if bc_loss_history else 1.0
    eps_end = 0.05
    eps_decay_steps = max(1, total_episodes - learning_start_episodes)
    if bc_loss_history:
        logger.info(
            "BC completed (%d epoch losses); self-play eps_start=%.2f (was 1.0 without BC)",
            len(bc_loss_history),
            eps_start,
        )

    # Initialize competitive self-play environment (Kaggle sim or offline mock).
    env = create_competitive_env(
        use_kaggle=use_kaggle_env,
        max_steps=max_episode_steps if use_kaggle_env else min(50, max_episode_steps),
        seed=seed + start_episode,
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

    if start_episode < total_episodes:
        logger.info(
            "--- BEGINNING KAGGRICULTURE SELF-PLAY PIPELINE (episodes %d → %d) ---",
            start_episode + 1, total_episodes,
        )
    for ep in range(start_episode + 1, total_episodes + 1):
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
            shaped_reward_p0 = reward_shaper.shape_reward(obs_p0, raw_reward_p0)
            
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

        # Periodically save models and append to Self-Play pool
        if ep % checkpoint_interval == 0:
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
        json.dump({**config, "last_completed_episode": start_episode if start_episode >= total_episodes else total_episodes}, fh, indent=2)

    final_model_path = dirs["models"] / "model.pth"
    torch.save(online_net.state_dict(), final_model_path)
    logger.info("Final model saved to: %s", final_model_path)

    def _path_b_policy(obs: Dict[str, Any]) -> Dict[str, Any]:
        online_net.eval()
        parsed = parser.parse_observation(obs)
        tiles_t = torch.as_tensor(parsed["tiles"], dtype=torch.float32, device=device).unsqueeze(0)
        numeric_t = torch.as_tensor(parsed["numeric"], dtype=torch.float32, device=device).unsqueeze(0)
        masks = HierarchicalActionMasker.get_dynamic_masks(obs)
        with torch.no_grad():
            q_out = online_net(tiles_t, numeric_t)
            masked_q = apply_hierarchical_masks(q_out, masks, device)
            verb_idx = int(masked_q["farmer_verb"].argmax(dim=-1).item())
            crop_idx = int(masked_q["crop_parameter"].argmax(dim=-1).item())
            hands_indices = [
                int(masked_q["hands"][i].argmax(dim=-1).item())
                for i in range(online_net.num_hands)
            ]
            market_seq = masked_q["market"].argmax(dim=-1).squeeze(0)
            market_indices = [int(market_seq[t].item()) for t in range(online_net.max_market_orders)]
        return decode_path_b_action(verb_idx, crop_idx, hands_indices, market_indices, obs)

    # Post-training eval is league-only (``opponents/``). ``win_rate_eval.json`` is a
    # thin aggregate of ``ladder_eval.json`` — never SB3-vs-env / random baseline.
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

    agent_path = dirs["root"] / "agent.py"
    _export_path_b_agent(agent_path, dirs["root"], code_src=code_src)
    logger.info("Agent export saved to: %s", agent_path)

    metrics_path = save_episode_metrics(dirs["root"], episode_metrics)
    logger.info("Episode metrics saved to: %s", metrics_path)

    config["last_completed_episode"] = total_episodes
    progress_path = progress.finalize_run(config)
    logger.info("Cumulative training progress saved to: %s", progress_path)
    logger.info("--- SELF-PLAY TRAINING LOOP COMPLETED ---")


def _resolve_code_src(explicit: Optional[str] = None) -> Path:
    """Locate read-only Kaggle code dataset (adapter modules)."""
    if explicit:
        return Path(explicit)
    for candidate in (
        Path("/kaggle/input/datasets/scottweeden/kaggriculture-self-training-code"),
        Path("/kaggle/input/kaggriculture-self-training-code"),
        Path(__file__).resolve().parent,
    ):
        if (candidate / "kaggriculture_adapter.py").exists():
            return candidate
    return Path(__file__).resolve().parent


def _export_path_b_agent(
    agent_path: Path,
    experiment_root: Path,
    *,
    code_src: Optional[str] = None,
) -> None:
    """Write a minimal Kaggle submission agent using shared adapter decode."""
    import shutil

    src_root = _resolve_code_src(code_src)
    for module_name in ("kaggriculture_adapter.py", "kaggriculture_path_b_rebuild.py"):
        src = src_root / module_name
        dst = experiment_root / module_name
        if not src.exists():
            raise FileNotFoundError(f"Missing adapter module in code dataset: {src}")
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

    agent_code = f'''"""Kaggle Kaggriculture Path B agent export."""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggriculture_adapter import decode_path_b_action, parse_observation
from kaggriculture_path_b_rebuild import (
    KaggricultureJSONParser,
    KaggricultureFeatureExtractor,
    HierarchicalDQNBranching,
    HierarchicalActionMasker,
    apply_hierarchical_masks,
)


class Agent:
    def __init__(self):
        self.device = torch.device("cpu")
        self.parser = KaggricultureJSONParser()
        extractor = KaggricultureFeatureExtractor(latent_dim=512)
        self.net = HierarchicalDQNBranching(extractor, latent_dim=512, shared_dim=256)
        model_path = os.path.join(os.path.dirname(__file__), "models", "model.pth")
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.net.eval()

    def act(self, obs, action_space=None):
        agent_obs = parse_observation(obs)
        parsed = self.parser.parse_observation(agent_obs)
        tiles_t = torch.as_tensor(parsed["tiles"], dtype=torch.float32, device=self.device).unsqueeze(0)
        numeric_t = torch.as_tensor(parsed["numeric"], dtype=torch.float32, device=self.device).unsqueeze(0)
        masks = HierarchicalActionMasker.get_dynamic_masks(agent_obs)
        with torch.no_grad():
            q_out = self.net(tiles_t, numeric_t)
            masked_q = apply_hierarchical_masks(q_out, masks, self.device)
            verb_idx = int(masked_q["farmer_verb"].argmax(dim=-1).item())
            crop_idx = int(masked_q["crop_parameter"].argmax(dim=-1).item())
            hands = [int(masked_q["hands"][i].argmax(dim=-1).item()) for i in range(self.net.num_hands)]
            market_seq = masked_q["market"].argmax(dim=-1).squeeze(0)
            market = [int(market_seq[t].item()) for t in range(self.net.max_market_orders)]
        return decode_path_b_action(verb_idx, crop_idx, hands, market, agent_obs)
'''
    agent_path.write_text(agent_code, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kaggriculture hierarchical DQN self-play training (Path B rebuild)"
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="experiments/self_play",
        help="Experiment output directory (default: experiments/self_play)",
    )
    parser.add_argument("--total-episodes", type=int, default=15)
    parser.add_argument("--learning-start-episodes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps", "mlx"],
        default="auto",
    )
    parser.add_argument("--use-kaggle-env", action="store_true")
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument(
        "--turns-per-cycle",
        type=int,
        default=72,
        help=(
            "Kinematic end-of-cycle refresh period (engine turnsPerDay). "
            "Default 72 = 3×4×6; season is max_episode_steps / turns_per_cycle cycles "
            "(10×72=720). Competition ladder parity uses 24."
        ),
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=5,
        help=(
            "Episodes per league opponent when --ladder-eval-episodes is 0. "
            "Never runs a random baseline; requires opponents/."
        ),
    )
    parser.add_argument(
        "--bootstrap-episodes",
        type=int,
        default=0,
        help="Load up to N Kaggle episode JSONs into replay buffer before self-play (0=off)",
    )
    parser.add_argument(
        "--bootstrap-transitions",
        type=int,
        default=50_000,
        help="Maximum expert transitions to load during bootstrap",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/kaggle_episodes",
        help="Directory containing episodes/*.json for bootstrap",
    )
    parser.add_argument(
        "--download-bootstrap",
        action="store_true",
        help="Download Kaggle episodes dataset before bootstrap",
    )
    parser.add_argument(
        "--bootstrap-passes",
        type=int,
        default=1,
        help="Shuffle-fill-BC passes over corpus (>1 enables streaming bootstrap)",
    )
    parser.add_argument(
        "--bc-epochs-per-pass",
        type=int,
        default=1,
        help="BC epochs after each bootstrap pass (used when --bootstrap-passes > 1)",
    )
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=15,
        help="Behavioral cloning pretrain epochs on bootstrapped buffer (0=skip)",
    )
    parser.add_argument(
        "--bc-batch-size",
        type=int,
        default=64,
        help="Batch size for BC pretrain",
    )
    parser.add_argument(
        "--bc-steps-per-epoch",
        type=int,
        default=None,
        help="Cap BC gradient steps per epoch (default: buffer // batch_size)",
    )
    parser.add_argument(
        "--buffer-capacity",
        type=int,
        default=10_000,
        help="Replay buffer capacity",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=None,
        help="Merged metadata.json for score-ranked bootstrap ordering",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Resume training from an experiment directory or checkpoint file. "
            "Prefers checkpoints/training_state_latest.pt (full state). "
            "--total-episodes is the cumulative target episode count."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging for bootstrap, BC, and self-play steps",
    )
    parser.add_argument(
        "--code-src",
        type=str,
        default=None,
        help=(
            "Read-only code dataset root for kaggriculture_adapter.py "
            "(default: /kaggle/input/datasets/scottweeden/kaggriculture-self-training-code)"
        ),
    )
    parser.add_argument(
        "--bootstrap-mode",
        type=str,
        choices=["streaming", "daily_incremental"],
        default="streaming",
        help=(
            "Bootstrap strategy: streaming multi-pass corpus walk, or daily_incremental "
            "(all episodes from N random unseen days per run, persisted in bootstrap_state.json)"
        ),
    )
    parser.add_argument(
        "--bootstrap-days-per-run",
        type=int,
        default=3,
        help="Days to bootstrap per run when --bootstrap-mode daily_incremental",
    )
    parser.add_argument(
        "--publish-code-dataset",
        action="store_true",
        help="After bootstrapping new days, publish model artifacts to code dataset via Kaggle CLI",
    )
    parser.add_argument(
        "--opponents-dir",
        type=str,
        default=None,
        help="Reference ladder opponents/ directory for post-training ladder eval",
    )
    parser.add_argument(
        "--ladder-eval-episodes",
        type=int,
        default=0,
        help="Head-to-head episodes per reference opponent after training (0=skip)",
    )
    parser.add_argument(
        "--ladder-win-rate-target",
        type=float,
        default=0.5,
        help="Win rate threshold counted as clearing an opponent in ladder eval",
    )
    parser.add_argument(
        "--min-self-play-episodes",
        type=int,
        default=0,
        help="When resuming past total_episodes, run at least this many additional self-play episodes",
    )
    args = parser.parse_args()

    experiment_dir = args.experiment_dir
    if args.resume:
        exp_root, _, _ = resolve_resume_path(args.resume)
        if Path(args.experiment_dir).resolve() != exp_root.resolve():
            if args.experiment_dir != "experiments/self_play":
                logging.warning(
                    "Both --experiment-dir and --resume given; using resume directory %s",
                    exp_root,
                )
            experiment_dir = str(exp_root)

    train_self_play(
        total_episodes=args.total_episodes,
        learning_start_episodes=args.learning_start_episodes,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        experiment_dir=experiment_dir,
        seed=args.seed,
        device_name=args.device,
        use_kaggle_env=args.use_kaggle_env,
        max_episode_steps=args.max_episode_steps,
        turns_per_cycle=args.turns_per_cycle,
        n_eval_episodes=args.n_eval_episodes,
        resume=args.resume,
        bootstrap_episodes=args.bootstrap_episodes,
        bootstrap_transitions=args.bootstrap_transitions,
        data_dir=args.data_dir,
        download_bootstrap=args.download_bootstrap,
        bc_epochs=args.bc_epochs,
        bc_batch_size=args.bc_batch_size,
        bc_steps_per_epoch=args.bc_steps_per_epoch,
        buffer_capacity=args.buffer_capacity,
        metadata_path=args.metadata_path,
        bootstrap_passes=args.bootstrap_passes,
        bc_epochs_per_pass=args.bc_epochs_per_pass,
        verbose=args.verbose,
        code_src=args.code_src,
        bootstrap_mode=args.bootstrap_mode,
        bootstrap_days_per_run=args.bootstrap_days_per_run,
        publish_code_dataset=args.publish_code_dataset,
        opponents_dir=args.opponents_dir,
        ladder_eval_episodes=args.ladder_eval_episodes,
        ladder_win_rate_target=args.ladder_win_rate_target,
        min_self_play_episodes=args.min_self_play_episodes,
    )


if __name__ == "__main__":
    main()
