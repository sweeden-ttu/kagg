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
        CROPS,
        decode_path_b_action,
        parse_observation,
        resolve_training_device,
    )
    from eval_policy import (
        evaluate_ladder,
        evaluate_win_rate,
        random_baseline_policy,
        resolve_opponents_dir,
        save_eval_report,
    )
    from path_b_bootstrap import (
        bootstrap_path_b_replay_buffer,
        incremental_daily_bootstrap_bc,
        resolve_bootstrap_episode_files,
        run_bc_pretrain,
        stream_bootstrap_bc_pretrain,
    )
    print("Successfully imported Kaggriculture Path B components.")
except ImportError as exc:
    raise ImportError(
        "Failed to import Path B training modules. Ensure kaggriculture_adapter.py, "
        "kaggriculture_path_b_rebuild.py, path_b_bootstrap.py, and eval_policy.py are on PYTHONPATH."
    ) from exc

# =============================================================================
# 1. ENVIRONMENT (Kaggle or mock fallback)
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

    def __init__(self, max_steps: int = 720, seed: int = 42):
        import kaggle_environments
        self.max_steps = max_steps
        self.env = kaggle_environments.make(
            "kaggriculture",
            configuration={"episodeSteps": max_steps, "seed": seed},
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


def create_competitive_env(use_kaggle: bool = True, max_steps: int = 720, seed: int = 42):
    if use_kaggle:
        try:
            return KaggleCompetitiveEnv(max_steps=max_steps, seed=seed)
        except ImportError:
            logging.getLogger(__name__).warning(
                "kaggle-environments unavailable; falling back to mock env"
            )
    return MockKaggricultureEnv(max_steps=max_steps)


# Legacy mock kept for offline smoke tests

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer storing structured state/action pairs 
    with temporal-difference error scaling to accelerate learning.
    """
    def __init__(self, capacity: int = 50000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, 
             tiles: np.ndarray, 
             numeric: np.ndarray, 
             action_verb: int, 
             action_crop: int, 
             action_hands: np.ndarray, 
             action_market: np.ndarray, 
             reward: float, 
             next_tiles: np.ndarray, 
             next_numeric: np.ndarray, 
             done: bool):
        """
        Saves a transition. Priority is initialized to the max priority currently in buffer.
        """
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        transition = (
            tiles, numeric, action_verb, action_crop, action_hands, 
            action_market, reward, next_tiles, next_numeric, done
        )
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
            
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        if len(self.buffer) == 0:
            return {}, np.array([]), np.array([])
            
        prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate Importance Sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        # Unzip samples
        tiles_b, numeric_b, act_v_b, act_c_b, act_h_b, act_m_b, r_b, n_tiles_b, n_num_b, d_b = zip(*samples)
        
        batch = {
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
            "weights": torch.as_tensor(weights, dtype=torch.float32)
        }
        
        return batch, indices, weights

    def sample_uniform(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Uniform random sample for behavioral cloning (no PER weights)."""
        if len(self.buffer) == 0:
            return {}

        n = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), n, replace=False)
        samples = [self.buffer[idx] for idx in indices]

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
        }

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = max(prio, 1e-6)

    def state_dict(self) -> Dict[str, Any]:
        n = len(self.buffer)
        return {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "pos": self.pos,
            "buffer": self.buffer,
            "priorities": self.priorities[:n].copy(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.capacity = int(state["capacity"])
        self.alpha = float(state["alpha"])
        self.pos = int(state["pos"])
        self.buffer = state["buffer"]
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        prios = state.get("priorities")
        if prios is not None and len(self.buffer):
            self.priorities[: len(self.buffer)] = np.asarray(prios, dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def clear(self) -> None:
        """Drop all stored transitions (used between shuffle bootstrap passes)."""
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)


# =============================================================================
# 3. MOCK ENVIRONMENT (offline fallback)
# =============================================================================

class MockKaggricultureEnv:
    """
    Simulation engine replicating the official Kaggriculture game mechanics,
    providing identical reset/step nested dictionary observation spaces.
    Guarantees local testing capability if kaggle-environments is absent.
    """
    def __init__(self, max_steps: int = 100):
        self.max_steps = max_steps
        self.current_step = 0
        self.player_money = [3000.0, 3000.0]
        self._prev_money = [3000.0, 3000.0]
        self.grid_size = (10, 10)
        self.farmer_pos = [[2, 2], [7, 7]]
        self.tiles_status = [
            [[{"kind": None, "crop": None, "watered_today": False, "yield_units": 0.0} for _ in range(10)] for _ in range(10)],
            [[{"kind": None, "crop": None, "watered_today": False, "yield_units": 0.0} for _ in range(10)] for _ in range(10)]
        ]
        self.seeds_inv = [{c: (10 if c == "WHEAT" else 0) for c in CROPS} for _ in range(2)]
        self.shed_inv = [{c: 0 for c in CROPS} for _ in range(2)]
        self.market_prices = {c: float(10 + i * 5) for i, c in enumerate(CROPS)}
        self.market_inventory = {c: 1000 for c in CROPS}
        self.unlocked_shops = ["BAKERY", "GROCERY"]
        self.hands_list = [[{"id": 0}], [{"id": 0}]]

    def reset(self) -> Dict[str, Any]:
        self.current_step = 0
        self.player_money = [3000.0, 3000.0]
        self._prev_money = [3000.0, 3000.0]
        self.farmer_pos = [[2, 2], [7, 7]]
        self.tiles_status = [
            [[{"kind": None, "crop": None, "watered_today": False, "yield_units": 0.0} for _ in range(10)] for _ in range(10)],
            [[{"kind": None, "crop": None, "watered_today": False, "yield_units": 0.0} for _ in range(10)] for _ in range(10)]
        ]
        self.seeds_inv = [{c: (10 if c == "WHEAT" else 0) for c in CROPS} for _ in range(2)]
        self.shed_inv = [{c: 0 for c in CROPS} for _ in range(2)]
        return self._get_obs(player=0)

    def _get_obs(self, player: int) -> Dict[str, Any]:
        return {
            "player": player,
            "day": int(self.current_step // 24) + 1,
            "hour": int(self.current_step % 24),
            "farms": [
                {
                    "money": self.player_money[0],
                    "tiles": self.tiles_status[0],
                    "farmer": self.farmer_pos[0],
                    "hands": self.hands_list[0]
                },
                {
                    "money": self.player_money[1],
                    "tiles": self.tiles_status[1],
                    "farmer": self.farmer_pos[1],
                    "hands": self.hands_list[1]
                }
            ],
            "market": {
                "inventory": self.market_inventory,
                "prices": self.market_prices
            },
            "town": {
                "unlocked_shops": self.unlocked_shops
            },
            "private": {
                "shed": self.shed_inv[player],
                "seeds": self.seeds_inv[player]
            }
        }

    def step(self, actions: List[Dict[str, Any]]) -> Tuple[Tuple[Dict[str, Any], Dict[str, Any]], List[float], bool, Dict[str, Any]]:
        self.current_step += 1
        move_map = {"NORTH": (-1, 0), "SOUTH": (1, 0), "WEST": (0, -1), "EAST": (0, 1)}

        for p_idx in range(2):
            act = actions[p_idx]
            f_act = act.get("farmer", ["PASS"])
            verb = f_act[0] if f_act else "PASS"
            curr_pos = self.farmer_pos[p_idx]

            if verb in move_map:
                dy, dx = move_map[verb]
                curr_pos[0] = max(0, min(9, curr_pos[0] + dy))
                curr_pos[1] = max(0, min(9, curr_pos[1] + dx))
            elif verb == "DIG":
                y, x = curr_pos
                self.tiles_status[p_idx][y][x]["kind"] = None
            elif verb == "PLANT" and len(f_act) > 1:
                crop_name = f_act[1]
                y, x = curr_pos
                if self.seeds_inv[p_idx].get(crop_name, 0) > 0:
                    self.tiles_status[p_idx][y][x] = {
                        "kind": "PLANT", "crop": crop_name,
                        "watered_today": True, "yield_units": 0.0,
                    }
                    self.seeds_inv[p_idx][crop_name] -= 1
            elif verb == "WATER":
                y, x = curr_pos
                tile = self.tiles_status[p_idx][y][x]
                if tile.get("kind") == "PLANT":
                    tile["watered_today"] = True
            elif verb == "HARVEST":
                y, x = curr_pos
                tile = self.tiles_status[p_idx][y][x]
                if tile.get("kind") == "PLANT" and tile.get("yield_units", 0) >= 1.0:
                    crop = tile["crop"]
                    self.shed_inv[p_idx][crop] = self.shed_inv[p_idx].get(crop, 0) + 1
                    self.tiles_status[p_idx][y][x] = {
                        "kind": None, "crop": None, "watered_today": False, "yield_units": 0.0,
                    }

            for order in act.get("market", []) or []:
                if not order:
                    continue
                order_verb = order[0]
                if order_verb == "BUY_SEED" and len(order) >= 3:
                    crop_name, qty = order[1], order[2]
                    cost = self.market_prices.get(crop_name, 10.0) * qty
                    if self.player_money[p_idx] >= cost:
                        self.player_money[p_idx] -= cost
                        self.seeds_inv[p_idx][crop_name] = self.seeds_inv[p_idx].get(crop_name, 0) + qty
                elif order_verb == "SELL" and len(order) >= 3:
                    crop_name, qty = order[1], order[2]
                    sell_qty = min(qty, self.shed_inv[p_idx].get(crop_name, 0))
                    if sell_qty > 0:
                        self.player_money[p_idx] += self.market_prices.get(crop_name, 10.0) * sell_qty
                        self.shed_inv[p_idx][crop_name] -= sell_qty

        for p_idx in range(2):
            for y in range(10):
                for x in range(10):
                    tile = self.tiles_status[p_idx][y][x]
                    if tile.get("kind") == "PLANT":
                        if tile.get("watered_today"):
                            tile["yield_units"] = min(1.0, tile.get("yield_units", 0.0) + 0.1)
                        tile["watered_today"] = False

        rewards = []
        for p in range(2):
            delta = (self.player_money[p] - self._prev_money[p]) / 100.0
            rewards.append(delta)
            self._prev_money[p] = self.player_money[p]

        done = self.current_step >= self.max_steps
        return (self._get_obs(0), self._get_obs(1)), rewards, done, {}


# =============================================================================
# 4. SELF-PLAY TRAINER COORDINATOR
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
                    n_eval_episodes: int = 5,
                    resume: Optional[str] = None,
                    bootstrap_episodes: Optional[int] = 0,
                    bootstrap_transitions: Optional[int] = 50_000,
                    data_dir: str = "./data/kaggle_episodes",
                    download_bootstrap: bool = False,
                    bc_epochs: int = 0,
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
            for key in ("seed", "use_kaggle_env", "max_episode_steps", "learning_start_episodes"):
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
                    verbose=verbose,
                )
                bootstrap_count = int(stream_stats.get("total_transitions_loaded", 0))
                bc_loss_history = list(stream_stats.get("epoch_losses", []))
                if stream_stats.get("new_days"):
                    config["bootstrap_days_this_run"] = stream_stats["new_days"]
                    config["bootstrapped_dates"] = stream_stats.get("bootstrapped_dates", [])
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

    # Exploration parameter decay
    eps_start = 1.0
    eps_end = 0.05
    eps_decay_steps = max(1, total_episodes - learning_start_episodes)

    # Initialize competitive self-play environment (Kaggle sim or offline mock).
    env = create_competitive_env(
        use_kaggle=use_kaggle_env,
        max_steps=max_episode_steps if use_kaggle_env else min(50, max_episode_steps),
        seed=seed + start_episode,
    )
    if verbose:
        env_name = type(env).__name__
        logger.debug(
            "Self-play env: %s max_steps=%d use_kaggle=%s",
            env_name,
            max_episode_steps if use_kaggle_env else min(50, max_episode_steps),
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

        # Performance Monitoring
        avg_loss = np.mean(loss_history) if loss_history else 0.0
        logger.info(
            "Episode %02d/%02d | Epsilon: %.3f | Buffer Size: %d | Raw Reward: %.2f | "
            "Shaped Reward: %.2f | Avg Loss: %.5f",
            ep, total_episodes, eps, len(buffer), ep_raw_reward, ep_shaped_reward, avg_loss,
        )
        episode_metrics.append({
            "episode": ep,
            "epsilon": eps,
            "buffer_size": len(buffer),
            "raw_reward": ep_raw_reward,
            "shaped_reward": ep_shaped_reward,
            "avg_loss": avg_loss,
        })

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

    if n_eval_episodes > 0:
        try:
            eval_stats = evaluate_win_rate(
                _path_b_policy, random_baseline_policy,
                n_episodes=n_eval_episodes, max_steps=max_episode_steps, base_seed=seed + 1000,
            )
            save_eval_report(eval_stats, dirs["metrics"] / "win_rate_eval.json")
            logger.info(
                "Win-rate eval vs random baseline: %.2f (%d/%d)",
                eval_stats["win_rate"], eval_stats["wins"], eval_stats["n_episodes"],
            )
        except Exception as exc:
            logger.warning("Win-rate eval skipped: %s", exc)

    if ladder_eval_episodes > 0:
        opp_root = resolve_opponents_dir(opponents_dir)
        if opp_root is None:
            logger.warning(
                "ladder_eval_episodes=%d but opponents/ directory not found (opponents_dir=%r)",
                ladder_eval_episodes,
                opponents_dir,
            )
        else:
            try:
                ladder_report = evaluate_ladder(
                    _path_b_policy,
                    opponents_dir=str(opp_root),
                    n_episodes=ladder_eval_episodes,
                    max_steps=max_episode_steps,
                    base_seed=seed + 2000,
                    win_rate_target=ladder_win_rate_target,
                )
                ladder_path = dirs["metrics"] / "ladder_eval.json"
                with open(ladder_path, "w", encoding="utf-8") as fh:
                    json.dump(ladder_report, fh, indent=2)
                logger.info(
                    "Ladder eval (%d opponents, %d ep each, target %.0f%%): beats_all=%s → %s",
                    len(ladder_report.get("results", {})),
                    ladder_eval_episodes,
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
                logger.warning("Ladder eval skipped: %s", exc)

    agent_path = dirs["root"] / "agent.py"
    _export_path_b_agent(agent_path, dirs["root"], code_src=code_src)
    logger.info("Agent export saved to: %s", agent_path)

    metrics_path = dirs["metrics"] / "episode_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump({"episodes": episode_metrics}, fh, indent=2)
    logger.info("Episode metrics saved to: %s", metrics_path)
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
    parser.add_argument("--n-eval-episodes", type=int, default=5)
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
        default=0,
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
