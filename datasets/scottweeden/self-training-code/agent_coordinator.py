"""Self-play coordination and experiment directory setup.

Extracted from the monolithic kaggriculture_self_play_training.py.
Provides setup_experiment_dirs() and SelfPlayCoordinator.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

_SCRIPT_DIR = Path(__file__).resolve().parent
for _extra in (_SCRIPT_DIR, _SCRIPT_DIR / "artifacts", _SCRIPT_DIR / "scratch"):
    if _extra.exists() and str(_extra) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(_extra))

from kaggriculture_path_b_rebuild import (
    KaggricultureJSONParser,
    KaggricultureFeatureExtractor,
    HierarchicalDQNBranching,
    HierarchicalActionMasker,
    apply_hierarchical_masks,
)
from kaggriculture_adapter import decode_path_b_action


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
        # Round-robin cursor for deterministic historical / online selection.
        self._opponent_select_i = 0

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

    def select_opponent(self) -> Optional[str]:
        """Pick the next opponent checkpoint deterministically.

        Schedule (repeats): four historical round-robin picks, then one online
        (``None`` = current weights). Empty pool always returns ``None``.
        """
        if not self.opponent_pool:
            return None
        step = self._opponent_select_i
        self._opponent_select_i = step + 1
        # Fixed 4:1 historical:online (replaces former 80%/20% chance).
        if step % 5 == 4:
            return None
        hist_step = step - (step // 5)
        return self.opponent_pool[hist_step % len(self.opponent_pool)]

    def get_agent_policy_fn(self,
                            checkpoint_path: Optional[str],
                            online_net: HierarchicalDQNBranching,
                            device: torch.device):
        """
        Generates an agent execution policy function mapping observation to action.

        ``checkpoint_path`` may be ``None`` to clone the current online weights.
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
