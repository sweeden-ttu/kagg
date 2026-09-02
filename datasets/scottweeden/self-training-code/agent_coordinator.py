"""Self-play coordination and experiment directory setup.

Extracted from the monolithic kaggriculture_self_play_training.py.
Provides setup_experiment_dirs() and SelfPlayCoordinator.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

_SCRIPT_DIR = Path(__file__).resolve().parent
for _extra in (_SCRIPT_DIR, _SCRIPT_DIR / "artifacts", _SCRIPT_DIR / "scratch"):
    if _extra.exists() and str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

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


from eval_policy import (
    discover_reference_opponent_files,
    load_kaggle_agent_policy,
    resolve_opponents_dir,
)


class SelfPlayCoordinator:
    """
    Coordinates self-play training by managing an opponent pool of historical checkpoints
    and enforcing a progressive tier-by-tier curriculum against benchmark opponents.
    """
    def __init__(self,
                 latent_dim: int = 512,
                 shared_dim: int = 256,
                 checkpoint_dir: Optional[str] = None,
                 opponents_dir: Optional[str] = None):
        self.latent_dim = latent_dim
        self.shared_dim = shared_dim
        if checkpoint_dir is None:
            checkpoint_dir = str(_SCRIPT_DIR / "experiments" / "self_play" / "checkpoints")
        self.checkpoint_dir = str(checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.opponent_pool = []

        # Discover reference ladder opponents ordered strictly by tier (Tier 0 -> Tier 9)
        self.opponents_dir = opponents_dir or (str(resolve_opponents_dir()) if resolve_opponents_dir() else None)
        self.ladder_opponents: List[Tuple[str, int, str]] = []
        if self.opponents_dir and os.path.isdir(self.opponents_dir):
            entries = discover_reference_opponent_files(self.opponents_dir)
            self.ladder_opponents = [(slug, tier, str(p)) for slug, tier, p in entries]
            if self.ladder_opponents:
                tiers_str = ", ".join(f"T{t}:{s}" for s, t, _ in self.ladder_opponents)
                print(f"[Curriculum] Discovered {len(self.ladder_opponents)} tiered reference opponents: [{tiers_str}]")

        # Progressive Tier-by-Tier Curriculum State
        self.current_tier_idx: int = 0
        self.tier_history: Dict[str, List[Dict[str, Any]]] = {
            s: [] for s, _, _ in self.ladder_opponents
        }
        self.tier_clear_wins_target: int = 2  # Requires 2 wins to promote

        # Round-robin cursor for deterministic historical / online / reference selection.
        self._opponent_select_i = 0
        self._cached_opp_net: Optional[HierarchicalDQNBranching] = None
        self._cached_opp_device: Optional[torch.device] = None

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

    def record_match_result(
        self, opponent_path: str, won: bool, p0_money: float, p1_money: float
    ) -> bool:
        """Record match result against an opponent and evaluate tier promotion."""
        if not self.ladder_opponents:
            return False

        opp_p = Path(opponent_path).resolve()
        # Find matching ladder entry
        matched_idx = None
        for idx, (slug, tier, path_str) in enumerate(self.ladder_opponents):
            if Path(path_str).resolve() == opp_p:
                matched_idx = idx
                break

        if matched_idx is None:
            return False

        slug, tier, _ = self.ladder_opponents[matched_idx]
        if slug not in self.tier_history:
            self.tier_history[slug] = []
        self.tier_history[slug].append({
            "won": won,
            "p0_money": p0_money,
            "p1_money": p1_money,
        })

        # Only evaluate promotion if playing the active tier boss
        if matched_idx == self.current_tier_idx:
            history = self.tier_history[slug]
            wins = sum(1 for h in history if h["won"])
            total = len(history)
            win_rate = wins / total if total > 0 else 0.0

            # Promote if at least target wins and win rate >= 60%, or last 2 consecutive wins
            consecutive_wins = len(history) >= 2 and history[-1]["won"] and history[-2]["won"]
            if (wins >= self.tier_clear_wins_target and win_rate >= 0.60) or consecutive_wins:
                if self.current_tier_idx + 1 < len(self.ladder_opponents):
                    self.current_tier_idx += 1
                    next_slug, next_tier, _ = self.ladder_opponents[self.current_tier_idx]
                    print(
                        f"\n{'='*70}\n"
                        f"🏆 [CURRICULUM PROMOTION] Defeated Tier {tier} ({slug})! "
                        f"(Record: {wins}W-{total-wins}L, {win_rate:.0%})\n"
                        f"🚀 Unlocking active training target: Tier {next_tier} ({next_slug})\n"
                        f"{'='*70}\n"
                    )
                    return True
        return False

    def select_opponent(self) -> Optional[str]:
        """Pick the next opponent following the progressive tier curriculum.

        Curriculum:
        - When reference opponents exist:
          - Every 4th or 5th step, fight the active tier boss (or occasionally review a prior cleared tier).
          - Other steps: online self-play (None) or historical checkpoint (.pt).
        - If checkpoint pool is empty, exclusively trains against the active tier boss!
        """
        step = self._opponent_select_i
        self._opponent_select_i = step + 1

        if self.ladder_opponents:
            active_slug, active_tier, active_path = self.ladder_opponents[self.current_tier_idx]

            # If no historical checkpoints exist, focus on active tier boss
            if not self.opponent_pool:
                # 80% active tier, 20% prior cleared tier (if any)
                if self.current_tier_idx > 0 and step % 5 == 0:
                    prior_idx = (step // 5) % self.current_tier_idx
                    return self.ladder_opponents[prior_idx][2]
                return active_path

            # With historical checkpoints: every 4th step plays the ladder boss
            if step % 4 == 0:
                if self.current_tier_idx > 0 and (step // 4) % 4 == 0:
                    prior_idx = (step // 16) % self.current_tier_idx
                    return self.ladder_opponents[prior_idx][2]
                return active_path

        if not self.opponent_pool:
            return None

        # Every 5th non-boss step is online self-play (None)
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

        If ``checkpoint_path`` is a Python script (ends in .py), loads the opponent directly.
        ``checkpoint_path`` may be ``None`` to clone the current online weights.
        """
        if checkpoint_path is not None and checkpoint_path.endswith(".py") and os.path.exists(checkpoint_path):
            return load_kaggle_agent_policy(checkpoint_path)

        parser = KaggricultureJSONParser()

        # Lazily instantiate or reuse opponent network
        if self._cached_opp_net is None or self._cached_opp_device != device:
            opp_extractor = KaggricultureFeatureExtractor(latent_dim=self.latent_dim)
            self._cached_opp_net = HierarchicalDQNBranching(
                opp_extractor, latent_dim=self.latent_dim, shared_dim=self.shared_dim
            ).to(device)
            self._cached_opp_device = device

        opp_net = self._cached_opp_net

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            try:
                opp_net.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
            except TypeError:
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
