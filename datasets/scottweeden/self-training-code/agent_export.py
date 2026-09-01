"""Agent export for Kaggle submission.

Extracted from the monolithic kaggriculture_self_play_training.py.
Provides _export_path_b_agent().
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional


def _export_path_b_agent(
    agent_path: Path,
    experiment_root: Path,
    *,
    code_src: Optional[str] = None,
) -> None:
    """Write a minimal Kaggle submission agent using shared adapter decode."""
    from _resolve_code_src import _resolve_code_src

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
from kaggriculture_adapter import decode_path_b_action, parse_observation, select_hand_farm_verbs
from kaggriculture_path_b_rebuild import (
    KaggricultureJSONParser,
    KaggricultureFeatureExtractor,
    HierarchicalDQNBranching,
    HierarchicalActionMasker,
    apply_hierarchical_masks,
    break_pass_spawn_deadlock,
    prefer_farm_invest_actions,
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
            masked_q["farmer_verb"] = break_pass_spawn_deadlock(
                masked_q["farmer_verb"], masks["farmer_verb"]
            )
            farm_verb, farm_market = prefer_farm_invest_actions(
                masked_q["farmer_verb"],
                masks["farmer_verb"],
                masked_q["market"],
                masks.get("market"),
                observation=agent_obs,
            )
            masked_q["farmer_verb"] = farm_verb
            if farm_market is not None:
                masked_q["market"] = farm_market
            verb_idx = int(masked_q["farmer_verb"].argmax(dim=-1).item())
            crop_idx = int(masked_q["crop_parameter"].argmax(dim=-1).item())
            hands = select_hand_farm_verbs(agent_obs)
            market_seq = masked_q["market"].argmax(dim=-1).squeeze(0)
            market = [int(market_seq[t].item()) for t in range(self.net.max_market_orders)]
        return decode_path_b_action(verb_idx, crop_idx, hands, market, agent_obs)


_AGENT = Agent()


def agent(obs, cfg=None):
    return _AGENT.act(obs)
'''
    agent_path.write_text(agent_code, encoding="utf-8")
