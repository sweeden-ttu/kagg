"""Agent export for Kaggle submission.

Extracted from the monolithic kaggriculture_self_play_training.py.
Provides _export_path_b_agent() and submission size gating (90 MB / 100 MB).
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import List, Optional

from hard_limits import (
    DEFAULT_LIMITS,
    build_submission_archive,
    check_model_size,
    limits_manifest,
)

logger = logging.getLogger(__name__)


def _export_path_b_agent(
    agent_path: Path,
    experiment_root: Path,
    *,
    code_src: Optional[str] = None,
    training_hours: Optional[float] = None,
) -> None:
    """Write a minimal Kaggle submission agent using shared adapter decode.

    After writing, validates model ≤ 100 MB and packs a submission archive
    that must be ≤ 90 MB (10 MB patch buffer under the 100 MB ceiling).
    Training data size is not checked. If ``training_hours`` ≥ 24, the
    post-training size gate requires the model alone to fit the 90 MB budget.
    """
    from _resolve_code_src import _resolve_code_src

    src_root = _resolve_code_src(code_src)
    for module_name in (
        "kaggriculture_adapter.py",
        "kaggriculture_path_b_rebuild.py",
        "hard_limits.py",
    ):
        src = src_root / module_name
        dst = experiment_root / module_name
        if not src.exists():
            if module_name == "hard_limits.py":
                continue
            raise FileNotFoundError(f"Missing adapter module in code dataset: {src}")
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

    agent_code = '''"""Kaggle Kaggriculture Path B agent export."""
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
                masked_q["farmer_verb"], masks["farmer_verb"], observation=agent_obs
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
            hands = [int(masked_q["hands"][i].argmax(dim=-1).item()) for i in range(self.net.num_hands)]
            market_seq = masked_q["market"].argmax(dim=-1).squeeze(0)
            market = [int(market_seq[t].item()) for t in range(self.net.max_market_orders)]
        return decode_path_b_action(verb_idx, crop_idx, hands, market, agent_obs)


_AGENT = Agent()


def agent(obs, cfg=None):
    return _AGENT.act(obs)
'''
    agent_path.write_text(agent_code, encoding="utf-8")

    model_path = experiment_root / "models" / "model.pth"
    size_report = check_model_size(
        model_path,
        limits=DEFAULT_LIMITS,
        after_training_hours=training_hours,
    )
    sources: List[Path] = [
        agent_path,
        experiment_root / "kaggriculture_adapter.py",
        experiment_root / "kaggriculture_path_b_rebuild.py",
    ]
    hl = experiment_root / "hard_limits.py"
    if hl.exists():
        sources.append(hl)
    if model_path.exists():
        sources.append(model_path)

    archive = experiment_root / "submission.tar.gz"
    pack_report = build_submission_archive(sources, archive, limits=DEFAULT_LIMITS)

    gate = {
        "limits": limits_manifest(),
        "model_size": size_report,
        "submission_archive": pack_report,
        "training_hours": training_hours,
        "training_data_size_capped": False,
    }
    metrics_dir = experiment_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "submission_limits.json", "w", encoding="utf-8") as fh:
        json.dump(gate, fh, indent=2)
    logger.info(
        "Submission gate: model=%.2fMB within_100=%s archive=%.2fMB passed_90=%s",
        size_report.get("mb", 0),
        size_report.get("within_100mb"),
        pack_report.get("mb", 0),
        pack_report.get("passed_90mb"),
    )
    if not pack_report.get("passed_90mb", False):
        raise RuntimeError(
            f"Submission archive {pack_report.get('mb')} MB exceeds 90 MB hard limit "
            f"(100 MB ceiling with 10 MB patch buffer)"
        )
