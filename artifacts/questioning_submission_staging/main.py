"""Kaggle submission entrypoint — QuestioningAgent (Cursor / Agent2).

Experiment 3 sealed protocol: Fib hours, cumulative farm-delta scoring provenance
in experiment3/. Deterministic farming policy with memory-fidelity gating.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kaggriculture_adapter import parse_observation
from questioning_agent import QuestioningAgent

# Best collaborative memory from Exp10 was 10 slots under farm-delta.
_AGENT = QuestioningAgent(memory_slots=10)
_PASS = {"farmer": ["PASS"], "hands": [], "market": []}


def agent(observation: Any, configuration: Optional[Dict[str, Any]] = None):
    """Kaggriculture competition entrypoint."""
    try:
        if isinstance(observation, dict):
            raw = observation
        else:
            raw = {"observation": getattr(observation, "__dict__", {}), "id": 0}
        player = 0
        if isinstance(raw, dict):
            if "player" in raw:
                player = int(raw.get("player") or 0)
            elif "id" in raw:
                pid = raw.get("id")
                if isinstance(pid, str) and pid.startswith("p"):
                    player = int(pid[1:] or 0)
                else:
                    try:
                        player = int(pid)
                    except (TypeError, ValueError):
                        player = 0
        parsed = parse_observation(raw, player_id=player)
        action = _AGENT.act(parsed)
        if not isinstance(action, dict):
            return dict(_PASS)
        farmer = action.get("farmer") or ["PASS"]
        hands = action.get("hands") or []
        market = action.get("market") or []
        return {"farmer": list(farmer), "hands": list(hands), "market": list(market)}
    except Exception:
        return dict(_PASS)


# Some Kaggle loaders look for this alias.
def act(observation, configuration=None):
    return agent(observation, configuration)
