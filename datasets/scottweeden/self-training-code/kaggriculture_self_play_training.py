"""Kaggriculture self-play training — public API.

All implementation has been extracted into separate SRP modules.
This module re-exports the public entry points for backward compatibility.
"""

from train_orchestrator import train_self_play
from environment import KaggleCompetitiveEnv, create_competitive_env
from replay_buffer import PrioritizedReplayBuffer
from agent_coordinator import SelfPlayCoordinator
from agent_export import _export_path_b_agent
from _resolve_code_src import _resolve_code_src

__all__ = [
    "train_self_play",
    "KaggleCompetitiveEnv",
    "create_competitive_env",
    "PrioritizedReplayBuffer",
    "SelfPlayCoordinator",
    "_export_path_b_agent",
    "_resolve_code_src",
]

# Re-export CLI entry point for `if __name__ == "__main__"` usage
from cli import main

if __name__ == "__main__":
    main()
