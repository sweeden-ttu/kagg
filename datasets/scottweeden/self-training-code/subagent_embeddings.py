"""Subagent registry: exactly 10 letter+number names + static 11th antigravity.

Directory ``subagents/`` is hard-capped at 10 entries. Names must contain both
letters and digits (no special characters). The 11th agent ``antigravity`` lives
outside that directory as a static deterministic anchor.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ── Hard caps ───────────────────────────────────────────────────────────────

MAX_DIRECTORY_SUBAGENTS = 10
EMBEDDING_DIM = 64
NAME_PATTERN = re.compile(r"^(?=.*[A-Za-z])(?=.*[0-9])[A-Za-z0-9]+$")

# Exactly 10 directory subagents (letters + numbers only).
DIRECTORY_SUBAGENT_IDS: Tuple[str, ...] = (
    "detslot01",
    "detslot02",
    "detslot03",
    "detslot04",
    "detslot05",
    "detslot06",
    "detslot07",
    "detslot08",
    "detslot09",
    "detslot10",
)

# 11th static — outside the capped directory.
STATIC_SUBAGENT_ID = "antigravity"

assert len(DIRECTORY_SUBAGENT_IDS) == MAX_DIRECTORY_SUBAGENTS
assert all(NAME_PATTERN.match(n) for n in DIRECTORY_SUBAGENT_IDS)
assert STATIC_SUBAGENT_ID.isalpha()  # static name as specified (letters only)

_CODE_ROOT = Path(__file__).resolve().parent
SUBAGENTS_DIR = _CODE_ROOT / "subagents"
# Repo root: .../kagg/datasets/scottweeden/self-training-code → parents[3] == kagg
_KAGG_ROOT = _CODE_ROOT.parents[3] if len(_CODE_ROOT.parents) >= 4 else _CODE_ROOT
if (_CODE_ROOT.parents[2] / ".antigravity").exists():
    _KAGG_ROOT = _CODE_ROOT.parents[2]
elif (_CODE_ROOT.parents[3] / ".antigravity").exists() or (_CODE_ROOT.parents[3] / "kaggriculture.py").exists():
    _KAGG_ROOT = _CODE_ROOT.parents[3]
ANTIGRAVITY_DIR = _KAGG_ROOT / ".antigravity" / "antigravity"


def validate_directory_name(name: str) -> None:
    if not NAME_PATTERN.match(name):
        raise ValueError(
            f"Subagent name {name!r} must contain letters and numbers only "
            f"(no special characters): {NAME_PATTERN.pattern}"
        )


def deterministic_embedding(agent_id: str, role: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Hash-seeded unit vector embedding (reproducible, no network)."""
    seed_material = f"{agent_id}|{role}|kaggriculture-subagent-v1".encode("utf-8")
    digest = hashlib.sha256(seed_material).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**32 - 1)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def _role_for_directory_slot(idx: int, agent_id: str) -> str:
    roles = [
        "deterministic_rules",
        "crop_first_yield",
        "hire_fib_cost",
        "market_mask",
        "water_feed_care",
        "land_unlock",
        "shed_inventory",
        "flop_budget_42",
        "day29_judgment",
        "slot10_fellowship_anchor",
    ]
    return roles[idx]


def build_manifest(agent_id: str, role: str, *, static: bool, slot: int | None) -> Dict[str, Any]:
    return {
        "id": agent_id,
        "role": role,
        "static": static,
        "slot": slot,
        "embedding_dim": EMBEDDING_DIM,
        "name_has_letters_and_numbers": bool(NAME_PATTERN.match(agent_id)),
        "directory_capped_at": MAX_DIRECTORY_SUBAGENTS if not static else None,
    }


def ensure_subagent_tree(root: Path | None = None) -> Dict[str, Any]:
    """Create 10 directory subagents + static antigravity; write embeddings."""
    root = root or SUBAGENTS_DIR
    if root.exists():
        existing = [p for p in root.iterdir() if p.is_dir()]
        # Allow rebuild: clear only known detslot* dirs, not unrelated files.
        for p in existing:
            if p.name.startswith("detslot") or p.name in DIRECTORY_SUBAGENT_IDS:
                for child in p.iterdir():
                    child.unlink()
                p.rmdir()

    root.mkdir(parents=True, exist_ok=True)

    registry: List[Dict[str, Any]] = []
    embeddings: Dict[str, np.ndarray] = {}

    for i, agent_id in enumerate(DIRECTORY_SUBAGENT_IDS):
        validate_directory_name(agent_id)
        role = _role_for_directory_slot(i, agent_id)
        slot = i + 1  # 1..10
        agent_dir = root / agent_id
        agent_dir.mkdir(parents=True, exist_ok=True)
        emb = deterministic_embedding(agent_id, role)
        embeddings[agent_id] = emb
        np.save(agent_dir / "embedding.npy", emb)
        manifest = build_manifest(agent_id, role, static=False, slot=slot)
        with open(agent_dir / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        registry.append(manifest)

    if len(registry) > MAX_DIRECTORY_SUBAGENTS:
        raise RuntimeError(f"Directory subagent count exceeds {MAX_DIRECTORY_SUBAGENTS}")

    # Registry lists only the 10 directory agents.
    with open(root / "registry.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "max_directory_subagents": MAX_DIRECTORY_SUBAGENTS,
                "count": len(registry),
                "name_rule": "letters_and_numbers_only",
                "name_pattern": NAME_PATTERN.pattern,
                "static_eleventh": STATIC_SUBAGENT_ID,
                "static_path": str(ANTIGRAVITY_DIR),
                "subagents": registry,
            },
            fh,
            indent=2,
        )

    # 11th static antigravity (outside capped directory).
    ANTIGRAVITY_DIR.mkdir(parents=True, exist_ok=True)
    ag_role = "static_deterministic_correctness_and_completion"
    ag_emb = deterministic_embedding(STATIC_SUBAGENT_ID, ag_role)
    embeddings[STATIC_SUBAGENT_ID] = ag_emb
    np.save(ANTIGRAVITY_DIR / "embedding.npy", ag_emb)
    ag_manifest = build_manifest(STATIC_SUBAGENT_ID, ag_role, static=True, slot=11)
    ag_manifest["outside_directory_cap"] = True
    ag_manifest["note"] = (
        "Static 11th subagent; not counted in subagents/ directory limit of 10."
    )
    with open(ANTIGRAVITY_DIR / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(ag_manifest, fh, indent=2)

    # Combined embedding matrix for experimental loaders.
    order = list(DIRECTORY_SUBAGENT_IDS) + [STATIC_SUBAGENT_ID]
    matrix = np.stack([embeddings[i] for i in order], axis=0)
    np.savez(
        root / "embeddings_all.npz",
        ids=np.array(order, dtype=object),
        vectors=matrix,
        dim=np.array([EMBEDDING_DIM]),
        directory_count=np.array([MAX_DIRECTORY_SUBAGENTS]),
        static_id=np.array([STATIC_SUBAGENT_ID], dtype=object),
    )
    # Also store combined under .antigravity for handoff.
    handoff = ANTIGRAVITY_DIR.parent / "subagent_embeddings.npz"
    np.savez(
        handoff,
        ids=np.array(order, dtype=object),
        vectors=matrix,
        dim=np.array([EMBEDDING_DIM]),
    )

    return {
        "directory": str(root),
        "count": len(registry),
        "ids": list(DIRECTORY_SUBAGENT_IDS),
        "static": STATIC_SUBAGENT_ID,
        "static_dir": str(ANTIGRAVITY_DIR),
        "embedding_dim": EMBEDDING_DIM,
        "npz": [str(root / "embeddings_all.npz"), str(handoff)],
    }


def load_directory_embeddings(root: Path | None = None) -> Dict[str, np.ndarray]:
    root = root or SUBAGENTS_DIR
    out: Dict[str, np.ndarray] = {}
    for agent_id in DIRECTORY_SUBAGENT_IDS:
        path = root / agent_id / "embedding.npy"
        if path.exists():
            out[agent_id] = np.load(path)
    return out


def load_antigravity_embedding() -> np.ndarray:
    path = ANTIGRAVITY_DIR / "embedding.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


if __name__ == "__main__":
    summary = ensure_subagent_tree()
    print(json.dumps(summary, indent=2))
