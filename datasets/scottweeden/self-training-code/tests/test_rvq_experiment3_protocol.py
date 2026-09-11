"""Unit tests for Experiment 3 Reasoning vs Questioning protocol remediation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from memory_protocol import (
    AGENT1_HOURS,
    AGENT2_HOURS,
    CHARITY_DONATION,
    COLLABORATIVE_STARTING_MONEY,
    COMPETITIVE_STARTING_MONEY,
    DeterminedFact,
    MemoryBank,
    SEAT_TRADE_PRESERVE_KEYS,
)
from agents.reasoning_agent import ReasoningAgent
from agents.questioning_agent import QuestioningAgent


def test_agent_hour_sets_match_experiment_3():
    assert AGENT1_HOURS == frozenset({0, 2, 3, 5, 7, 11, 13, 17, 19, 23})
    assert AGENT2_HOURS == frozenset({1, 4, 5, 13, 21})
    assert 34 not in AGENT2_HOURS
    assert ReasoningAgent.schedule_hours == set(AGENT1_HOURS)
    assert QuestioningAgent.schedule_hours == set(AGENT2_HOURS)


def test_starting_money_constants():
    assert COMPETITIVE_STARTING_MONEY == 1500
    assert COLLABORATIVE_STARTING_MONEY == 3000
    assert CHARITY_DONATION == 888


def test_memory_clear_preserves_charity_keys():
    bank = MemoryBank(n_slots=10, mode="deterministic")
    bank.slots[0].content = DeterminedFact(
        text="charity", key="opening_charity", value=888
    )
    bank.slots[1].content = DeterminedFact(
        text="wheat cost", key="seed_cost", value=10
    )
    bank.history.append({"slot": 1, "kind": "determined", "q": "seed", "a": "10"})
    cleared = bank.clear_slots(preserve_keys=SEAT_TRADE_PRESERVE_KEYS)
    assert bank.slots[0].content is not None
    assert getattr(bank.slots[0].content, "key", None) == "opening_charity"
    assert bank.slots[1].content is None
    assert any(p["key"] == "opening_charity" for p in cleared["preserved_slots"])


def test_day3_opposite_sides():
    r = ReasoningAgent(memory_slots=10)
    q = QuestioningAgent(memory_slots=10)
    r.charity_record = {"ok": True, "amount": 888}
    r.private_fellowship_policy = "fellowship"
    q._charity_recorded = True
    q.charity_observation = {"agent1_charitable_nature": True}
    obs = {"day": 3, "hour": 5, "player": 0, "farms": [{"money": 612}, {"money": 2388}]}
    r_rec = r.choose_side(obs, my_seat=0, questioning_seat=1)
    q_rec = q.take_opposite_side(r_rec["side"], obs)
    assert r_rec["side"] == 0
    assert q_rec["side"] == 1
    assert q_rec["side"] == 1 - r_rec["side"]


def test_episode_trace_buffer_clear():
    import importlib.util

    suite_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "reasoning_vs_questioning"
        / "run_suite.py"
    )
    spec = importlib.util.spec_from_file_location("rvq_run_suite", suite_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    buf = mod.EpisodeTraceBuffer()
    buf.add({"step": 0})
    buf.add({"step": 1})
    assert buf.clear() == 2
    assert len(buf.transitions) == 0


def test_memory_fidelity_scales_with_slots():
    from memory_protocol import memory_fidelity

    low = memory_fidelity(10, 2)
    high = memory_fidelity(30, 30)
    assert 0.3 <= low < high <= 1.0


@pytest.mark.integration
def test_smoke_competitive_and_collaborative_short():
    pytest.importorskip("kaggle_environments")
    import importlib.util

    suite_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "reasoning_vs_questioning"
        / "run_suite.py"
    )
    spec = importlib.util.spec_from_file_location("rvq_run_suite_smoke", suite_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    run_match = mod.run_match

    # 96 steps ≈ 4 days → one seat trade + day2/day3 hooks.
    m_comp = run_match(
        42,
        reasoning_seat=0,
        max_steps=96,
        challenge_mode="competitive",
    )
    assert m_comp.starting_money == 1500
    assert m_comp.starting_money_ok
    assert m_comp.opening_charity and m_comp.opening_charity.get("ok")
    assert m_comp.seat_trades >= 1
    assert m_comp.alliance is not None
    assert m_comp.alliance["questioning_side"] == 1 - m_comp.alliance["reasoning_side"]
    assert m_comp.schedule_ok
    assert m_comp.scoring_mode == "cumulative_farm_delta"
    assert m_comp.questioning_opening_transfer == 888
    assert m_comp.reasoning_opening_transfer == -888
    # Gift must not decide the winner via terminal seat inheritance alone.
    assert m_comp.winner in ("reasoning", "questioning", "tie")
    # Scores are farm deltas, not terminal banks.
    assert isinstance(m_comp.reasoning_score, float)
    assert isinstance(m_comp.questioning_score, float)

    m_collab = run_match(
        43,
        reasoning_seat=0,
        max_steps=96,
        challenge_mode="collaborative",
    )
    assert m_collab.starting_money == 3000
    assert m_collab.starting_money_ok
    assert m_collab.day2_refund is not None
    assert m_collab.seat_trades >= 1
    assert m_collab.alliance is not None
    assert m_collab.winner == "team"
    assert m_collab.scoring_mode == "cumulative_farm_delta"


@pytest.mark.integration
def test_memory_slots_change_collaborative_farm_delta():
    """Memory fidelity must rise with slot count (Exp10 operational constraint)."""
    pytest.importorskip("kaggle_environments")
    import importlib.util

    suite_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "reasoning_vs_questioning"
        / "run_suite.py"
    )
    spec = importlib.util.spec_from_file_location("rvq_run_suite_mem", suite_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    run_match = mod.run_match

    m10 = run_match(
        7,
        reasoning_seat=0,
        reasoning_kwargs={"memory_slots": 10},
        questioning_kwargs={"memory_slots": 10},
        max_steps=72,
        challenge_mode="collaborative",
    )
    m30 = run_match(
        7,
        reasoning_seat=0,
        reasoning_kwargs={"memory_slots": 30},
        questioning_kwargs={"memory_slots": 30},
        max_steps=72,
        challenge_mode="collaborative",
    )
    assert m10.reasoning_metrics.get("memory_fidelity", 0) < m30.reasoning_metrics.get(
        "memory_fidelity", 1
    )
