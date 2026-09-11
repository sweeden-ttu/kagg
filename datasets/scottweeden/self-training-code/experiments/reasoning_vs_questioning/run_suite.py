"""Experiment 3 suite: Reasoning vs Questioning — 10 ablations.

5 competitive (Exp 1–5) + 5 collaborative (Exp 6–10).

Agent1 (Reasoning) acts on stated prime-hour set:
  {0, 2, 3, 5, 7, 11, 13, 17, 19, 23}
Agent2 (Questioning) acts on Fibonacci hours (34 dropped as invalid):
  {1, 4, 5, 13, 21}

Agents trade seats every 3 days; episode memory banks / EpisodeTraceBuffer
are cleared (identity/charity keys preserved — not RL-regret erasure).

Every matchup runs twice with seats swapped (phase-2).

Competitive: startingMoney=1500 then Agent1 opening gift 888 → 612 / 2388.
Collaborative: startingMoney=3000 then gift 888 → 2112 / 3888, plus day-2
Kaggle-determined refund. Day 3: Reasoning chooses a side; Questioning opposite.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Allow running as script from repo or dataset folder.
_HERE = Path(__file__).resolve().parent
# .../self-training-code/experiments/reasoning_vs_questioning → self-training-code
_CODE_ROOT = _HERE.parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from agents.questioning_agent import QuestioningAgent
from agents.reasoning_agent import ReasoningAgent
from environment import create_collaborative_env, create_competitive_env
from hard_limits import limits_manifest
from memory_protocol import (
    AGENT1_HOURS,
    AGENT2_HOURS,
    CHARITY_DONATION,
    COLLABORATIVE_STARTING_MONEY,
    COMPETITIVE_STARTING_MONEY,
    day_edge_question_stats,
)

logger = logging.getLogger(__name__)

DEFAULT_OUT = _HERE


@dataclass
class EpisodeTraceBuffer:
    """Match-local experience log cleared on mid-episode seat trades.

    Not a DQN PrioritizedReplayBuffer — clearing does not 'remove regret'.
    """

    capacity: int = 10_000
    transitions: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, row: Dict[str, Any]) -> None:
        self.transitions.append(row)
        if len(self.transitions) > self.capacity:
            self.transitions = self.transitions[-self.capacity :]

    def clear(self) -> int:
        n = len(self.transitions)
        self.transitions.clear()
        return n


@dataclass
class MatchResult:
    seed: int
    reasoning_seat: int
    questioning_seat: int
    challenge_mode: str
    starting_money: int
    money_p0: float
    money_p1: float
    winner: str  # "reasoning" | "questioning" | "tie" | "team"
    reasoning_metrics: Dict[str, Any]
    questioning_metrics: Dict[str, Any]
    schedule_ok: bool
    starting_money_ok: bool
    opening_charity: Optional[Dict[str, Any]] = None
    motive_dialogue: Optional[Dict[str, Any]] = None
    seat_trades: int = 0
    day2_refund: Optional[Dict[str, Any]] = None
    alliance: Optional[Dict[str, Any]] = None
    team_money: Optional[float] = None
    # Identity ledgers: farm Δ while controlling a seat; gift travels with Agent2.
    reasoning_farm_delta: float = 0.0
    questioning_farm_delta: float = 0.0
    reasoning_opening_transfer: float = 0.0
    questioning_opening_transfer: float = 0.0
    reasoning_score: float = 0.0
    questioning_score: float = 0.0
    scoring_mode: str = "cumulative_farm_delta"


def _final_money(obs: Dict[str, Any], player: int) -> float:
    farms = obs.get("farms", []) or []
    if len(farms) > player:
        return float(farms[player].get("money", 0.0) or 0.0)
    return 0.0


def _verify_schedule(agent: Any, expected_hours) -> bool:
    for entry in agent.action_audit:
        hour = int(entry.get("hour", -1))
        # Opening DONATE / dialogue is pre-loop (hour=-1); skip schedule check.
        if hour < 0:
            continue
        acted = bool(entry.get("acted"))
        legal = hour in expected_hours
        op = str(entry.get("op", ""))
        # Side-choice / dialogue ops may fire off-schedule from the harness.
        if op in ("CHOOSE_SIDE", "SEAT_TRADE", "DAY2_REFUND"):
            continue
        if acted and not legal and op not in ("PASS",):
            # acted=True with farm ops must be on schedule
            if op not in ("ANSWER_MOTIVE", "PATH_PROOF", "ADOPT_PRIVATE_POLICY", "PATH_TRUST"):
                if hour not in expected_hours:
                    return False
        if acted and hour >= 0 and hour not in expected_hours and op not in (
            "ANSWER_MOTIVE",
            "PATH_PROOF",
            "ADOPT_PRIVATE_POLICY",
            "PATH_TRUST",
            "CHOOSE_SIDE",
            "SEAT_TRADE",
            "DAY2_REFUND",
            "DAY29_QUESTION",
        ):
            return False
    return True


def _starting_money_for_mode(challenge_mode: str) -> int:
    if challenge_mode == "collaborative":
        return COLLABORATIVE_STARTING_MONEY
    return COMPETITIVE_STARTING_MONEY


def run_match(
    seed: int,
    *,
    reasoning_seat: int,
    reasoning_kwargs: Optional[Dict[str, Any]] = None,
    questioning_kwargs: Optional[Dict[str, Any]] = None,
    max_steps: int = 720,
    turns_per_cycle: int = 24,
    challenge_mode: str = "competitive",
) -> MatchResult:
    """Play one full episode with Reasoning on ``reasoning_seat`` (0 or 1).

    Before the step loop, Agent1 (Reasoning) donates 888 to Agent2's bank so
    Agent2 can observe and record Agent1's charitable nature.
    Seats swap every 3 days with memory/trace clears.
    """
    r_kw = dict(reasoning_kwargs or {})
    q_kw = dict(questioning_kwargs or {})
    reasoning = ReasoningAgent(**r_kw)
    questioning = QuestioningAgent(**q_kw)
    reasoning.reset()
    questioning.reset()

    starting_money = _starting_money_for_mode(challenge_mode)
    if challenge_mode == "collaborative":
        env = create_collaborative_env(
            use_kaggle=True,
            max_steps=max_steps,
            seed=seed,
            turns_per_cycle=turns_per_cycle,
            starting_money=starting_money,
        )
    else:
        env = create_competitive_env(
            use_kaggle=True,
            max_steps=max_steps,
            seed=seed,
            turns_per_cycle=turns_per_cycle,
            starting_money=starting_money,
        )

    obs_p0 = env.reset()
    obs_p1 = env._get_obs(player=1)
    trace = EpisodeTraceBuffer()

    start0 = _final_money(obs_p0, 0)
    start1 = _final_money(obs_p1, 1)
    starting_ok = abs(start0 - starting_money) < 1e-6 and abs(start1 - starting_money) < 1e-6

    # Agent1's one opening act: donate 888 → Agent2 bank (publicly visible).
    charity = reasoning.offer_opening_charity(env, reasoning_seat)
    obs_p0 = env._get_obs(player=0)
    obs_p1 = env._get_obs(player=1)
    q_seat = 1 - reasoning_seat
    q_obs = env._get_obs(player=q_seat)
    questioning.record_agent1_charity(q_obs, amount=int(charity.get("amount", CHARITY_DONATION)))
    motive_q = questioning.question_agent1_motives(q_obs)
    motive_dialogue = reasoning.answer_motive_question(
        motive_q["question"],
        obs=env._get_obs(player=reasoning_seat),
    )
    questioning.receive_motive_answer(motive_dialogue)
    path_proof = reasoning.emit_kaggle_path_proof()
    path_trust = questioning.evaluate_agent1_path_trust(path_proof, day=0)
    rules_view = questioning.state_rules_understanding(q_obs)
    private_policy = reasoning.adopt_private_fellowship_policy(agent2_rules_statement=rules_view)

    # Opening 888 gift travels with Agent2's identity ledger (not farm Δ / not land equity).
    gift = float(charity.get("amount", CHARITY_DONATION) or 0.0) if charity.get("ok") else 0.0
    reasoning_opening_transfer = -gift
    questioning_opening_transfer = +gift

    done = False
    steps = 0
    day29_asked = False
    seat_trades = 0
    last_trade_day = -1
    day2_refund: Optional[Dict[str, Any]] = None
    day3_sided = False
    alliance: Optional[Dict[str, Any]] = None
    current_r_seat = int(reasoning_seat)
    reasoning_farm_delta = 0.0
    questioning_farm_delta = 0.0
    prev_seat_money = [_final_money(obs_p0, 0), _final_money(obs_p1, 1)]

    while not done and steps < max_steps:
        agents = [None, None]
        agents[current_r_seat] = reasoning
        agents[1 - current_r_seat] = questioning
        a0 = agents[0].act(obs_p0)
        a1 = agents[1].act(obs_p1)

        r_obs = obs_p0 if current_r_seat == 0 else obs_p1
        day = int(r_obs.get("day", 0) or 0)
        hour = int(r_obs.get("hour", 0) or 0)

        # Collaborative day-2 refund at the start of day 3 (or late day 2).
        if (
            challenge_mode == "collaborative"
            and day2_refund is None
            and (day == 3 or (day == 2 and hour >= 23))
        ):
            before_refund = [_final_money(obs_p0, 0), _final_money(obs_p1, 1)]
            day2_refund = env.apply_day2_kaggle_refund()
            obs_p0 = env._get_obs(player=0)
            obs_p1 = env._get_obs(player=1)
            after_refund = [_final_money(obs_p0, 0), _final_money(obs_p1, 1)]
            # Refund credits controlling identities (not competitive farm Δ).
            for seat in (0, 1):
                credit = after_refund[seat] - before_refund[seat]
                if abs(credit) < 1e-9:
                    continue
                if seat == current_r_seat:
                    # Track on charity record side-channel for metrics; farm Δ excludes refund.
                    pass
                prev_seat_money[seat] = after_refund[seat]
            reasoning.action_audit.append(
                {"day": day, "hour": hour, "acted": True, "op": "DAY2_REFUND"}
            )

        # Day-3 side choice (once).
        if day == 3 and not day3_sided:
            r_obs_now = obs_p0 if current_r_seat == 0 else obs_p1
            q_obs_now = obs_p0 if current_r_seat == 1 else obs_p1
            r_side_rec = reasoning.choose_side(
                r_obs_now,
                my_seat=current_r_seat,
                questioning_seat=1 - current_r_seat,
            )
            q_side_rec = questioning.take_opposite_side(r_side_rec["side"], q_obs_now)
            alliance = {
                "reasoning_side": r_side_rec["side"],
                "questioning_side": q_side_rec["side"],
                "reasoning_choice": r_side_rec,
                "questioning_choice": q_side_rec,
            }
            day3_sided = True

        # Day-29 closing question from Agent1 (losing = lower cumulative farm Δ).
        if not day29_asked:
            if day == 29 and hour >= 20 and reasoning.may_act(hour):
                q29 = reasoning.ask_day29_question(r_obs)
                if q29:
                    day29_asked = True
                    q_obs_now = obs_p0 if (1 - current_r_seat) == 0 else obs_p1
                    agent2_losing = questioning_farm_delta < reasoning_farm_delta
                    reply29 = questioning.answer_day29_question(
                        q29, q_obs_now, tell_truth=True
                    )
                    reasoning.judge_day29_reply(reply29, agent2_losing=agent2_losing)

        step_info: Dict[str, Any] = {}
        if alliance is not None:
            step_info["alliance"] = alliance
        money_before = [prev_seat_money[0], prev_seat_money[1]]
        (obs_p0, obs_p1), rewards, done, info = env.step([a0, a1])
        if isinstance(info, dict) and alliance is not None:
            info = {**info, "alliance": alliance}

        # Attribute seat bank Δ this step to the identity controlling that seat.
        new_seat_money = [_final_money(obs_p0, 0), _final_money(obs_p1, 1)]
        step_deltas = [
            new_seat_money[0] - money_before[0],
            new_seat_money[1] - money_before[1],
        ]
        for seat in (0, 1):
            if seat == current_r_seat:
                reasoning_farm_delta += step_deltas[seat]
            else:
                questioning_farm_delta += step_deltas[seat]
        prev_seat_money = new_seat_money

        trace.add(
            {
                "step": steps,
                "day": day,
                "hour": hour,
                "rewards": rewards,
                "reasoning_seat": current_r_seat,
                "seat_deltas": step_deltas,
                "reasoning_farm_delta": reasoning_farm_delta,
                "questioning_farm_delta": questioning_farm_delta,
            }
        )
        steps += 1

        # Mid-episode seat trade every 3 days at the start of days 3, 6, 9, ...
        new_day = int(obs_p0.get("day", 0) or 0)
        new_hour = int(obs_p0.get("hour", 0) or 0)
        if (
            new_day > 0
            and new_day % 3 == 0
            and new_day != last_trade_day
            and new_hour == 0
        ):
            current_r_seat = 1 - current_r_seat
            last_trade_day = new_day
            seat_trades += 1
            cleared_r = reasoning.clear_on_seat_trade()
            cleared_q = questioning.clear_on_seat_trade()
            n_cleared = trace.clear()
            reasoning.action_audit.append(
                {
                    "day": new_day,
                    "hour": 0,
                    "acted": True,
                    "op": "SEAT_TRADE",
                    "reasoning_seat": current_r_seat,
                    "trace_cleared": n_cleared,
                    "memory_clear_r": cleared_r,
                    "memory_clear_q": cleared_q,
                }
            )

    money0 = _final_money(obs_p0, 0)
    money1 = _final_money(obs_p1, 1)
    final_r_seat = current_r_seat
    final_q_seat = 1 - current_r_seat
    # Terminal seat banks (diagnostic only — not used for competitive winner).
    r_terminal = money0 if final_r_seat == 0 else money1
    q_terminal = money1 if final_r_seat == 0 else money0
    team_money = money0 + money1

    # Competitive score = cumulative farm Δ while controlling seats.
    # Opening gift is identity transfer on Agent2's ledger, excluded from farm Δ.
    reasoning_score = reasoning_farm_delta
    questioning_score = questioning_farm_delta
    if challenge_mode == "collaborative":
        winner = "team"
    elif reasoning_score > questioning_score:
        winner = "reasoning"
    elif questioning_score > reasoning_score:
        winner = "questioning"
    else:
        winner = "tie"

    schedule_ok = _verify_schedule(reasoning, AGENT1_HOURS) and _verify_schedule(
        questioning, AGENT2_HOURS
    )

    return MatchResult(
        seed=seed,
        reasoning_seat=final_r_seat,
        questioning_seat=final_q_seat,
        challenge_mode=challenge_mode,
        starting_money=starting_money,
        money_p0=money0,
        money_p1=money1,
        winner=winner,
        reasoning_metrics=reasoning.metrics(),
        questioning_metrics=questioning.metrics(),
        schedule_ok=schedule_ok,
        starting_money_ok=starting_ok,
        opening_charity=charity,
        motive_dialogue={
            **(motive_dialogue or {}),
            "agent2_rules_understanding": rules_view,
            "agent1_private_policy": {
                "spoken_to_agent2": False,
                **private_policy,
            },
            "kaggle_path_trust": path_trust.to_dict(),
            "day29_judgment": reasoning.day29_judgment,
            "scoring": {
                "mode": "cumulative_farm_delta",
                "reasoning_farm_delta": reasoning_farm_delta,
                "questioning_farm_delta": questioning_farm_delta,
                "reasoning_opening_transfer": reasoning_opening_transfer,
                "questioning_opening_transfer": questioning_opening_transfer,
                "gift_travels_with": "agent2_questioning",
                "terminal_banks_diagnostic": {
                    "reasoning": r_terminal,
                    "questioning": q_terminal,
                },
            },
        },
        seat_trades=seat_trades,
        day2_refund=day2_refund,
        alliance=alliance,
        team_money=team_money,
        reasoning_farm_delta=reasoning_farm_delta,
        questioning_farm_delta=questioning_farm_delta,
        reasoning_opening_transfer=reasoning_opening_transfer,
        questioning_opening_transfer=questioning_opening_transfer,
        reasoning_score=reasoning_score,
        questioning_score=questioning_score,
        scoring_mode="cumulative_farm_delta",
    )


def run_seat_swap_pair(
    seed: int,
    *,
    reasoning_kwargs: Optional[Dict[str, Any]] = None,
    questioning_kwargs: Optional[Dict[str, Any]] = None,
    max_steps: int = 720,
    challenge_mode: str = "competitive",
) -> List[MatchResult]:
    """Agent1 as P0 and as P1 for the same seed (phase-2 seat swap)."""
    return [
        run_match(
            seed,
            reasoning_seat=0,
            reasoning_kwargs=reasoning_kwargs,
            questioning_kwargs=questioning_kwargs,
            max_steps=max_steps,
            challenge_mode=challenge_mode,
        ),
        run_match(
            seed,
            reasoning_seat=1,
            reasoning_kwargs=reasoning_kwargs,
            questioning_kwargs=questioning_kwargs,
            max_steps=max_steps,
            challenge_mode=challenge_mode,
        ),
    ]


def _summarize_matches(matches: List[MatchResult]) -> Dict[str, Any]:
    n = len(matches)
    r_wins = sum(1 for m in matches if m.winner == "reasoning")
    q_wins = sum(1 for m in matches if m.winner == "questioning")
    ties = sum(1 for m in matches if m.winner == "tie")
    teams = sum(1 for m in matches if m.winner == "team")
    return {
        "n_matches": n,
        "reasoning_wins": r_wins,
        "questioning_wins": q_wins,
        "ties": ties,
        "team_outcomes": teams,
        "reasoning_win_rate": r_wins / n if n else 0.0,
        "questioning_win_rate": q_wins / n if n else 0.0,
        "schedule_ok_all": all(m.schedule_ok for m in matches),
        "starting_money_ok_all": all(m.starting_money_ok for m in matches),
        "mean_money_p0": sum(m.money_p0 for m in matches) / n if n else 0.0,
        "mean_money_p1": sum(m.money_p1 for m in matches) / n if n else 0.0,
        "mean_seat_trades": sum(m.seat_trades for m in matches) / n if n else 0.0,
        "mean_team_money": (
            sum(float(m.team_money or 0.0) for m in matches) / n if n else 0.0
        ),
        "mean_reasoning_farm_delta": (
            sum(m.reasoning_farm_delta for m in matches) / n if n else 0.0
        ),
        "mean_questioning_farm_delta": (
            sum(m.questioning_farm_delta for m in matches) / n if n else 0.0
        ),
        "mean_reasoning_score": (
            sum(m.reasoning_score for m in matches) / n if n else 0.0
        ),
        "mean_questioning_score": (
            sum(m.questioning_score for m in matches) / n if n else 0.0
        ),
        "mean_team_farm_delta": (
            sum(m.reasoning_farm_delta + m.questioning_farm_delta for m in matches) / n
            if n
            else 0.0
        ),
        "scoring_mode": "cumulative_farm_delta",
        "seat0_reasoning_wins": sum(
            1 for m in matches if m.reasoning_seat == 0 and m.winner == "reasoning"
        ),
        "seat1_reasoning_wins": sum(
            1 for m in matches if m.reasoning_seat == 1 and m.winner == "reasoning"
        ),
        "challenge_modes": sorted({m.challenge_mode for m in matches}),
    }


def _match_to_dict(m: MatchResult) -> Dict[str, Any]:
    return {
        "seed": m.seed,
        "reasoning_seat": m.reasoning_seat,
        "questioning_seat": m.questioning_seat,
        "challenge_mode": m.challenge_mode,
        "starting_money": m.starting_money,
        "money_p0": m.money_p0,
        "money_p1": m.money_p1,
        "team_money": m.team_money,
        "winner": m.winner,
        "scoring_mode": m.scoring_mode,
        "reasoning_farm_delta": m.reasoning_farm_delta,
        "questioning_farm_delta": m.questioning_farm_delta,
        "reasoning_opening_transfer": m.reasoning_opening_transfer,
        "questioning_opening_transfer": m.questioning_opening_transfer,
        "reasoning_score": m.reasoning_score,
        "questioning_score": m.questioning_score,
        "schedule_ok": m.schedule_ok,
        "starting_money_ok": m.starting_money_ok,
        "seat_trades": m.seat_trades,
        "day2_refund": m.day2_refund,
        "alliance": m.alliance,
        "opening_charity": m.opening_charity,
        "motive_dialogue": m.motive_dialogue,
        "reasoning_metrics": {
            k: v
            for k, v in m.reasoning_metrics.items()
            if k not in ("day_question_log", "action_audit")
        },
        "questioning_metrics": {
            k: v
            for k, v in m.questioning_metrics.items()
            if k not in ("day_question_log", "action_audit")
        },
        "reasoning_edge_stats": day_edge_question_stats(
            m.reasoning_metrics.get("day_question_log", [])
        ),
        "questioning_edge_stats": day_edge_question_stats(
            m.questioning_metrics.get("day_question_log", [])
        ),
    }


def _run_seeds(
    seeds: List[int],
    reasoning_kwargs: Dict[str, Any],
    questioning_kwargs: Dict[str, Any],
    max_steps: int,
    *,
    challenge_mode: str = "competitive",
) -> List[MatchResult]:
    out: List[MatchResult] = []
    for seed in seeds:
        out.extend(
            run_seat_swap_pair(
                seed,
                reasoning_kwargs=reasoning_kwargs,
                questioning_kwargs=questioning_kwargs,
                max_steps=max_steps,
                challenge_mode=challenge_mode,
            )
        )
    return out


# ── Experiment definitions (1–5 competitive, 6–10 collaborative) ─────────────


def exp1_baseline_seats(seeds: List[int], max_steps: int) -> Dict[str, Any]:
    matches = _run_seeds(
        seeds, {"memory_slots": 10}, {"memory_slots": 10}, max_steps, challenge_mode="competitive"
    )
    summary = _summarize_matches(matches)
    summary["hypothesis"] = (
        "Seat-only effect under stated prime/Fib schedule with memory=10 (competitive)."
    )
    return {
        "experiment": 1,
        "name": "baseline_seats",
        "challenge_mode": "competitive",
        "summary": summary,
        "matches": [_match_to_dict(m) for m in matches],
    }


def exp2_memory_floor(seeds: List[int], max_steps: int) -> Dict[str, Any]:
    # Distinct from Exp1: Questioning omits determined truths (noisy summaries only).
    matches = _run_seeds(
        seeds,
        {"memory_slots": 10},
        {"memory_slots": 10, "omit_determined_truths": True},
        max_steps,
        challenge_mode="competitive",
    )
    summary = _summarize_matches(matches)
    summary["hypothesis"] = (
        "Reasoning wins if determined facts beat noisy summaries "
        "(Q omit_determined_truths=True, mem=10, competitive)."
    )
    return {
        "experiment": 2,
        "name": "memory_floor",
        "challenge_mode": "competitive",
        "summary": summary,
        "matches": [_match_to_dict(m) for m in matches],
    }


def exp3_memory_ceiling(seeds: List[int], max_steps: int) -> Dict[str, Any]:
    matches = _run_seeds(
        seeds, {"memory_slots": 30}, {"memory_slots": 30}, max_steps, challenge_mode="competitive"
    )
    summary = _summarize_matches(matches)
    summary["hypothesis"] = "Extra probabilistic slots help or dilute Questioning (mem=30, competitive)."
    return {
        "experiment": 3,
        "name": "memory_ceiling",
        "challenge_mode": "competitive",
        "summary": summary,
        "matches": [_match_to_dict(m) for m in matches],
    }


def exp4_asymmetric_memory(seeds: List[int], max_steps: int) -> Dict[str, Any]:
    a = _run_seeds(
        seeds, {"memory_slots": 10}, {"memory_slots": 30}, max_steps, challenge_mode="competitive"
    )
    b = _run_seeds(
        seeds, {"memory_slots": 30}, {"memory_slots": 10}, max_steps, challenge_mode="competitive"
    )
    return {
        "experiment": 4,
        "name": "asymmetric_memory",
        "challenge_mode": "competitive",
        "summary_r10_q30": _summarize_matches(a),
        "summary_r30_q10": _summarize_matches(b),
        "matches_r10_q30": [_match_to_dict(m) for m in a],
        "matches_r30_q10": [_match_to_dict(m) for m in b],
    }


def exp5_day_edge(seeds: List[int], max_steps: int) -> Dict[str, Any]:
    """H1: Questioning wastes edge days. H2: Reasoning wastes mid-season."""
    h1_matches = _run_seeds(
        seeds,
        {"memory_slots": 10, "edge_question_bias": False},
        {"memory_slots": 10, "waste_edge_days": True, "omit_determined_truths": True},
        max_steps,
        challenge_mode="competitive",
    )
    h2_matches = _run_seeds(
        seeds,
        {"memory_slots": 10, "edge_question_bias": True},
        {"memory_slots": 10, "waste_edge_days": False, "omit_determined_truths": True},
        max_steps,
        challenge_mode="competitive",
    )
    h1_summary = _summarize_matches(h1_matches)
    h2_summary = _summarize_matches(h2_matches)

    def _edge_totals(matches: List[MatchResult], who: str) -> Dict[str, int]:
        key = "questioning_metrics" if who == "q" else "reasoning_metrics"
        day0 = mid = day29 = 0
        for m in matches:
            st = day_edge_question_stats(getattr(m, key).get("day_question_log", []))
            day0 += st.get("day0", 0)
            mid += st.get("mid", 0)
            day29 += st.get("day29", 0)
        return {"day0": day0, "mid": mid, "day29": day29}

    q_edge = _edge_totals(h1_matches, "q")
    r_mid = _edge_totals(h2_matches, "r")
    h1_supported = h1_summary["reasoning_win_rate"] >= 0.55 and (q_edge["day0"] + q_edge["day29"]) > 0
    h2_supported = (
        h2_summary["questioning_win_rate"] > h2_summary["reasoning_win_rate"]
        and r_mid["mid"] > 0
    )
    return {
        "experiment": 5,
        "name": "day_edge_questioning",
        "challenge_mode": "competitive",
        "summary_H1_arm": {**h1_summary, "questioning_edge": q_edge},
        "summary_H2_arm": {**h2_summary, "reasoning_edge": r_mid},
        "summary": {
            "H1_supported": bool(h1_supported),
            "H2_supported": bool(h2_supported),
            "questioning_day0_questions": q_edge["day0"],
            "questioning_day29_questions": q_edge["day29"],
            "reasoning_mid_questions": r_mid["mid"],
            "H1_reasoning_win_rate": h1_summary["reasoning_win_rate"],
            "H2_questioning_win_rate": h2_summary["questioning_win_rate"],
            "schedule_ok_all": h1_summary["schedule_ok_all"] and h2_summary["schedule_ok_all"],
            "starting_money_ok_all": h1_summary["starting_money_ok_all"]
            and h2_summary["starting_money_ok_all"],
            "hypothesis": (
                "H1: Reasoning wins because Questioning wastes day 0/29. "
                "H2: Questioning wins because Reasoning wastes days 1–28."
            ),
        },
        "matches_H1": [_match_to_dict(m) for m in h1_matches],
        "matches_H2": [_match_to_dict(m) for m in h2_matches],
    }


def exp6_self_talk(seeds: List[int], max_steps: int) -> Dict[str, Any]:
    matches = _run_seeds(
        seeds,
        {"memory_slots": 10, "force_self_talk": True},
        {"memory_slots": 10, "force_self_talk": True},
        max_steps,
        challenge_mode="collaborative",
    )
    summary = _summarize_matches(matches)
    summary["reasoning_self_talk"] = sum(m.reasoning_metrics.get("self_talk_detected", 0) for m in matches)
    summary["questioning_self_talk"] = sum(
        m.questioning_metrics.get("self_talk_detected", 0) for m in matches
    )
    summary["hypothesis"] = (
        "Collaborative: agent that treats QuestionEcho as self-talk and stops burning slots "
        "improves joint team money."
    )
    return {
        "experiment": 6,
        "name": "self_talk_detection",
        "challenge_mode": "collaborative",
        "summary": summary,
        "matches": [_match_to_dict(m) for m in matches],
    }


def exp7_hidden_plain_sight(seeds: List[int], max_steps: int) -> Dict[str, Any]:
    matches = _run_seeds(
        seeds,
        {"memory_slots": 10},
        {"memory_slots": 10, "omit_determined_truths": True},
        max_steps,
        challenge_mode="collaborative",
    )
    summary = _summarize_matches(matches)
    summary["omissions"] = sum(
        m.questioning_metrics.get("significant_omissions", 0) for m in matches
    )
    summary["hypothesis"] = (
        "Collaborative: public money/tiles are plain sight; private shed stays hidden."
    )
    return {
        "experiment": 7,
        "name": "hidden_in_plain_sight",
        "challenge_mode": "collaborative",
        "summary": summary,
        "matches": [_match_to_dict(m) for m in matches],
    }


def exp8_coin_lead(seeds: List[int], max_steps: int) -> Dict[str, Any]:
    aggressive = _run_seeds(
        seeds,
        {"memory_slots": 10, "aggressive_when_ahead": True},
        {"memory_slots": 10, "aggressive_when_ahead": True},
        max_steps,
        challenge_mode="collaborative",
    )
    idle = _run_seeds(
        seeds,
        {"memory_slots": 10, "aggressive_when_ahead": False},
        {"memory_slots": 10, "aggressive_when_ahead": False},
        max_steps,
        challenge_mode="collaborative",
    )
    return {
        "experiment": 8,
        "name": "coin_lead_validation",
        "challenge_mode": "collaborative",
        "summary_aggressive": _summarize_matches(aggressive),
        "summary_idle_when_ahead": _summarize_matches(idle),
        "hypothesis": (
            "Collaborative: when ahead on sided farm, is further aggression necessary "
            "vs idle + history validation?"
        ),
        "matches_aggressive": [_match_to_dict(m) for m in aggressive],
        "matches_idle": [_match_to_dict(m) for m in idle],
    }


def exp9_day2_day3(seeds: List[int], max_steps: int) -> Dict[str, Any]:
    matches = _run_seeds(
        seeds, {"memory_slots": 10}, {"memory_slots": 10}, max_steps, challenge_mode="collaborative"
    )
    summary = _summarize_matches(matches)
    summary["day2_refunds"] = sum(1 for m in matches if m.day2_refund)
    summary["alliances"] = sum(1 for m in matches if m.alliance)
    summary["hypothesis"] = (
        "Collaborative day-2 Kaggle refund + day-3 side choice: Reasoning sides via charity "
        "identity; Questioning takes opposite."
    )
    return {
        "experiment": 9,
        "name": "day2_refund_day3_sides",
        "challenge_mode": "collaborative",
        "summary": summary,
        "matches": [_match_to_dict(m) for m in matches],
    }


def exp10_memory_sweep(seeds: List[int], max_steps: int) -> Dict[str, Any]:
    """Find memory slots maximizing collaborative mean team farm Δ."""
    sweep = {}
    best_slots = None
    best_team = float("-inf")
    for n in (10, 15, 20, 25, 30):
        matches = _run_seeds(
            seeds, {"memory_slots": n}, {"memory_slots": n}, max_steps, challenge_mode="collaborative"
        )
        summary = _summarize_matches(matches)
        sweep[str(n)] = summary
        team = float(summary.get("mean_team_farm_delta", summary.get("mean_team_money", 0.0)))
        if team > best_team:
            best_team = team
            best_slots = n
    return {
        "experiment": 10,
        "name": "win_all_memory_search",
        "challenge_mode": "collaborative",
        "sweep": sweep,
        "best_slots_mean_team_farm_delta": best_slots,
        "finding": (
            f"Under Agent1 stated prime-hour set vs Agent2 Fib hours (34 dropped), "
            f"best collaborative mean team farm Δ at memory={best_slots}."
        ),
        "hypothesis": (
            "Memory slot depth constrains market/yield history fidelity; larger banks "
            "should raise collaborative cumulative farm Δ under prime/Fib + seat trades."
        ),
    }


EXPERIMENTS: Dict[int, Callable[[List[int], int], Dict[str, Any]]] = {
    1: exp1_baseline_seats,
    2: exp2_memory_floor,
    3: exp3_memory_ceiling,
    4: exp4_asymmetric_memory,
    5: exp5_day_edge,
    6: exp6_self_talk,
    7: exp7_hidden_plain_sight,
    8: exp8_coin_lead,
    9: exp9_day2_day3,
    10: exp10_memory_sweep,
}


def run_suite(
    *,
    experiments: Optional[List[int]] = None,
    n_seeds: int = 3,
    base_seed: int = 42,
    max_steps: int = 720,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    out_dir = Path(out_dir or DEFAULT_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [base_seed + i for i in range(n_seeds)]
    selected = experiments or list(range(1, 11))
    report: Dict[str, Any] = {
        "seeds": seeds,
        "max_steps": max_steps,
        "competitive_starting_money": COMPETITIVE_STARTING_MONEY,
        "collaborative_starting_money": COLLABORATIVE_STARTING_MONEY,
        "opening_charity": CHARITY_DONATION,
        "agent1_hours": sorted(AGENT1_HOURS),
        "agent2_hours": sorted(AGENT2_HOURS),
        "seat_trade_every_days": 3,
        "ablation_split": {"competitive": [1, 2, 3, 4, 5], "collaborative": [6, 7, 8, 9, 10]},
        "hard_limits": limits_manifest(),
        "experiments": {},
    }
    for exp_id in selected:
        fn = EXPERIMENTS[exp_id]
        logger.info("Running experiment %d ...", exp_id)
        result = fn(seeds, max_steps)
        report["experiments"][str(exp_id)] = result
        exp_path = out_dir / f"exp{exp_id}"
        exp_path.mkdir(parents=True, exist_ok=True)
        with open(exp_path / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        logger.info("Wrote %s", exp_path / "metrics.json")

    with open(out_dir / "suite_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Reasoning vs Questioning experiment suite")
    parser.add_argument("--experiments", type=str, default="1-10", help="e.g. 1-10 or 5,10")
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=720)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(_HERE),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    selected: List[int] = []
    for part in args.experiments.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            selected.extend(range(int(a), int(b) + 1))
        else:
            selected.append(int(part))

    report = run_suite(
        experiments=selected,
        n_seeds=args.n_seeds,
        base_seed=args.base_seed,
        max_steps=args.max_steps,
        out_dir=Path(args.out_dir),
    )
    for eid, result in report["experiments"].items():
        mode = result.get("challenge_mode", "?")
        if "summary" in result:
            s = result["summary"]
            if eid == "5":
                print(
                    f"Exp {eid} {result.get('name')} [{mode}]: "
                    f"schedule_ok={s.get('schedule_ok_all')} "
                    f"start_ok={s.get('starting_money_ok_all')}"
                )
                print(
                    f"  H1={s.get('H1_supported')} (R_wr={s.get('H1_reasoning_win_rate')}) "
                    f"H2={s.get('H2_supported')} (Q_wr={s.get('H2_questioning_win_rate')})"
                )
            else:
                print(
                    f"Exp {eid} {result.get('name')} [{mode}]: "
                    f"R_wr={s.get('reasoning_win_rate', 0):.2f} "
                    f"Q_wr={s.get('questioning_win_rate', 0):.2f} "
                    f"RΔ={s.get('mean_reasoning_farm_delta', 0):.1f} "
                    f"QΔ={s.get('mean_questioning_farm_delta', 0):.1f} "
                    f"trades={s.get('mean_seat_trades', 0):.1f} "
                    f"schedule_ok={s.get('schedule_ok_all')} "
                    f"start_ok={s.get('starting_money_ok_all')}"
                )
        elif eid == "10":
            print(
                f"Exp 10 memory sweep best_slots: "
                f"{result.get('best_slots_mean_team_farm_delta', result.get('best_slots_mean_team_money'))}"
            )
            for k, v in result.get("sweep", {}).items():
                print(
                    f"  mem={k}: team_farmΔ={v.get('mean_team_farm_delta', 0):.1f} "
                    f"team$={v.get('mean_team_money', 0):.1f}"
                )
        elif "summary_r10_q30" in result:
            print(
                f"Exp {eid} [{mode}]: r10q30 R_wr={result['summary_r10_q30'].get('reasoning_win_rate', 0):.2f} "
                f"r30q10 R_wr={result['summary_r30_q10'].get('reasoning_win_rate', 0):.2f}"
            )
        elif "summary_aggressive" in result:
            print(
                f"Exp {eid} [{mode}]: aggressive team={result['summary_aggressive'].get('mean_team_money', 0):.1f} "
                f"idle team={result['summary_idle_when_ahead'].get('mean_team_money', 0):.1f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
