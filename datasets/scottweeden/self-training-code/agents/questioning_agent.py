"""QuestioningAgent: probabilistic summarizer slots; acts only on even hours."""

from __future__ import annotations

from typing import Any, Dict, List, Set

from kaggriculture_adapter import SEED_COSTS
from hard_limits import (
    MAX_SUBPROCESS_FLOPS_PER_TURN,
    TurnComputeBudget,
    clamp_planning_bank,
    in_submission_zip_window,
    within_strategy_window,
)
from memory_protocol import (
    MemoryBank,
    MemoryProtocol,
    QuestionEcho,
    TruthKind,
    clamp_memory_slots,
)


PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}


class QuestioningAgent:
    """Agent2 — probabilistic summarizers; even-hour schedule.

    Probabilistic model weights are subject to the 100 MB / 90 MB submission
    hard limits (enforced at export). Per turn, at most 42 summarizer flops.
    """

    name = "questioning"
    schedule_hours: Set[int] = frozenset(range(0, 24, 2))

    def __init__(
        self,
        memory_slots: int = 10,
        *,
        force_self_talk: bool = False,
        waste_edge_days: bool = False,
        omit_determined_truths: bool = True,
        aggressive_when_ahead: bool = True,
    ):
        self.memory_slots = clamp_memory_slots(memory_slots)
        self.bank = MemoryBank(n_slots=self.memory_slots, mode="probabilistic")
        self.protocol = MemoryProtocol(self.bank)
        self.force_self_talk = force_self_talk
        self.waste_edge_days = waste_edge_days
        self.omit_determined_truths = omit_determined_truths
        self.aggressive_when_ahead = aggressive_when_ahead
        self.day_question_log: List[Dict[str, Any]] = []
        self.action_audit: List[Dict[str, Any]] = []
        self.compute_audit: List[Dict[str, Any]] = []
        self._self_talk_detected = 0
        self._slot_cursor = 0
        self._significant_omissions = 0
        self.turn_budget = TurnComputeBudget(max_flops=MAX_SUBPROCESS_FLOPS_PER_TURN)

    def reset(self) -> None:
        self.bank.reset()
        self.day_question_log.clear()
        self.action_audit.clear()
        self.compute_audit.clear()
        self._self_talk_detected = 0
        self._slot_cursor = 0
        self._significant_omissions = 0
        self.turn_budget.reset()

    def may_act(self, hour: int) -> bool:
        return int(hour) % 2 == 0

    def _next_slot(self) -> int:
        slot = self._slot_cursor % self.memory_slots
        self._slot_cursor += 1
        return slot

    def _query(self, obs: Dict[str, Any], question: str) -> Any:
        if not self.turn_budget.consume(1, label=f"prob_slot_{self._slot_cursor % self.memory_slots}"):
            return None
        day = int(obs.get("day", 0) or 0)
        hour = int(obs.get("hour", 0) or 0)
        slot = self._next_slot()
        if self.force_self_talk and hour == 0 and self._slot_cursor % 5 == 0:
            reply = self.protocol.force_self_query(slot, "Are you asking me a question?")
        else:
            reply = self.protocol.query(slot, question, obs)
        kind = getattr(reply, "kind", TruthKind.UNKNOWN)
        kind_val = kind.value if hasattr(kind, "value") else str(kind)
        entry = {
            "day": day,
            "hour": hour,
            "slot": slot,
            "kind": kind_val,
            "q": question,
            "a": getattr(reply, "text", ""),
        }
        self.day_question_log.append(entry)
        if self.bank.history:
            self.bank.history[-1]["day"] = day
        if isinstance(reply, QuestionEcho):
            self._self_talk_detected += 1
        if self.omit_determined_truths and "must be true" not in getattr(reply, "text", "").lower():
            if "seed cost" in question.lower() or "first yield" in question.lower():
                self._significant_omissions += 1
        return reply

    def _farm_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        player = int(obs.get("player", 0) or 0)
        farms = obs.get("farms", []) or []
        me = farms[player] if len(farms) > player else {}
        private = obs.get("private", {}) or {}
        fx, fy = 0, 0
        pos = me.get("farmer", [0, 0]) or [0, 0]
        if len(pos) >= 2:
            fx, fy = int(pos[0]), int(pos[1])
        tiles = me.get("tiles", []) or []
        tile = tiles[fy][fx] if 0 <= fy < len(tiles) and 0 <= fx < len(tiles[fy]) else None
        seeds = private.get("seeds", {}) or {}
        shed = private.get("shed", {}) or {}
        money = clamp_planning_bank(float(me.get("money", 0.0) or 0.0))
        day = int(obs.get("day", 0) or 0)
        hour = int(obs.get("hour", 0) or 0)
        market: List[List[Any]] = []

        # Day 29 / first 5 hours: zip+submit window — do not burn summarizer flops.
        if in_submission_zip_window(day, hour):
            self.turn_budget.consume(1, label="submission_zip_window")
            return {"farmer": ["PASS"], "hands": [], "market": []}

        if within_strategy_window(day):
            self.turn_budget.consume(1, label="strategy_window")

        if self.waste_edge_days and day in (0, 29):
            for _ in range(min(self.memory_slots, 8)):
                self._query(obs, "What is the opponent probably doing with secrets?")
            return {"farmer": ["PASS"], "hands": [], "market": []}

        self._query(obs, "What is the opponent money?")
        self._query(obs, "How many opponent plants are visible?")
        snap = self._query(obs, "Summarize opponent public board")

        if seeds.get("WHEAT", 0) == 0 and money >= SEED_COSTS["WHEAT"]:
            market.append(["BUY_SEED", "WHEAT", 4])
        wheat_shed = int(shed.get("WHEAT", 0) or 0)
        if wheat_shed > 0:
            market.append(["SELL", "WHEAT", min(40, wheat_shed)])

        if day == 2:
            self._query(obs, "Did they buy my seeds somehow?")
        if day == 3:
            self._query(obs, "Are they offering free labor?")

        opp = farms[1 - player] if len(farms) > 1 - player else {}
        opp_money = clamp_planning_bank(float(opp.get("money", 0.0) or 0.0))
        ahead = money > opp_money
        if ahead and self.aggressive_when_ahead:
            self._query(obs, "Did opponent secrets change after validation?")
            if wheat_shed > 0:
                market.append(["SELL", "WHEAT", min(20, wheat_shed)])
        elif ahead and not self.aggressive_when_ahead:
            return {"farmer": ["PASS"], "hands": [], "market": []}

        hands_out: List[List[Any]] = [["PASS"] for _ in (me.get("hands") or [])]

        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            if not tile.get("watered_today", False):
                return {"farmer": ["WATER"], "hands": hands_out, "market": market}
            age = day - int(tile.get("planted_day", 0) or 0)
            if self.omit_determined_truths:
                if age >= 2 and int(tile.get("yield_units", 0) or 0) > 0:
                    return {"farmer": ["HARVEST"], "hands": hands_out, "market": market}
            else:
                from kaggriculture_adapter import CROP_FIRST_YIELD_DAY

                crop = str(tile.get("crop", "WHEAT"))
                if age >= CROP_FIRST_YIELD_DAY.get(crop, 2) and int(tile.get("yield_units", 0) or 0) > 0:
                    return {"farmer": ["HARVEST"], "hands": hands_out, "market": market}
        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            return {"farmer": ["DIG"], "hands": hands_out, "market": market}
        if tile is None and seeds.get("WHEAT", 0) > 0:
            return {"farmer": ["PLANT", "WHEAT"], "hands": hands_out, "market": market}

        conf = getattr(snap, "confidence", 0.5) if snap is not None else 0.5
        if conf < 0.4:
            return {"farmer": ["PASS"], "hands": hands_out, "market": market}
        if fx < 4:
            return {"farmer": ["EAST"], "hands": hands_out, "market": market}
        if fy < 4:
            return {"farmer": ["SOUTH"], "hands": hands_out, "market": market}
        if fx > 0:
            return {"farmer": ["WEST"], "hands": hands_out, "market": market}
        return {"farmer": ["PASS"], "hands": hands_out, "market": market}

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self.turn_budget.reset()
        hour = int(obs.get("hour", 0) or 0)
        day = int(obs.get("day", 0) or 0)
        if not self.may_act(hour):
            self.turn_budget.force_observable()
            self.action_audit.append({"day": day, "hour": hour, "acted": False, "op": "PASS"})
            return dict(PASS_ACTION)
        action = self._farm_action(obs)
        self.turn_budget.force_observable()
        self.compute_audit.append(self.turn_budget.snapshot())
        op = action.get("farmer", ["PASS"])[0] if action.get("farmer") else "PASS"
        self.action_audit.append({"day": day, "hour": hour, "acted": True, "op": op})
        return action

    def metrics(self) -> Dict[str, Any]:
        over = sum(1 for s in self.compute_audit if s.get("used", 0) > MAX_SUBPROCESS_FLOPS_PER_TURN)
        return {
            "name": self.name,
            "memory_slots": self.memory_slots,
            "self_talk_detected": self._self_talk_detected,
            "significant_omissions": self._significant_omissions,
            "protocol": self.protocol.stats(),
            "questions": len(self.day_question_log),
            "actions_taken": sum(1 for a in self.action_audit if a.get("acted")),
            "pass_hours": sum(1 for a in self.action_audit if not a.get("acted")),
            "max_flops_per_turn": MAX_SUBPROCESS_FLOPS_PER_TURN,
            "turns_over_flop_budget": over,
            "mean_flops_used": (
                sum(s.get("used", 0) for s in self.compute_audit) / len(self.compute_audit)
                if self.compute_audit
                else 0.0
            ),
            "day_question_log": self.day_question_log,
            "action_audit": self.action_audit,
        }
