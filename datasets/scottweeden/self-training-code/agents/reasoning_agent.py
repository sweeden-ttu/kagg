"""ReasoningAgent: deterministic memory slots; acts only on prime hours < 11."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from kaggriculture_adapter import CROP_FIRST_YIELD_DAY, SEED_COSTS, hire_cost_today
from hard_limits import (
    MAX_SUBPROCESS_FLOPS_PER_TURN,
    TurnComputeBudget,
    clamp_planning_bank,
    in_submission_zip_window,
    within_strategy_window,
)
from memory_protocol import (
    PRIME_HOURS_LT_11,
    MemoryBank,
    MemoryProtocol,
    QuestionEcho,
    TruthKind,
    clamp_memory_slots,
)


PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}


class ReasoningAgent:
    """Agent1 — deterministic sub-agents in memory slots; prime-hour schedule."""

    name = "reasoning"
    schedule_hours: Set[int] = set(PRIME_HOURS_LT_11)

    def __init__(
        self,
        memory_slots: int = 10,
        *,
        force_self_talk: bool = False,
        aggressive_when_ahead: bool = True,
        edge_question_bias: bool = False,
    ):
        self.memory_slots = clamp_memory_slots(memory_slots)
        self.bank = MemoryBank(n_slots=self.memory_slots, mode="deterministic")
        self.protocol = MemoryProtocol(self.bank)
        self.force_self_talk = force_self_talk
        self.aggressive_when_ahead = aggressive_when_ahead
        self.edge_question_bias = edge_question_bias  # if True, waste mid-season (H2)
        self.day_question_log: List[Dict[str, Any]] = []
        self.action_audit: List[Dict[str, Any]] = []
        self._self_talk_detected = 0
        self._slot_cursor = 0
        self.turn_budget = TurnComputeBudget(max_flops=MAX_SUBPROCESS_FLOPS_PER_TURN)
        self.compute_audit: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self.bank.reset()
        self.day_question_log.clear()
        self.action_audit.clear()
        self.compute_audit.clear()
        self._self_talk_detected = 0
        self._slot_cursor = 0
        self.turn_budget.reset()

    def may_act(self, hour: int) -> bool:
        return int(hour) in self.schedule_hours

    def _next_slot(self) -> int:
        slot = self._slot_cursor % self.memory_slots
        self._slot_cursor += 1
        return slot

    def _query(self, obs: Dict[str, Any], question: str) -> Any:
        # Each memory-slot sub-agent query costs 1 of 42 CPU flop-units this turn.
        if not self.turn_budget.consume(1, label=f"det_slot_{self._slot_cursor % self.memory_slots}"):
            return None
        day = int(obs.get("day", 0) or 0)
        hour = int(obs.get("hour", 0) or 0)
        slot = self._next_slot()
        if self.force_self_talk and hour == 2 and self._slot_cursor % 7 == 0:
            reply = self.protocol.force_self_query(slot, question)
        else:
            reply = self.protocol.query(slot, question, obs)
        entry = {
            "day": day,
            "hour": hour,
            "slot": slot,
            "kind": getattr(reply, "kind", TruthKind.UNKNOWN).value
            if hasattr(getattr(reply, "kind", None), "value")
            else str(getattr(reply, "kind", "unknown")),
            "q": question,
            "a": getattr(reply, "text", ""),
        }
        self.day_question_log.append(entry)
        if self.bank.history:
            self.bank.history[-1]["day"] = day
        if isinstance(reply, QuestionEcho):
            self._self_talk_detected += 1
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
        market: List[List[Any]] = []

        # Day 29 zip window: conserve flops for packaging (deterministic code path).
        if in_submission_zip_window(day, int(obs.get("hour", 0) or 0)):
            self.turn_budget.consume(1, label="submission_zip_window")
            return {"farmer": ["PASS"], "hands": [], "market": []}

        # Prefer strategy discovery inside the first 14 days.
        if within_strategy_window(day):
            self.turn_budget.consume(1, label="strategy_window")

        self._query(obs, "What is my seed inventory?")
        self._query(obs, "What must be true about wheat seed cost?")
        hire_reply = self._query(obs, "What is the next hire cost?")

        if seeds.get("WHEAT", 0) == 0 and money >= SEED_COSTS["WHEAT"]:
            market.append(["BUY_SEED", "WHEAT", 4])
        wheat_shed = int(shed.get("WHEAT", 0) or 0)
        if wheat_shed > 0:
            market.append(["SELL", "WHEAT", min(40, wheat_shed)])

        if day == 2 and money >= SEED_COSTS["WHEAT"]:
            market.append(["BUY_SEED", "WHEAT", 2])
        if day == 3:
            cost = hire_cost_today(int(me.get("hires_today", 0) or 0))
            if hire_reply is not None and money >= cost:
                market.append(["HIRE"])

        opp = farms[1 - player] if len(farms) > 1 - player else {}
        opp_money = clamp_planning_bank(float(opp.get("money", 0.0) or 0.0))
        ahead = money > opp_money
        if ahead and not self.aggressive_when_ahead:
            return {"farmer": ["PASS"], "hands": [], "market": market[:1]}

        hands_out: List[List[Any]] = [["PASS"] for _ in (me.get("hands") or [])]

        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            crop = str(tile.get("crop", "WHEAT"))
            self._query(obs, f"What is the first yield day for {crop}?")
            if not tile.get("watered_today", False):
                return {"farmer": ["WATER"], "hands": hands_out, "market": market}
            age = day - int(tile.get("planted_day", 0) or 0)
            if age >= CROP_FIRST_YIELD_DAY.get(crop, 2) and int(tile.get("yield_units", 0) or 0) > 0:
                return {"farmer": ["HARVEST"], "hands": hands_out, "market": market}
            if int(shed.get("FERTILIZER", 0) or 0) > 0:
                return {"farmer": ["FERTILIZE"], "hands": hands_out, "market": market}
        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            return {"farmer": ["DIG"], "hands": hands_out, "market": market}
        if tile is None and seeds.get("WHEAT", 0) > 0:
            return {"farmer": ["PLANT", "WHEAT"], "hands": hands_out, "market": market}
        if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE") and tile.get("animal"):
            if not tile.get("fed_today", False):
                return {"farmer": ["FEED"], "hands": hands_out, "market": market}

        if self.edge_question_bias and 1 <= day <= 28:
            for _ in range(min(3, self.memory_slots)):
                self._query(obs, "What must be true today?")
            return {"farmer": ["PASS"], "hands": hands_out, "market": market}

        if fx > 0:
            return {"farmer": ["WEST"], "hands": hands_out, "market": market}
        if fy > 0:
            return {"farmer": ["NORTH"], "hands": hands_out, "market": market}
        if fx < 4:
            return {"farmer": ["EAST"], "hands": hands_out, "market": market}
        if fy < 4:
            return {"farmer": ["SOUTH"], "hands": hands_out, "market": market}
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
