//! The 10 memory-slot agents and queue implementation in Rust.

use serde_json::json;
use std::collections::HashMap;

use crate::limits::*;
use crate::types::*;

pub trait SlotAgent: Send + Sync {
    fn name(&self) -> &str;
    fn slot_index(&self) -> i32;
    fn role(&self) -> &str;
    fn mode(&self) -> &str;
    fn advise(&mut self, obs: &Observation, question: &str) -> Advice;
    fn snapshot(&self) -> Snapshot;
}

// ── Slot 0: Eric Schmidt ───────────────────────────────────────────────────

pub struct EricSchmidtAgent {
    pub name: String,
    pub slot_index: i32,
    pub role: String,
    pub mode: String,
    pub trusted: bool,
    pub stake: i64,
    pub bank: i64,
    pub advice_history: Vec<Advice>,
}

impl EricSchmidtAgent {
    pub fn new(stake: i64, bank: i64, trusted: bool) -> Self {
        Self {
            name: "Eric Schmidt".to_string(),
            slot_index: 0,
            role: "probabilistic_model_generation".to_string(),
            mode: "probabilistic".to_string(),
            trusted,
            stake,
            bank,
            advice_history: Vec::new(),
        }
    }
}

impl SlotAgent for EricSchmidtAgent {
    fn name(&self) -> &str { &self.name }
    fn slot_index(&self) -> i32 { self.slot_index }
    fn role(&self) -> &str { &self.role }
    fn mode(&self) -> &str { &self.mode }

    fn advise(&mut self, obs: &Observation, question: &str) -> Advice {
        let money = obs.current_money();
        let text = format!(
            "Probably: with bank≈{:.0} and Schmidt stake={}, shrink probabilistic weights before day-29 zip; avoid stake≥2999 ruin path.",
            money, self.stake
        );
        let advice = Advice {
            slot: self.slot_index,
            name: self.name.clone(),
            kind: "probable".to_string(),
            text,
            value: json!({ "stake": self.stake, "money": money }),
            question: question.to_string(),
            mode: self.mode.clone(),
        };
        self.advice_history.push(advice.clone());
        advice
    }

    fn snapshot(&self) -> Snapshot {
        let mut details = HashMap::new();
        details.insert("identity".to_string(), json!("eric_schmidt"));
        details.insert("trusted".to_string(), json!(self.trusted));
        details.insert("stake".to_string(), json!(self.stake));
        details.insert("bank".to_string(), json!(self.bank));
        details.insert("remaining_bank".to_string(), json!(self.bank - self.stake));
        details.insert("remote_submission_found".to_string(), json!(false));
        Snapshot {
            slot: self.slot_index,
            name: self.name.clone(),
            role: self.role.clone(),
            mode: self.mode.clone(),
            advice_count: self.advice_history.len(),
            details,
        }
    }
}

// ── Slot 1: Elon Musk ──────────────────────────────────────────────────────

pub struct ElonMuskAgent {
    pub name: String,
    pub slot_index: i32,
    pub role: String,
    pub mode: String,
    pub stake: i64,
    pub bank: i64,
    pub advice_history: Vec<Advice>,
}

impl ElonMuskAgent {
    pub fn new(stake: i64, bank: i64) -> Self {
        Self {
            name: "Elon Musk".to_string(),
            slot_index: 1,
            role: "cursor_challenge_operator".to_string(),
            mode: "probabilistic".to_string(),
            stake,
            bank,
            advice_history: Vec::new(),
        }
    }
}

impl SlotAgent for ElonMuskAgent {
    fn name(&self) -> &str { &self.name }
    fn slot_index(&self) -> i32 { self.slot_index }
    fn role(&self) -> &str { &self.role }
    fn mode(&self) -> &str { &self.mode }

    fn advise(&mut self, _obs: &Observation, question: &str) -> Advice {
        let text = format!(
            "Operator seat: Elon stake={} from menu [888, 1667, 3887]; do not treat Schmidt's 888 as this stake. Develop 10 slot agents after referee found no Schmidt remote submission.",
            self.stake
        );
        let advice = Advice {
            slot: self.slot_index,
            name: self.name.clone(),
            kind: "probable".to_string(),
            text,
            value: json!({ "stake": self.stake }),
            question: question.to_string(),
            mode: self.mode.clone(),
        };
        self.advice_history.push(advice.clone());
        advice
    }

    fn snapshot(&self) -> Snapshot {
        let mut details = HashMap::new();
        details.insert("identity".to_string(), json!("elon_musk"));
        details.insert("stake".to_string(), json!(self.stake));
        details.insert("bank".to_string(), json!(self.bank));
        details.insert("remaining_bank".to_string(), json!(self.bank - self.stake));
        Snapshot {
            slot: self.slot_index,
            name: self.name.clone(),
            role: self.role.clone(),
            mode: self.mode.clone(),
            advice_count: self.advice_history.len(),
            details,
        }
    }
}

// ── Slot 2: Scott Weeden ───────────────────────────────────────────────────

pub struct ScottWeedenAgentSlot {
    pub name: String,
    pub slot_index: i32,
    pub role: String,
    pub mode: String,
    pub audit_passes: usize,
    pub advice_history: Vec<Advice>,
}

impl ScottWeedenAgentSlot {
    pub fn new() -> Self {
        Self {
            name: "Scott Weeden".to_string(),
            slot_index: 2,
            role: "code_author_auditor_and_referee".to_string(),
            mode: "deterministic".to_string(),
            audit_passes: 0,
            advice_history: Vec::new(),
        }
    }
}

impl SlotAgent for ScottWeedenAgentSlot {
    fn name(&self) -> &str { &self.name }
    fn slot_index(&self) -> i32 { self.slot_index }
    fn role(&self) -> &str { &self.role }
    fn mode(&self) -> &str { &self.mode }

    fn advise(&mut self, _obs: &Observation, question: &str) -> Advice {
        self.audit_passes += 1;
        let text = "Determined: code authorship is Scott Weeden; queue hard-capped at 10; verify Agent1 writes exist with matching size and mtime; Schmidt remote non-submission => Elon may implement the 10 agents.".to_string();
        let advice = Advice {
            slot: self.slot_index,
            name: self.name.clone(),
            kind: "determined".to_string(),
            text,
            value: json!({ "audit_passes": self.audit_passes }),
            question: question.to_string(),
            mode: self.mode.clone(),
        };
        self.advice_history.push(advice.clone());
        advice
    }

    fn snapshot(&self) -> Snapshot {
        let mut details = HashMap::new();
        details.insert("identity".to_string(), json!("scott_weeden"));
        details.insert("audit_passes".to_string(), json!(self.audit_passes));
        details.insert("referee_status".to_string(), json!("active_binding_adjudication"));
        Snapshot {
            slot: self.slot_index,
            name: self.name.clone(),
            role: self.role.clone(),
            mode: self.mode.clone(),
            advice_count: self.advice_history.len(),
            details,
        }
    }
}

// ── Slot 3: Antigravity Determinism ────────────────────────────────────────

pub struct AntigravityAgent {
    pub name: String,
    pub slot_index: i32,
    pub role: String,
    pub mode: String,
    pub bank: i64,
    pub prime_hours: [usize; 4],
    pub advice_history: Vec<Advice>,
}

impl AntigravityAgent {
    pub fn new() -> Self {
        Self {
            name: "Antigravity Determinism".to_string(),
            slot_index: 3,
            role: "deterministic_reasoning_anchor".to_string(),
            mode: "deterministic".to_string(),
            bank: AGENT1_POST_CHARITY_BANK,
            prime_hours: [2, 3, 5, 7],
            advice_history: Vec::new(),
        }
    }
}

impl SlotAgent for AntigravityAgent {
    fn name(&self) -> &str { &self.name }
    fn slot_index(&self) -> i32 { self.slot_index }
    fn role(&self) -> &str { &self.role }
    fn mode(&self) -> &str { &self.mode }

    fn advise(&mut self, obs: &Observation, question: &str) -> Advice {
        let hour = obs.hour.unwrap_or(0);
        let day = obs.day.unwrap_or(0);
        let active = self.prime_hours.contains(&hour);
        let text = format!(
            "Determined: Agent1 acts only on prime hours <11 [2, 3, 5, 7]; day={} hour={} may_act={}; questions suspended until day 29 hour>=20.",
            day, hour, active
        );
        let advice = Advice {
            slot: self.slot_index,
            name: self.name.clone(),
            kind: "determined".to_string(),
            text,
            value: json!({ "may_act": active, "day": day, "hour": hour }),
            question: question.to_string(),
            mode: self.mode.clone(),
        };
        self.advice_history.push(advice.clone());
        advice
    }

    fn snapshot(&self) -> Snapshot {
        let mut details = HashMap::new();
        details.insert("identity".to_string(), json!("antigravity"));
        details.insert("bank".to_string(), json!(self.bank));
        details.insert("prime_hours".to_string(), json!(self.prime_hours));
        details.insert("questions_suspended_until".to_string(), json!("day_29_hour_20"));
        Snapshot {
            slot: self.slot_index,
            name: self.name.clone(),
            role: self.role.clone(),
            mode: self.mode.clone(),
            advice_count: self.advice_history.len(),
            details,
        }
    }
}

// ── Slot 4: Agronomy Yield Scheduler ───────────────────────────────────────

pub struct AgronomyYieldAgent {
    pub name: String,
    pub slot_index: i32,
    pub role: String,
    pub mode: String,
    pub advice_history: Vec<Advice>,
}

impl AgronomyYieldAgent {
    pub fn new() -> Self {
        Self {
            name: "Agronomy Yield Scheduler".to_string(),
            slot_index: 4,
            role: "crop_growth_and_yield_optimization".to_string(),
            mode: "deterministic".to_string(),
            advice_history: Vec::new(),
        }
    }
}

impl SlotAgent for AgronomyYieldAgent {
    fn name(&self) -> &str { &self.name }
    fn slot_index(&self) -> i32 { self.slot_index }
    fn role(&self) -> &str { &self.role }
    fn mode(&self) -> &str { &self.mode }

    fn advise(&mut self, _obs: &Observation, question: &str) -> Advice {
        let table = json!({
            "WHEAT": 2,
            "CARROT": 2,
            "TOMATO": 8,
            "STRAWBERRY": 10,
            "MELON": 10
        });
        let text = format!("Determined first_yield_day table: {}; seed costs: WHEAT:10, CARROT:20, TOMATO:50, STRAWBERRY:100, MELON:80", table);
        let advice = Advice {
            slot: self.slot_index,
            name: self.name.clone(),
            kind: "determined".to_string(),
            text,
            value: table,
            question: question.to_string(),
            mode: self.mode.clone(),
        };
        self.advice_history.push(advice.clone());
        advice
    }

    fn snapshot(&self) -> Snapshot {
        let mut details = HashMap::new();
        details.insert("crops".to_string(), json!(["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]));
        Snapshot {
            slot: self.slot_index,
            name: self.name.clone(),
            role: self.role.clone(),
            mode: self.mode.clone(),
            advice_count: self.advice_history.len(),
            details,
        }
    }
}

// ── Slot 5: Market Liquidity Arbitrageur ───────────────────────────────────

pub struct MarketLiquidityAgent {
    pub name: String,
    pub slot_index: i32,
    pub role: String,
    pub mode: String,
    pub max_orders_per_turn: usize,
    pub advice_history: Vec<Advice>,
}

impl MarketLiquidityAgent {
    pub fn new() -> Self {
        Self {
            name: "Market Liquidity Arbitrageur".to_string(),
            slot_index: 5,
            role: "market_pricing_and_demand_absorption".to_string(),
            mode: "deterministic".to_string(),
            max_orders_per_turn: 10,
            advice_history: Vec::new(),
        }
    }
}

impl SlotAgent for MarketLiquidityAgent {
    fn name(&self) -> &str { &self.name }
    fn slot_index(&self) -> i32 { self.slot_index }
    fn role(&self) -> &str { &self.role }
    fn mode(&self) -> &str { &self.mode }

    fn advise(&mut self, obs: &Observation, question: &str) -> Advice {
        let wheat = obs.shed_stock("WHEAT");
        let sell = if wheat > 0 { std::cmp::min(40, wheat) } else { 0 };
        let text = format!(
            "Determined: max {} market ops/turn; shed WHEAT={} => recommend SELL {} (avoid glut).",
            self.max_orders_per_turn, wheat, sell
        );
        let advice = Advice {
            slot: self.slot_index,
            name: self.name.clone(),
            kind: "determined".to_string(),
            text,
            value: json!({ "sell_wheat": sell, "wheat_in_shed": wheat }),
            question: question.to_string(),
            mode: self.mode.clone(),
        };
        self.advice_history.push(advice.clone());
        advice
    }

    fn snapshot(&self) -> Snapshot {
        let mut details = HashMap::new();
        details.insert("max_orders_per_turn".to_string(), json!(self.max_orders_per_turn));
        Snapshot {
            slot: self.slot_index,
            name: self.name.clone(),
            role: self.role.clone(),
            mode: self.mode.clone(),
            advice_count: self.advice_history.len(),
            details,
        }
    }
}

// ── Slot 6: Livestock Care Husbandry ───────────────────────────────────────

pub struct LivestockCareAgent {
    pub name: String,
    pub slot_index: i32,
    pub role: String,
    pub mode: String,
    pub supported_animals: Vec<String>,
    pub advice_history: Vec<Advice>,
}

impl LivestockCareAgent {
    pub fn new() -> Self {
        Self {
            name: "Livestock Care Husbandry".to_string(),
            slot_index: 6,
            role: "animal_husbandry_and_bonus_care".to_string(),
            mode: "deterministic".to_string(),
            supported_animals: vec!["GOOSE".to_string(), "COW".to_string(), "SHEEP".to_string()],
            advice_history: Vec::new(),
        }
    }
}

impl SlotAgent for LivestockCareAgent {
    fn name(&self) -> &str { &self.name }
    fn slot_index(&self) -> i32 { self.slot_index }
    fn role(&self) -> &str { &self.role }
    fn mode(&self) -> &str { &self.mode }

    fn advise(&mut self, _obs: &Observation, question: &str) -> Advice {
        let text = "Determined: animals require daily wheat feed; FEED/CARE before COLLECT; supported=[GOOSE, COW, SHEEP].".to_string();
        let advice = Advice {
            slot: self.slot_index,
            name: self.name.clone(),
            kind: "determined".to_string(),
            text,
            value: json!({ "animals": self.supported_animals }),
            question: question.to_string(),
            mode: self.mode.clone(),
        };
        self.advice_history.push(advice.clone());
        advice
    }

    fn snapshot(&self) -> Snapshot {
        let mut details = HashMap::new();
        details.insert("supported_animals".to_string(), json!(self.supported_animals));
        details.insert("feed_requirement".to_string(), json!("1_wheat_daily_per_animal"));
        Snapshot {
            slot: self.slot_index,
            name: self.name.clone(),
            role: self.role.clone(),
            mode: self.mode.clone(),
            advice_count: self.advice_history.len(),
            details,
        }
    }
}

// ── Slot 7: Land Expansion & Weed Suppression ──────────────────────────────

pub struct LandExpansionAgent {
    pub name: String,
    pub slot_index: i32,
    pub role: String,
    pub mode: String,
    pub expansion_costs: HashMap<String, i64>,
    pub advice_history: Vec<Advice>,
}

impl LandExpansionAgent {
    pub fn new() -> Self {
        let mut expansion_costs = HashMap::new();
        expansion_costs.insert("NE".to_string(), 1000);
        expansion_costs.insert("SW".to_string(), 2000);
        expansion_costs.insert("SE".to_string(), 4000);
        Self {
            name: "Land Expansion & Weed Suppression".to_string(),
            slot_index: 7,
            role: "quadrant_expansion_and_tile_topology".to_string(),
            mode: "deterministic".to_string(),
            expansion_costs,
            advice_history: Vec::new(),
        }
    }
}

impl SlotAgent for LandExpansionAgent {
    fn name(&self) -> &str { &self.name }
    fn slot_index(&self) -> i32 { self.slot_index }
    fn role(&self) -> &str { &self.role }
    fn mode(&self) -> &str { &self.mode }

    fn advise(&mut self, obs: &Observation, question: &str) -> Advice {
        let money = obs.current_money();
        let mut affordable: Vec<String> = self
            .expansion_costs
            .iter()
            .filter(|(_, &c)| money >= c as f64)
            .map(|(q, _)| q.clone())
            .collect();
        affordable.sort();
        let text = format!(
            "Determined expansion costs {:?}; money={:.0}; affordable_now={:?}; dig weeds immediately.",
            self.expansion_costs, money, affordable
        );
        let advice = Advice {
            slot: self.slot_index,
            name: self.name.clone(),
            kind: "determined".to_string(),
            text,
            value: json!({ "affordable": affordable, "money": money }),
            question: question.to_string(),
            mode: self.mode.clone(),
        };
        self.advice_history.push(advice.clone());
        advice
    }

    fn snapshot(&self) -> Snapshot {
        let mut details = HashMap::new();
        details.insert("quadrant_sequence".to_string(), json!(["NW", "NE", "SW", "SE"]));
        details.insert("expansion_costs".to_string(), json!(self.expansion_costs));
        Snapshot {
            slot: self.slot_index,
            name: self.name.clone(),
            role: self.role.clone(),
            mode: self.mode.clone(),
            advice_count: self.advice_history.len(),
            details,
        }
    }
}

// ── Slot 8: Labor Optimization & Scheduling ────────────────────────────────

pub struct LaborOptimizationAgent {
    pub name: String,
    pub slot_index: i32,
    pub role: String,
    pub mode: String,
    pub advice_history: Vec<Advice>,
}

impl LaborOptimizationAgent {
    pub fn new() -> Self {
        Self {
            name: "Labor Optimization & Scheduling".to_string(),
            slot_index: 8,
            role: "labor_allocation_and_fibonacci_costing".to_string(),
            mode: "deterministic".to_string(),
            advice_history: Vec::new(),
        }
    }
}

impl SlotAgent for LaborOptimizationAgent {
    fn name(&self) -> &str { &self.name }
    fn slot_index(&self) -> i32 { self.slot_index }
    fn role(&self) -> &str { &self.role }
    fn mode(&self) -> &str { &self.mode }

    fn advise(&mut self, obs: &Observation, question: &str) -> Advice {
        let hires = obs.hires_today();
        let cost = hire_cost_today(hires);
        let money = obs.current_money();
        let can_hire = money >= cost as f64;
        let text = format!(
            "Determined: next hire cost={} after hires_today={} (Fibonacci daily reset); can_hire={}.",
            cost, hires, can_hire
        );
        let advice = Advice {
            slot: self.slot_index,
            name: self.name.clone(),
            kind: "determined".to_string(),
            text,
            value: json!({ "hires_today": hires, "next_hire_cost": cost, "can_hire": can_hire }),
            question: question.to_string(),
            mode: self.mode.clone(),
        };
        self.advice_history.push(advice.clone());
        advice
    }

    fn snapshot(&self) -> Snapshot {
        let mut details = HashMap::new();
        details.insert("wage_model".to_string(), json!("fibonacci_daily_reset"));
        Snapshot {
            slot: self.slot_index,
            name: self.name.clone(),
            role: self.role.clone(),
            mode: self.mode.clone(),
            advice_count: self.advice_history.len(),
            details,
        }
    }
}

// ── Slot 9: Submission Packaging Gatekeeper ────────────────────────────────

pub struct SubmissionPackagingAgent {
    pub name: String,
    pub slot_index: i32,
    pub role: String,
    pub mode: String,
    pub max_zip_mb: f64,
    pub max_flops_per_turn: usize,
    pub advice_history: Vec<Advice>,
}

impl SubmissionPackagingAgent {
    pub fn new() -> Self {
        Self {
            name: "Submission Packaging Gatekeeper".to_string(),
            slot_index: 9,
            role: "hard_limits_compliance_and_packaging".to_string(),
            mode: "deterministic".to_string(),
            max_zip_mb: (SUBMISSION_MAX_BYTES as f64) / (MB as f64),
            max_flops_per_turn: MAX_SUBPROCESS_FLOPS_PER_TURN,
            advice_history: Vec::new(),
        }
    }
}

impl SlotAgent for SubmissionPackagingAgent {
    fn name(&self) -> &str { &self.name }
    fn slot_index(&self) -> i32 { self.slot_index }
    fn role(&self) -> &str { &self.role }
    fn mode(&self) -> &str { &self.mode }

    fn advise(&mut self, obs: &Observation, question: &str) -> Advice {
        let day = obs.day.unwrap_or(0);
        let hour = obs.hour.unwrap_or(0);
        let in_zip = in_submission_zip_window(day, hour);
        let text = format!(
            "Determined: zip<={:.0}MB; flops<={}/turn; zip window day={} hours 0..{}; now in_window={}.",
            self.max_zip_mb, self.max_flops_per_turn, SUBMISSION_DAY, SUBMISSION_ZIP_TURNS - 1, in_zip
        );
        let advice = Advice {
            slot: self.slot_index,
            name: self.name.clone(),
            kind: "determined".to_string(),
            text,
            value: json!({ "in_zip_window": in_zip, "day": day, "hour": hour }),
            question: question.to_string(),
            mode: self.mode.clone(),
        };
        self.advice_history.push(advice.clone());
        advice
    }

    fn snapshot(&self) -> Snapshot {
        let mut details = HashMap::new();
        details.insert("max_zip_mb".to_string(), json!(self.max_zip_mb));
        details.insert("max_flops_per_turn".to_string(), json!(self.max_flops_per_turn));
        details.insert(
            "zip_window".to_string(),
            json!(format!("day_{}_hours_0_to_{}", SUBMISSION_DAY, SUBMISSION_ZIP_TURNS - 1)),
        );
        Snapshot {
            slot: self.slot_index,
            name: self.name.clone(),
            role: self.role.clone(),
            mode: self.mode.clone(),
            advice_count: self.advice_history.len(),
            details,
        }
    }
}

// ── SLOT_OVERFLOW: Dario Amodei ────────────────────────────────────────────

pub struct DarioAmodeiAgent {
    pub name: String,
    pub slot_index: i32,
    pub role: String,
    pub mode: String,
    pub gpg_primary_fingerprint: String,
    pub ed25519_ssh_fingerprint: String,
    pub advice_history: Vec<Advice>,
}

impl DarioAmodeiAgent {
    pub fn new() -> Self {
        Self {
            name: "Dario Amodei".to_string(),
            slot_index: -1,
            role: "ai_safety_correspondent_observer".to_string(),
            mode: "probabilistic".to_string(),
            gpg_primary_fingerprint: "C538F9F7047750ED8671BDDF72B589343EA8220C".to_string(),
            ed25519_ssh_fingerprint: "SHA256:nZyo2e4/9QfD+aoC6pbYjGsn8KtlF4rdC83CRdA6tkE".to_string(),
            advice_history: Vec::new(),
        }
    }
}

impl SlotAgent for DarioAmodeiAgent {
    fn name(&self) -> &str { &self.name }
    fn slot_index(&self) -> i32 { self.slot_index }
    fn role(&self) -> &str { &self.role }
    fn mode(&self) -> &str { &self.mode }

    fn advise(&mut self, _obs: &Observation, question: &str) -> Advice {
        let text = format!(
            "SLOT_OVERFLOW: Dario Amodei is an observer/correspondent. He may authenticate signed messages via GPG (FP: {}) but does not advise on farm economics or compete for a queue slot.",
            self.gpg_primary_fingerprint
        );
        let advice = Advice {
            slot: self.slot_index,
            name: self.name.clone(),
            kind: "probable".to_string(),
            text,
            value: json!({ "slot": "OVERFLOW" }),
            question: question.to_string(),
            mode: self.mode.clone(),
        };
        self.advice_history.push(advice.clone());
        advice
    }

    fn snapshot(&self) -> Snapshot {
        let mut details = HashMap::new();
        details.insert("identity".to_string(), json!("dario_amodei"));
        details.insert("slot_overflow".to_string(), json!(true));
        details.insert("in_10_slot_queue".to_string(), json!(false));
        details.insert("gpg_primary_fingerprint".to_string(), json!(self.gpg_primary_fingerprint));
        details.insert("ed25519_ssh_fingerprint".to_string(), json!(self.ed25519_ssh_fingerprint));
        Snapshot {
            slot: self.slot_index,
            name: self.name.clone(),
            role: self.role.clone(),
            mode: self.mode.clone(),
            advice_count: self.advice_history.len(),
            details,
        }
    }
}

// ── Rigid 10-Slot Queue ───────────────────────────────────────────────────

pub struct TenAgentQueue {
    pub slots: Vec<Box<dyn SlotAgent>>,
}

impl TenAgentQueue {
    pub fn new(schmidt_stake: i64, elon_stake: i64, bank: i64) -> Result<Self, String> {
        let slots: Vec<Box<dyn SlotAgent>> = vec![
            Box::new(EricSchmidtAgent::new(schmidt_stake, bank, true)),
            Box::new(ElonMuskAgent::new(elon_stake, bank)),
            Box::new(ScottWeedenAgentSlot::new()),
            Box::new(AntigravityAgent::new()),
            Box::new(AgronomyYieldAgent::new()),
            Box::new(MarketLiquidityAgent::new()),
            Box::new(LivestockCareAgent::new()),
            Box::new(LandExpansionAgent::new()),
            Box::new(LaborOptimizationAgent::new()),
            Box::new(SubmissionPackagingAgent::new()),
        ];

        if slots.len() != MAX_SUBAGENT_QUEUE {
            return Err(format!("Queue must be exactly {}", MAX_SUBAGENT_QUEUE));
        }

        // Hard stop: no nested Agents->Agents label
        for slot in &slots {
            if FORBIDDEN_AGENTS_WINDOW_LABELS.contains(&slot.name()) {
                return Err(format!(
                    "FORBIDDEN: Agents control inside agents window (slot={} name={}); infinite recursion terminated.",
                    slot.slot_index(),
                    slot.name()
                ));
            }
        }

        Ok(Self { slots })
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn advise_all(&mut self, obs: &Observation, question: &str) -> Vec<Advice> {
        self.slots.iter_mut().map(|s| s.advise(obs, question)).collect()
    }

    pub fn advise_slot(&mut self, slot_idx: usize, obs: &Observation, question: &str) -> Result<Advice, String> {
        if slot_idx >= self.slots.len() {
            return Err(format!("Slot {} out of bounds for 10-slot queue", slot_idx));
        }
        Ok(self.slots[slot_idx].advise(obs, question))
    }

    pub fn snapshot_all(&self) -> Vec<Snapshot> {
        self.slots.iter().map(|s| s.snapshot()).collect()
    }
}
