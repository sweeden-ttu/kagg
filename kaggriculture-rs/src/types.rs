//! Types and data models for observations, advice, and snapshots.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FarmState {
    pub money: Option<f64>,
    pub hires_today: Option<usize>,
    pub land: Option<Vec<String>>,
    pub inventory: Option<HashMap<String, usize>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivateState {
    pub shed: Option<HashMap<String, usize>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub player: Option<usize>,
    pub step: Option<usize>,
    pub day: Option<usize>,
    pub hour: Option<usize>,
    pub farms: Option<Vec<FarmState>>,
    pub private: Option<PrivateState>,
    pub market: Option<HashMap<String, Value>>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

impl Observation {
    pub fn player_farm(&self) -> Option<&FarmState> {
        let p = self.player.unwrap_or(0);
        self.farms.as_ref().and_then(|f| f.get(p))
    }

    pub fn current_money(&self) -> f64 {
        self.player_farm().and_then(|f| f.money).unwrap_or(0.0)
    }

    pub fn hires_today(&self) -> usize {
        self.player_farm().and_then(|f| f.hires_today).unwrap_or(0)
    }

    pub fn shed_stock(&self, item: &str) -> usize {
        self.private
            .as_ref()
            .and_then(|p| p.shed.as_ref())
            .and_then(|s| s.get(item))
            .copied()
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Advice {
    pub slot: i32,
    pub name: String,
    pub kind: String, // "determined" | "probable"
    pub text: String,
    pub value: Value,
    pub question: String,
    pub mode: String, // "deterministic" | "probabilistic"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub slot: i32,
    pub name: String,
    pub role: String,
    pub mode: String,
    pub advice_count: usize,
    #[serde(flatten)]
    pub details: HashMap<String, Value>,
}
