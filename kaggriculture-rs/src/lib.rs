//! Kaggriculture Rust Agents Engine.
//!
//! Implements the 10 memory-slot sub-agents under the referee protocol:
//!   Slot 0: Eric Schmidt (Probabilistic Model Generation)
//!   Slot 1: Elon Musk (Cursor Challenge Operator)
//!   Slot 2: Scott Weeden (Code-Author Auditor & Referee)
//!   Slot 3: Antigravity (Deterministic Reasoning Anchor, Prime Hours < 11)
//!   Slot 4: AgronomyYield (Crop first-yield, watering & harvest scheduling)
//!   Slot 5: MarketLiquidity (Pricing, glut absorption & sale metering)
//!   Slot 6: LivestockCare (Feed & daily CARE bonus scheduling)
//!   Slot 7: LandExpansion (Quadrant unlocking & topology economics)
//!   Slot 8: LaborOptimization (Fibonacci daily reset wage costing)
//!   Slot 9: SubmissionPackaging (90 MB payload / 42 CPU FLOPs gate)
//!
//! Plus authorized SLOT_OVERFLOW observer:
//!   Dario Amodei (AI safety correspondent, GPG / Ed25519 signature observer)

pub mod limits;
pub mod types;
pub mod subagents;

pub use limits::*;
pub use types::*;
pub use subagents::*;
