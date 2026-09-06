//! Operational ceilings and constants for Kaggriculture.

pub const MB: usize = 1024 * 1024;
pub const PROBABILISTIC_MODEL_MAX_BYTES: usize = 100 * MB;
pub const SUBMISSION_MAX_BYTES: usize = 90 * MB;
pub const SUBMISSION_PATCH_BUFFER_BYTES: usize = 10 * MB;

pub const MAX_SUBPROCESS_FLOPS_PER_TURN: usize = 42;
pub const MAX_PLANNING_BANK_COINS: usize = 50_000;

pub const SEASON_DAYS: usize = 30;
pub const SUBMISSION_DAY: usize = 29;
pub const SUBMISSION_ZIP_TURNS: usize = 5;

pub const MAX_SUBAGENT_QUEUE: usize = 10;

pub const AGENT1_POST_CHARITY_BANK: i64 = 2112; // 3000 - 888
pub const AGENT2_POST_CHARITY_BANK: i64 = 3888; // 3000 + 888

pub const SCHMIDT_STAKE_DEFAULT: i64 = 888;
pub const ELON_MUSK_STAKE_DEFAULT: i64 = 888;

pub const FORBIDDEN_AGENTS_WINDOW_LABELS: &[&str] = &[
    "Agents",
    "agents",
    "AGENTS",
    "AgentsWindow",
    "agents_window",
    "Open Agents",
];

pub fn in_submission_zip_window(day: usize, hour: usize) -> bool {
    day == SUBMISSION_DAY && hour < SUBMISSION_ZIP_TURNS
}

/// Fibonacci sequence for farm hand hiring costs (1, 1, 2, 3, 5, 8, 13, 21, ...)
pub fn hire_cost_today(hires_today: usize) -> i64 {
    if hires_today == 0 || hires_today == 1 {
        return 1;
    }
    let mut a = 1i64;
    let mut b = 1i64;
    for _ in 2..=hires_today {
        let next = a + b;
        a = b;
        b = next;
    }
    b
}
