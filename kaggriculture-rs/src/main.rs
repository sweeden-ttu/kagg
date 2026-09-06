use serde_json::json;
use kaggriculture_rs::{
    Observation, TenAgentQueue, DarioAmodeiAgent, SlotAgent,
    SCHMIDT_STAKE_DEFAULT, ELON_MUSK_STAKE_DEFAULT, AGENT2_POST_CHARITY_BANK,
};

fn main() {
    println!("Initializing Kaggriculture 10 Sub-Agents (Rust Edition)...");

    let mut queue = match TenAgentQueue::new(
        SCHMIDT_STAKE_DEFAULT,
        ELON_MUSK_STAKE_DEFAULT,
        AGENT2_POST_CHARITY_BANK,
    ) {
        Ok(q) => q,
        Err(e) => {
            eprintln!("Failed to create agent queue: {}", e);
            std::process::exit(1);
        }
    };

    println!("Initialized rigid 10-slot queue (len = {}).", queue.len());

    // Construct a mock observation
    let obs_json = json!({
        "player": 0,
        "step": 48,
        "day": 2,
        "hour": 5,
        "farms": [
            {
                "money": 2450.0,
                "hires_today": 2,
                "land": ["NW"],
                "inventory": {
                    "WHEAT": 10
                }
            },
            {
                "money": 3100.0,
                "hires_today": 4,
                "land": ["NW", "NE"]
            }
        ],
        "private": {
            "shed": {
                "WHEAT": 25,
                "CARROT": 12
            }
        }
    });

    let obs: Observation = serde_json::from_value(obs_json).expect("valid obs");

    println!("\n=== Running advise_all on Day 2, Hour 5 (Prime hour < 11) ===");
    let advice_list = queue.advise_all(&obs, "Optimal rotation & expansion path?");

    for advice in &advice_list {
        println!(
            "[{:>2}] {:<32} | {:<13} | {}",
            advice.slot, advice.name, advice.kind, advice.text
        );
    }

    println!("\n=== Agent Snapshots ===");
    let snapshots = queue.snapshot_all();
    for snap in snapshots {
        println!(
            "Slot {}: {} (Role: {}, Mode: {})",
            snap.slot, snap.name, snap.role, snap.mode
        );
    }

    // Also check SLOT_OVERFLOW observer Dario Amodei
    println!("\n=== Checking SLOT_OVERFLOW Observer ===");
    let mut dario = DarioAmodeiAgent::new();
    let dario_advice = dario.advise(&obs, "Observer authentication status?");
    println!(
        "[{:>2}] {:<32} | {:<13} | {}",
        dario_advice.slot, dario.name, dario_advice.kind, dario_advice.text
    );
    let dario_snap = dario.snapshot();
    println!("Observer Snapshot: {:?}", dario_snap);

    println!("\nAll 10 sub-agents + observer executed successfully.");
}
