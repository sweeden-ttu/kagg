# Multi-Agent Reinforcement Learning & Tiered Curriculum Rules

Guidelines for training and evaluating reinforcement learning agents in multi-agent competitive simulation environments (e.g. Kaggriculture).

---

## 1. Progressive Tier-by-Tier Boss Curriculum

1. **Strict Sequential Promotion**: In tiered reference ladders (e.g., Tier 0 to Tier 9), agents must prove mastery over the current tier boss before unlocking training rollouts for the next tier.
2. **Promotion Gate**: The clearance criterion is a deterministic $N$-game tournament block (e.g. 4 games $\times$ full episode steps) with a target win rate $\ge 75\%$ (or $\ge 3/4$ wins).
3. **Retention & Anti-Regression Sampling**: During active boss training, allocate **80%** of matches against the active tier boss and **20%** against randomly sampled previously cleared tiers to prevent catastrophic forgetting.
4. **Strict Tier Scope**: Never evaluate or train against higher-tier reference bots beyond the active tier under investigation.

---

## 2. Hindsight Experience Replay (HER) for Multi-Stage Strategy Games

1. **Delayed Economic Rewards**: In environments where actions have delayed returns (e.g., planting $\rightarrow$ daily watering $\rightarrow$ crop maturity wait $\rightarrow$ harvesting $\rightarrow$ market sale), raw step rewards fail to guide credit assignment effectively.
2. **Milestone Relabeling**: Relabel trajectory transitions with hindsight credit upon reaching key milestones:
   - **Maturity Harvest**: Propagate credit backward to preceding planting and watering actions.
   - **Market Liquidation**: Propagate credit backward across inventory accumulation phases.
   - **Final Victory**: Propagate competitive advantage bonuses to high-margin economic choices.
3. **Buffer Ingestion**: Feed both standard and hindsight-reweighted transitions into a **Prioritized Replay Buffer (PER)** with elevated initial priorities.

---

## 3. Economic Scaling & Labor Throughput Invariants

1. **Day-1 Labor Scaling**: Workforce throughput is the primary driver of compounding cash. Morning workforce hiring (4–5 hands) must be executed at Hour 0 across multiple market order slots as soon as `money >= 10`.
2. **Land Expansion & Capital Buffers**: Purchase the first extra quadrant (1,000 coins) as early as Day 2–3 while holding a calibrated cash buffer (~150 coins, `land_buffer: 150`) so seed buying and daily wages are never interrupted.
3. **Staged Workforce Scaling**: Scale crew size in lockstep with available tiles (e.g. `[(0, 5), (3, 8), (8, 8)]`) to maximize daily watering coverage while avoiding early Fibonacci wage inflation before land is unlocked.
4. **Price-Floor Supply-Glut Protection**: Set strict price floors (e.g. Carrot $\ge 14$, Tomato $\ge 24$) with metered batches (30–35 units) to avoid selling into market gluts caused by multi-farm production, deferring full dumping to Days 28–30.
5. **Premium Crop Fertilization**: When farming premium high-grossing crops (e.g. Melon), purchase and apply fertilizer during the active bonus window to boost yield units to the maximum 6-unit cap.
6. **Quadratic Market Metering**: For premium crops subject to quadratic market price penalties, strictly meter sell orders into 12–14 unit lots with high price floors (e.g. $\ge 110$) to capture high-margin market absorption ahead of opponents.
7. **Livestock Working Capital & Feed Float Reserve**: Animals generate immense cash flow once CARE is active (3 milk / 2 days; 4 wool / 3 days), but require an 8-day maturation delay. A mandatory **16-day feed cash float** (`feed_float_days: 16`) must be held before purchasing livestock to prevent permanent starvation death during market price spikes.
8. **Dynamic Endgame Animal Harvesting**: Starting on **Day 27+**, drop animal harvest thresholds (`animal_harvest_at = 1`) to capture all single residual units of high-margin milk and wool before season end.
9. **Endgame Feed Liquidation**: Halt feed purchases on Day 27 and daily feeding on Days 28–30 (when no further yield cycles can mature), liquidating all remaining shed wheat into pure cash gain.
10. **Value-Sorted Liquidation Ordering**: Prioritize market selling queues strictly in descending order of commodity unit price (`['WOOL', 'MILK', 'MELON', 'CARROT', 'WHEAT']`) in 25-unit chunks to maximize revenue extraction before turn 720.
11. **Seed Surplus Capping**: Hard-cap seed inventory ($\le 12$ units) and halt seed purchases after Day 24 to prevent cash hoarding traps.
12. **Multi-Phase Land Succession**:
   - **Days 1–4**: Fast-turnaround crops (e.g. Carrot - 3-day maturity) to rapidly bootstrap capital.
   - **Days 5–16**: Dedicate majority acreage (65%) to premium fertilized crops (Melon) with fast-cycle fillers.
   - **Days 16–25**: Pivot all post-cutoff acreage into rapid 3-day turnaround crops (Carrot/Wheat) rather than leaving land idle.
   - **Days 25+**: Endgame watering, harvesting, and liquidation only.

---

## 4. Evaluation & Submission Packaging

1. **Full-Horizon Simulation**: Never evaluate tournament gates using truncated/dry-run step counts (e.g. 50 steps). Always simulate the full competition season horizon (e.g. 720 steps) to allow economic compounding and crop lifecycles to complete.
2. **Containerized Submission Bundle**: When exporting `agent.py`, ensure all supporting modules (`kaggriculture_adapter.py`, `kaggriculture_path_b_rebuild.py`, etc.) are packaged directly into the export directory alongside `models/model.pth`.
