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
2. **High-Volume Liquidation Sizing**: Market sales must clear in chunks of `min(40, shed_qty)` across multiple market order slots whenever shed inventory reaches pressure thresholds or Day $\ge 27$, ensuring $0 unliquidated stock at endgame.
3. **Seed Surplus Capping**: Hard-cap seed inventory ($\le 12$ units) and halt seed purchases after Day 24 to prevent cash hoarding traps.
4. **Dynamic Crop Lifecycle Rotation**:
   - **Days 1–4**: Fast-turnaround crops (e.g. Carrot - 3-day maturity) to rapidly bootstrap capital.
   - **Days 5–18**: Blend fast turnover (Carrot) with recurring daily harvest crops (Tomato) and high-yield staples (Wheat).
   - **Days 19–24**: Short-cycle crops only.
   - **Days 25+**: Endgame watering, harvesting, and liquidation only.

---

## 4. Evaluation & Submission Packaging

1. **Full-Horizon Simulation**: Never evaluate tournament gates using truncated/dry-run step counts (e.g. 50 steps). Always simulate the full competition season horizon (e.g. 720 steps) to allow economic compounding and crop lifecycles to complete.
2. **Containerized Submission Bundle**: When exporting `agent.py`, ensure all supporting modules (`kaggriculture_adapter.py`, `kaggriculture_path_b_rebuild.py`, etc.) are packaged directly into the export directory alongside `models/model.pth`.
