# Multi-Agent Reinforcement Learning & Tiered Curriculum Rules

Guidelines for training and evaluating reinforcement learning agents in multi-agent competitive simulation environments (e.g. Kaggriculture).

---

## 1. Progressive Tier-by-Tier Boss Curriculum

1. **Strict Sequential Promotion**: In tiered reference ladders (e.g., Tier 0 to Tier 9), agents must prove mastery over the current tier boss before unlocking training rollouts for the next tier.
2. **Promotion Gate**: The clearance criterion is a deterministic $N$-game tournament block (e.g. 4 games $\times$ full episode steps) with a target win rate $\ge 75\%$ (or $\ge 3/4$ wins).
3. **Retention & Anti-Regression Sampling**: During active boss training, allocate **80%** of matches against the active tier boss and **20%** against randomly sampled previously cleared tiers to prevent catastrophic forgetting.

---

## 2. Hindsight Experience Replay (HER) for Multi-Stage Strategy Games

1. **Delayed Economic Rewards**: In environments where actions have delayed returns (e.g., planting $\rightarrow$ daily watering $\rightarrow$ crop maturity wait $\rightarrow$ harvesting $\rightarrow$ market sale), raw step rewards fail to guide credit assignment effectively.
2. **Milestone Relabeling**: Relabel trajectory transitions with hindsight credit upon reaching key milestones:
   - **Maturity Harvest**: Propagate credit backward to preceding planting and watering actions.
   - **Market Liquidation**: Propagate credit backward across inventory accumulation phases.
   - **Final Victory**: Propagate competitive advantage bonuses to high-margin economic choices.
3. **Buffer Ingestion**: Feed both standard and hindsight-reweighted transitions into a **Prioritized Replay Buffer (PER)** with elevated initial priorities.

---

## 3. Evaluation & Submission Packaging

1. **Full-Horizon Simulation**: Never evaluate tournament gates using truncated/dry-run step counts (e.g. 50 steps). Always simulate the full competition season horizon (e.g. 720 steps) to allow economic compounding and crop lifecycles to complete.
2. **Containerized Submission Bundle**: When exporting `agent.py`, ensure all supporting modules (`kaggriculture_adapter.py`, `kaggriculture_path_b_rebuild.py`, etc.) are packaged directly into the export directory alongside `models/model.pth`.
