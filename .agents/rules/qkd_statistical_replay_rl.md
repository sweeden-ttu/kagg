# Replay-Guided Statistical QKD Reinforcement Learning Architecture

## 1. Architectural Foundations
The **Replay-Guided Statistical QKD Agent** (`QKDReplayRLAgent`) fuses high-performing offline match replay trajectories with real-time probabilistic, spatio-temporal, and multi-head decision tree modulations:

1. **Replay Trace Core**:
   - Ingests compressed 720-step / 2,880-step champion trajectories ($a_0, a_1, \dots$) extracted from tournament-winning replays (e.g. 100k+ coin matches).
   - Encoded as `zlib` + `base85` serialized action sequences for zero-overhead runtime decoding.

2. **Strict Canonical 3-Probe QKD Basis**:
   - The strategic question bank is strictly scoped to 3 canonical game probes:
     - **$Q$-Channel**: `Q_DAYS_REMAINING` $=\max\left(0, \frac{30.0 - \text{day}}{30.0}\right)$ or normalized multi-season clock.
     - **$K$-Channel**: `K_OPP_WALLET_BALANCE` $=\frac{\ln(1 + \text{opp\_money})}{12.0}$ (opponent liquid bank balance).
     - **$D$-Channel**: `D_SUBAGENTS_WALLET_BALANCE` $=\frac{\ln(1 + \text{my\_money})}{12.0} \times (1.0 + 0.1 \times \text{hands})$ (self capital & active workforce capacity).
   - Zero hardcoded prime/Fibonacci questions. Observation mapping via `map_observation_to_qkd` and `map_observation_to_questions`.

3. **Two-Stage `Att(...)` Decision Pipeline**:
   - All agents execute attention in two strict stages:
     ```python
     # Stage 1: Intrinsic market-optimal evaluation
     stage1_action = agent.Att(obs, initial_terminal_configuration, market_functions)

     # Stage 2: Adversarial opponent-modulated synthesis
     stage2_action = agent.Att(obs, initial_terminal_configuration, market_functions, opponent_functions)
     ```
   - **`InitialTerminalConfiguration`**:
     - `number_of_days` (and `.days`): Total competition duration (30, 90, or 120 days).
     - `amount_of_money` (and `.initial_money`): Starting wallet capital ($3,000.00).
     - `start_price_of_every_commodity` (and `.start_prices`): Starting prices across all 9 commodities:
       `WHEAT` ($25), `CARROT` ($35), `TOMATO` ($60), `STRAWBERRY` ($120), `MELON` ($250), `EGG` ($50), `MILK` ($160), `WOOL` ($200), `FERTILIZER` ($100).
   - **`market_functions`** (Series of Future Price Functions):
     - `predict_future_price(item, inventory_delta, horizon_hours)`: Projects clearing prices post-trade and town absorption.
     - `predict_price_trajectory(item, horizon_days)`: Vector of daily future price projections.
     - `get_price_elasticity(item, cur_inventory)`: Marginal price slope $\frac{\partial P}{\partial Q}$.
     - `get_demand_absorption_rate(item)`: Aggregated town shop + center demand.
     - `compute_optimal_sell_batch(item, cur_inventory)`: Non-depressing liquidation batch sizes.
   - **`opponent_functions`**:
     - `predict_opponent_money_trajectory(horizon_days)`: Multi-day capital trajectory.
     - `predict_opponent_workforce_growth()`: Worker capacity count.
     - `estimate_opponent_market_impact(item)`: Sector impact from unlocked land.
     - `compute_adversarial_lead_gap()`: Margin lead $\Delta = \text{MyMoney} - \text{OppMoney}$.
     - `estimate_opponent_aggression()`: Land and hiring speed index.

4. **Floating-Point Convergence 2D Karnaugh Map (K-Map)**:
   - **Proportional Dimension Scaling**:
     - **Rows (Spatial Grid)**: $H \times W = 10 \times 10 = 100$ farm tiles (or $10 \times 10 \times 10 = 1,000$ voxels).
     - **Columns (Adversarial & Market Context)**: $N_{\text{opp}} \times N_{\text{commodities}} = 11 \times 9 = 99$ interaction channels.
     - **Exact Proportional Shape**: $(100, 99) = 9,900$ continuous floating-point decision cells.
   - **Continuous Float32 Storage Format (No Premature Quantization)**:
     - Stored as native `float32` probability matrix $P(x) \in [0.0, 1.0]$ requiring **$39,600$ bytes** ($38.7\text{ KB}$).
     - Preserves fractional decision confidence, continuous gradients, and epistemic uncertainty without quantization artifacts.
     - Soft continuous fuzzy logic operations ($\min$, $\max$, $1 - P$).
     - Compact binary serialization (`to_bytes()` / `from_bytes()`).
   - **Real-Time Asymptotic Convergence Tracking**:
     - **Polarization Ratio**: $\frac{1}{N} \sum |2P - 1| \in [0.0, 1.0]$ (monitors asymptotic convergence toward $\{0, 1\}$).
     - **Shannon Information Entropy**: $H(P) = -\frac{1}{N} \sum [P \log_2(P) + (1-P)\log_2(1-P)]$ in bits.
     - **Epistemic Uncertainty Variance**: $\frac{1}{N} \sum P(1 - P)$.
     - **Continuous Learning Dynamics**: Supports continuous Exponential Moving Average (EMA), logistic temperature annealing ($\tau \to \tau_{\min}$), and Bayesian log-odds accumulation.
   - **2-Dimensional Logarithmic Heatmap Output ($10^{-4} \to 10^0$)**:
     - Every training exit renders and saves a high-resolution 2-dimensional logarithmic heatmap (`kmap_logarithmic_heatmap.png`) using `LogNorm(vmin=1e-4, vmax=1.0)` over `inferno` colormap ($10,000 : 1$ dynamic range).
     - Separates farm quadrants with cyan dashed lines (every 25 rows) and league opponents with gold dotted lines (every 9 columns).
     - Also exports linear continuous probability heatmap (`kmap_polarization_heatmap.png`).

5. **$10 \times 10 \times 10$ Voxel Extractor & GBDT Capacity Scaling up to 100 MB**:
   - **3D Voxel Tensor**: $(10, 10, 10) = 1,000$ spatial elements representing tile types, crop IDs, growth stages, soil moisture, worker proximity fields, and price/threat potentials.
   - **1035-Dimensional Vector**: Concatenation of $1,000$ voxels $+ 3$ QKD probes $+ 32$ economic features.
   - **Multi-Head GBDT Policy**: Parameterized by Action Classifier, Spatial Tile Ranker, 9-Commodity Liquidation Regressor, and 120-Day Terminal Value Regressor.
   - **Iterative Capacity Scaling Loop**: Expands tree estimators across stages ($5\text{MB} \to 20\text{MB} \to 50\text{MB} \to 75\text{MB} \to \mathbf{99.25\text{MB}}$), strictly satisfying Kaggle's $< 100\text{ MB}$ payload limit.

6. **Dynamic Runtime RL Adaptation**:
   - **Weed Repair State Machine**: Tracks tile disruptions from random weed spawns, initiates dig operations, and replays / reschedules disrupted planting transactions.
   - **Market Price Impact & Urgency Scoring**: Sorts sell slots dynamically by marginal price impact:
     $$\text{Impact}(q) = q \cdot \max(0, P_{\text{curr}} - P_{\text{after}})$$
     $$\text{Urgency}(i) = \text{Score} \cdot \left(1 + \alpha \cdot \min\left(1.0, \frac{\text{Excess}_i / \text{Demand}_i}{10.0}\right)\right)$$

## 2. Competition & Submission Invariants
- Total standalone payload $\le 100\text{ MB}$ (with exact model accounting at ~99.25 MB).
- Turn execution latency $< 15.0\text{ ms}$ on CPU (well below the 1.0s actTimeout).
- Robust fallback legal actions (`PASS` / align hands) on unexpected runtime exceptions.
