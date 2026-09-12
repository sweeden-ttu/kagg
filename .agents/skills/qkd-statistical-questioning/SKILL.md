---
name: qkd-statistical-questioning
description: Evaluate and rank strategic questions over Q-K-D vector observations, compute continuous floating-point convergence 2D K-maps, and scale GBDT decision policies across 120-day scenarios up to 100 MB.
---

# QKD Statistical Questioning, Floating-Point Convergence 2D K-Maps & 100MB GBDT Scaling

This skill guides the construction, empirical evaluation, and execution of:
1. **Canonical Q-K-D Strategic Question Probing** ($Q = \text{Questions}, K = \text{Floating-Point Convergence 2D K-Map}, D = \text{Decisions}$).
2. **Floating-Point Convergence 2D Karnaugh Map (K-Map)** ($100 \text{ tiles} \times 99 \text{ channels}$ stored as continuous `float32` matrix of $39,600\text{ bytes}$ with real-time asymptotic convergence metrics).
3. **$10 \times 10 \times 10$ Spatial-Depth Voxel State Feature Extraction** ($1,035$ dense features).
4. **120-Day Scenario Multi-Head GBDT Policy Scaling up to 100 MB** ($4 \text{ seasons} = 120 \text{ days} = 2,880 \text{ turns}$).

---

## 1. Theoretical Foundations

### 1. The Canonical 3-Probe Strategic Basis:
1. **Questions Channel ($Q$)**:
   - `Q_DAYS_REMAINING`: $q_{\text{days}} = \max\left(0.0, \frac{30.0 - \text{day}}{30.0}\right)$ ($\sigma \approx 0.298$).
2. **K-Map Observer Channel ($K$)**:
   - `K_OPP_WALLET_BALANCE`: $k_{\text{opp}} = \frac{\ln(1.0 + \text{opp\_money})}{12.0}$ ($\sigma \approx 0.084$).
3. **Decisions Channel ($D$)**:
   - `D_SUBAGENTS_WALLET_BALANCE`: $d_{\text{sub}} = \left(\frac{\ln(1.0 + \text{my\_money})}{12.0}\right) \times (1.0 + 0.1 \times \text{hands})$ ($\sigma \approx 0.187$).

### 2. Floating-Point Convergence 2D K-Map Architecture:
- **Proportional Dimension Scaling**:
  - **Rows (Spatial Grid)**: $H \times W = 10 \times 10 = 100$ farm tiles.
  - **Columns (Adversarial & Market Context)**: $11\text{ Opponents} \times 9\text{ Commodities} = 99$ channels.
  - **Grid Shape**: $(100, 99) = 9,900$ continuous floating-point decision cells.
- **Continuous Float32 Storage Format (No Premature Quantization)**:
  - Stored as native `float32` probability matrix $P(x) \in [0.0, 1.0]$ requiring **$39,600$ bytes** ($38.7\text{ KB}$).
  - Preserves fractional decision confidence, continuous gradients, and epistemic uncertainty.
  - Soft continuous fuzzy logic operations ($\min$, $\max$, $1 - P$).
- **Real-Time Asymptotic Convergence Tracking**:
  - **Polarization Ratio**: $\frac{1}{N} \sum |2P - 1| \in [0.0, 1.0]$ (monitors asymptotic convergence toward $\{0, 1\}$).
  - **Shannon Information Entropy**: $H(P) = -\frac{1}{N} \sum [P \log_2(P) + (1-P)\log_2(1-P)]$ in bits.
  - **Epistemic Uncertainty Variance**: $\frac{1}{N} \sum P(1 - P)$.
  - **Continuous Learning Dynamics**: Continuous EMA updates, temperature annealing ($\tau \to \tau_{\min}$), and Bayesian log-odds accumulation.
- **2-Dimensional Logarithmic Heatmap Output ($10^{-4} \to 10^0$)**:
  - Training exits generate `kmap_logarithmic_heatmap.png` using `LogNorm(vmin=1e-4, vmax=1.0)` over `inferno` colormap, revealing continuous background interaction fields and active minterms across 4 decades ($10,000 : 1$ ratio).
  - Also outputs linear continuous probability heatmap (`kmap_polarization_heatmap.png`).

### 3. $10 \times 10 \times 10$ Voxel Extractor & GBDT Capacity Scaling:
- **3D Voxel Tensor**: Shape $(10, 10, 10) = 1,000$ spatial elements.
- **Full State Vector**: $1,000 \text{ voxels} + 3 \text{ QKD probes} + 32 \text{ economic features} = 1,035$ features.
- **Multi-Head GBDT Policy**: Action Classifier ($10$ classes) + Spatial Tile Ranker ($100$ tiles) + 9-Commodity Liquidation Regressor + 120-Day Terminal Value Regressor.
- **Iterative Scaling Loop**: Expands tree estimators until serialized file size reaches ~**98.5–100.0 MB** (strictly $\le 100$ MB Kaggle payload limit).

---

## 2. Standardized Workflow Execution

### Step 1: Initialize Configuration & Floating-Point Convergence K-Map
```python
from market_config_suite import InitialTerminalConfiguration, build_market_functions, build_opponent_functions
from proportional_kmap import ProportionalKMap, FloatingPointConvergenceKMap, build_proportional_kmap_mask
from vector_memory_bank import StateVectorEncoder, save_kmap_heatmap

# 120-Day Configuration
config = InitialTerminalConfiguration(number_of_days=120, amount_of_money=3000.0)

# Compute Continuous Float32 Convergence K-Map (100 x 99)
encoder = StateVectorEncoder()
proportional_kmap = encoder.compute_proportional_kmap(obs, spatial_rows=100, env_cols=99)

# Real-time convergence metrics
print(f"Polarization: {proportional_kmap.polarization_ratio * 100.0:.1f}%")
print(f"Entropy:      {proportional_kmap.entropy:.3f} bits")
print(f"Variance:     {proportional_kmap.epistemic_variance:.4f}")

# Render 2-Dimensional Logarithmic Heatmap on Exit
log_heatmap_path = proportional_kmap.render_logarithmic_heatmap("kmap_logarithmic_heatmap.png")
# Render Linear Probability Heatmap
linear_heatmap_path = proportional_kmap.render_heatmap("kmap_polarization_heatmap.png")
```

### Step 2: Extract 10x10x10 Voxel State Vector (1,035 Dims)
```python
from voxel_state_extractor import VoxelStateExtractor

extractor = VoxelStateExtractor(grid_height=10, grid_width=10, depth_channels=10)
market_fns = build_market_functions(obs, config)
opp_fns = build_opponent_functions(obs, config)

# Extract full 1035-dim vector
state_vec = extractor.extract_full_state_vector(obs, config, market_fns, opp_fns)
assert state_vec.shape == (1035,)
```

### Step 3: Two-Stage Attention & 100MB GBDT Policy Execution
```python
from agents.qkd_gbdt_agent import QKDGBDTAgent

agent = QKDGBDTAgent(model_path="models/qkd_gbdt_120day_100mb.pkl")

# Stage 1: Market-optimal focus
stage1_action = agent.Att(obs, config, market_fns)

# Stage 2: Opponent-adjusted synthesis
stage2_action = agent.Att(obs, config, market_fns, opp_fns)
```

### Step 4: Iterative 120-Day Scaling Training to 100 MB
```python
from gbdt_120day_scaling_trainer import GBDT120DayScalingTrainer

trainer = GBDT120DayScalingTrainer(total_days=120, target_model_mb=99.0)
X, y_act, y_tile, y_liq, y_val = trainer.harvest_120day_trajectories()
results = trainer.train_and_scale_gbdt_to_100mb(X, y_act, y_tile, y_liq, y_val, target_mb=99.0)

print(f"Final Model Payload: {results['final_model_size_mb']:.2f} MB (Status: SAFE)")
```
