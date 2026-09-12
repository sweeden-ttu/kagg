# 2D K-Map & QKD Tripartite Vector Architecture Specification

## 1. 2-Dimensional Karnaugh Map (K-Map) Matrix
- **Axis 0 (Rows)**: $S_{\text{self}} \in \mathbb{R}^{128}$ (Self farm state: bank liquidity, worker count, private seed and shed reserves, farm tile layout, calendar harmonics).
- **Axis 1 (Columns)**: $S_{\text{opp}} \in \mathbb{R}^{128}$ (Opponent observation: bank coins, relative coin margin $\Delta_{\text{coins}}$, opponent hiring scale, unlocked land quadrants, shared market spot prices & inventory, public tile crop stages).
- **2D Boolean Mask Matrix** $M_{2D} \in \{0, 1\}^{128 \times 128}$:
  - Value is $1.0$ for decisive cross-dependencies (e.g., self shed holdings $\times$ market price spikes; self seed reserves $\times$ opponent crop choices; farm stage $\times$ opponent land expansion).
  - Value is $0.0$ for "Don't-Care" cross-cells (uncoupled coordinates and private noise).
- **Bilinear Outer-Product Tensor & Dual-Marginal Projection**:
  $$\mathbf{I} = (S_{\text{self}}[:, \text{None}] \cdot S_{\text{opp}}[\text{None}, :]) \odot M_{2D} \in \mathbb{R}^{128 \times 128}$$
  $$\mathbf{z} = \left[ \sum_{j} \mathbf{I}_{i, j}, \; \sum_{i} \mathbf{I}_{i, j} \right] \in \mathbb{R}^{256} \quad (\text{L2-normalized for sub-millisecond retrieval})$$

## 2. QKD Tripartite Vector Stores
The architecture partitions memory and policy into three specialized INT8 quantized vector stores:
- **$Q$ (Questions Vector Store)**: Encodes probabilistic inquiries, liquidity confidence, hypothesis testing, and risk metrics.
- **$K$ (2D K-Map Observations Store)**: Encodes bilinear 2D K-map opponent behavioral archetypes and counter-tactics.
- **$D$ (Decisions Vector Store)**: Encodes state-action-reward transitions, workforce assignments, and market order queues.

## 3. Mathematical Decision Product
Every runtime action is synthesized as a joint product across the cognitive pillars:
$$\text{Decision}(a) = \mathcal{P}_{\text{Reasoning}}(a \mid \text{obs}) \cdot \mathcal{P}_{\text{Observer}}(a \mid \text{obs})^{w_O} \cdot \mathcal{P}_{\text{Questioning}}(a \mid \text{obs})^{w_Q} \cdot \mathcal{M}_{\text{legal}}(a)$$

## 4. Hard Limit Invariants
- Combined serialized payload ($Q + K + D$) must remain $\le 90.0\text{ MB}$ (10 MB buffer under the 100 MB hard ceiling).
- Joint CPU retrieval latency must remain $< 1.0\text{ ms}$ (target $< 5.0\text{ ms}$).
- Turn compute budget must not exceed $\le 42$ FLOPs.
