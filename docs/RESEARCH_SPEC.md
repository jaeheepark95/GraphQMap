# GraphQMap — Complete Implementation Specification

> **Purpose of this document:** This is a complete, self-contained specification for implementing GraphQMap. Every design decision described here has been confirmed and finalized. Use this document as the authoritative reference for all implementation work.

---

## Table of Contents

1. [Research Objective & Problem Definition](#1-research-objective--problem-definition)
2. [Core Pipeline](#2-core-pipeline)
3. [Graph Representation](#3-graph-representation)
4. [Model Architecture](#4-model-architecture)
5. [Training Strategy: 2-Stage Curriculum](#5-training-strategy-2-stage-curriculum)
6. [Dataset Usage](#6-dataset-usage)
7. [Multi-Programming Training Data](#7-multi-programming-training-data)
8. [Batching Strategy](#8-batching-strategy)
9. [Evaluation Protocol](#9-evaluation-protocol)
10. [Ablation Study Candidates](#10-ablation-study-candidates)
11. [Open Items](#11-open-items)
12. [Appendix A: Key Implementation Notes](#appendix-a-key-implementation-notes)
13. [Appendix B: Complete Hyperparameter Summary](#appendix-b-complete-hyperparameter-summary)

---

## 1. Research Objective & Problem Definition

### What We Are Building

**GraphQMap** is a single ML model that outputs initial qubit layouts (mappings from logical qubits to physical qubits) for quantum circuit compilation. The model must:

- Work across **multiple quantum hardware backends** without hardware-specific fine-tuning (hardware-agnostic)
- Handle **multi-programming** scenarios: compiling an arbitrary number of quantum circuits simultaneously onto a single backend (constrained only by total logical qubits ≤ physical qubits)
- Achieve high **PST (Probability of Successful Trials)** on NISQ hardware
- Run with **fast inference speed**

### What the Model Does NOT Do

GraphQMap outputs **only the initial layout**. All subsequent compilation stages — routing, optimization, scheduling — are handled by the transpiler (via custom PassManager). The model does not perform routing or scheduling; routing method is configurable at evaluation time (e.g., SABRE, NASSC). The unsupervised training (Stage 2) uses surrogate losses that evaluate layout quality **without routing**, making the model routing-agnostic by design.

### The Non-Differentiability Problem

The pipeline has a non-differentiable barrier:

```
GraphQMap → Initial Layout (A) → [Transpile (routing + optimization) — NON-DIFFERENTIABLE] → Transpiled Circuit (B) → PST Evaluation
```

The model cannot receive gradients through routing. This constraint shapes the entire training strategy — Stage 2's surrogate losses bypass routing entirely, evaluating layout quality directly from the circuit-hardware graph structure.

### Key Environment

- **Hardware:** Qiskit FakeBackendV2
- **Native 2-Qubit Gates:** `cx`, `ecr`, or `cz` (varies by backend; model abstracts over gate type via error/duration features)
- **Circuit Format:** OpenQASM (`.qasm`)
- **Target Metric:** PST (Probability of Successful Trials) maximization

---

## 2. Core Pipeline

### Inference Pipeline (Complete)

```
Input: Quantum circuit (.qasm) + Target hardware (FakeBackendV2)
  ↓
[Circuit Graph Construction] → Circuit graph with node/edge features
[Hardware Graph Construction] → Hardware graph with node/edge features (including noise)
  ↓
[Circuit GNN Encoder] → C (l×d) matrix, l = logical qubits, d = 64
[Hardware GNN Encoder] → H (h×d) matrix, h = physical qubits, d = 64
  ↓
[Cross-Attention Module] → C' (l×d), H' (h×d)  (2 layers of bidirectional cross-attention)
  ↓
[Score Head] → S (l×h) = (C'·W_q) × (H'·W_k)ᵀ / √d_k
  ↓
[Row-wise Softmax(S/τ)] → P (l×h) row-stochastic matrix
  ↓
[Hungarian Algorithm] → Discrete one-to-one mapping
  ↓
Output: Initial layout (logical qubit → physical qubit mapping)
  ↓
[Transpile(initial_layout=output, routing_method=configurable)] → Transpiled circuit
  ↓
[Noise simulation (AerSimulator)] → PST measurement
```

### Training Pipeline

Same as above except:
- **No Hungarian algorithm** during training — P matrix is used directly for loss computation
- **Softmax temperature τ** is annealed during both Stage 1 and Stage 2

### Multi-Programming Pipeline

For multiple circuits (e.g., C₁ with l₁ qubits, C₂ with l₂ qubits):
1. Merge circuit graphs into a single **disconnected graph** (no edges between circuits) — node features stay at **4 dimensions** (same as single-circuit)
2. Circuit GNN output becomes (l₁+l₂)×d
3. Rest of pipeline is identical — softmax row-stochastic output combined with Hungarian algorithm prevents mapping conflicts
4. **No architectural modifications required** — single-circuit is simply a special case of multi-programming with one circuit

---

## 3. Graph Representation

### 3.1 Circuit Graph

**Node:** Each logical qubit = one node.

**Edge:** Undirected. An edge exists between logical qubits i and j if any 2-qubit gate connects them. Multiple gates on the same pair are merged into a single edge.

**Node Features (per logical qubit) — configurable via YAML:**

Node features are selected from a registry in `data/circuit_graph.py`. The YAML config specifies which features to use and in what order. All features are computed during preprocessing; selection happens at load time.

| Feature | Description | Default |
|---------|-------------|---------|
| `gate_count` | Total number of gates on this qubit (includes both 1Q and 2Q gates) | ✓ |
| `two_qubit_gate_count` | Number of 2-qubit gates involving this qubit (symmetric) | ✓ |
| `degree` | Number of distinct qubits this qubit interacts with via 2-qubit gates | |
| `depth_participation` | Fraction of circuit depth in which this qubit is active (not idle) | |
| `weighted_degree` | Sum of interaction counts across all connected edges | |
| `single_qubit_gate_ratio` | (gate_count - 2q_count) / gate_count | ✓ |
| `critical_path_fraction` | Fraction of DAG critical path length involving this qubit | ✓ |
| `interaction_entropy` | −Σ p_ij log p_ij over neighbor interaction distribution | |

**Positional Encoding (optional, appended after node features):**

| Feature | Description | Config |
|---------|-------------|--------|
| `rwpe` | Random Walk Positional Encoding: k-step self-return probabilities | `rwpe_k: 2` |

RWPE computes k-step self-return probabilities from the random walk transition matrix M (row-normalized adjacency). With `start_step=2` (default), outputs `[M^2_ii, M^3_ii, ..., M^(k+1)_ii]`, skipping step 1 which is structurally zero for graphs without self-loops. This gives each node a structural fingerprint based on its local topology, helping the GNN distinguish structurally similar nodes. RWPE is **not z-score normalized** (values are probabilities in [0, 1]).

**RWPE k selection:** k=2 with start_step=2 outputs `[M^2, M^3]`. Both dimensions are non-trivial. Verified via `scripts/analyze_circuit_features.py` Phase 6:
- k=2: eff_dim +0.99 contribution, 0 fully-dead circuits
- k=3: eff_dim +1.22, but RWPE[2]↔2qc r=0.768 (redundant), 1.61/3 dead dims
- k=4: eff_dim +1.44, diminishing returns, 1.69/4 dead dims

**k=2 is optimal** — k=3+ adds marginal eff_dim at the cost of redundancy with existing features.

| k | Total dim | Eff. dim | Dead cols | RWPE eff_dim contribution |
|---|-----------|----------|-----------|--------------------------|
| 0 | 4 | 2.70 | — | — |
| 2 | 6 | 3.69 | 1.08 | +0.99 |
| 3 | 7 | 3.93 | 1.61 | +1.22 |
| 4 | 8 | 4.14 | 1.69 | +1.44 |

**YAML config format:**
```yaml
model:
  circuit_gnn:
    node_features:        # select from registry
      - gate_count
      - two_qubit_gate_count
      - single_qubit_gate_ratio
      - critical_path_fraction
    rwpe_k: 2             # append 2-step RWPE
    # node_input_dim is auto-computed: len(node_features) + rwpe_k = 6
```

**Feature diagnostics:** Run `python scripts/diagnose_features.py --config configs/stage1.yaml` to measure effective dimensionality, cosine similarity, and column correlations before training.

**Feature analysis findings (from `scripts/analyze_circuit_features.py`, 520 circuits, 2026-04-06):**

*Overall metrics:*
- Current default (gc, 2qc, sqr, cpf + RWPE2): EffDim 3.68/6, Indist 16.73%, max |r| = 0.753 (gc↔2qc)
- Normalization: all z-score confirmed better than mixed (mixed indist +7.03pp worse)
- RWPE k=2 confirmed optimal post-filtering (k=3 marginal, redundant with 2qc)

*Per-feature issues:*
- **`sqr`**: 37.7% all-zero (MLQD 2Q-only circuits), CoV<0.1 in 53.8%, medium(6-10Q) 54% constant
- **`cpf`**: median 0.956 (saturated near 1.0), xlarge(21Q+) 67% constant, CoV<0.1 in 32.5%
- `gc`, `2qc`: CoV healthy (< 0.1 in 8-17%), constant rate low (1-5%)

*Size-dependent:*
| Size | N | EffDim | Indist% | Key issue |
|------|---|--------|---------|-----------|
| tiny (2-3Q) | 23 | 2.57 | 20.29% | dim > qubits |
| small (4-5Q) | 121 | 3.66 | 8.93% | best bucket |
| medium (6-10Q) | 172 | 3.83 | 10.55% | sqr 54% constant |
| large (11-20Q) | 122 | 3.88 | 16.20% | interaction_count 38% const |
| xlarge (21Q+) | 82 | 3.44 | 41.03% | cpf 67% const, gc CoV 0.069 |

*Previous findings (still valid):*
- MQT parametric circuits (VQE/QNN/GHZ): 40-100% indist → removed by filtering
- After filtering (5,769 circuits): worst-case indist reduced from >30% to manageable levels

**Known feature degeneracy issues (motivating the registry):**
- `gate_count` ↔ `depth_participation`: |r| = 1.000 in 100% of circuits. Redundant.
- `weighted_degree` = `two_qubit_gate_count`: |r| = 1.000 in 100% of circuits. Zero independent information.
- `degree`: GNN learns topology via message passing + 2qc와 |r| > 0.9 in 64.5% of circuits. Excluded.
- `interaction_entropy`: degree와 |r| = 0.976 (95.3% > 0.9). H ≈ log(degree), redundant. Excluded.
- RWPE: step 1 structurally zero (no self-loops). Fixed via `start_step=2` — no dead dims.

**Node feature design notes:**
- 2-qubit gate counts are applied **symmetrically** to both control and target qubits. For initial layout, the interaction frequency (not the control/target role) is what drives placement decisions — both qubits experience the same decoherence during the gate.
- `weighted_degree` complements `degree`: degree counts unique neighbors, weighted_degree sums interaction counts. A qubit with 2 neighbors but 50 interactions is very different from one with 2 neighbors and 2 interactions.
- `critical_path_fraction` identifies bottleneck qubits that constrain circuit depth — these should be placed on high-quality physical qubits.

**Edge Features (per qubit pair) — current 5-dim:**

| Feature | Description |
|---------|-------------|
| `interaction_count` | Number of 2-qubit gates between this qubit pair |
| `earliest_interaction` | Normalized time (0~1) of the first 2-qubit gate between this pair |
| `latest_interaction` | Normalized time (0~1) of the last 2-qubit gate between this pair |
| `interaction_span` | `latest - earliest` — temporal duration of interaction between this pair |
| `interaction_density` | `count / (span + eps)` — burstiness of interaction (high = concentrated in time) |

Edge dim is configurable via `circuit_gnn.edge_input_dim` in YAML. Previous versions used 3-dim (count, earliest, latest).

**Edge feature design notes:**
- `interaction_count` is the primary feature: pairs with high counts must be placed close on the hardware to minimize SWAP overhead.
- `earliest_interaction` and `latest_interaction` encode when interactions occur in circuit execution.
- `interaction_span` captures the temporal duration of the interaction relationship — long span means the pair interacts throughout the circuit.
- `interaction_density` captures burstiness — high density means many gates concentrated in a short time window, which is harder to route around.

**Multi-Programming Note:**

Multi-programming uses the same 4 node features as single-circuit. Circuit graphs are merged into a disconnected graph without additional features. The GNN naturally distinguishes circuits through disconnected components, and Hungarian decoding prevents mapping conflicts. No per-circuit global summary features are used — this keeps single-circuit and multi-programming unified under the same input dimensions.

**Circuit Node Feature Normalization:** Z-score normalized **within each circuit** (across qubits, dim=0). This captures *relative* qubit importance within the circuit — which qubit is busier than others — rather than absolute counts. This is intentional: hardware features are also normalized within-backend, so the model learns to match "relatively busy logical qubit → relatively good physical qubit" in a consistent scale.

**Circuit Edge Feature Normalization:** Edge features (all 5 dimensions) are **z-score normalized within each circuit** (across edges, dim=0), same as node features. This ensures all edge features contribute equally to GNN message passing regardless of their original scales (`interaction_count` can be any positive integer while `earliest`/`latest` are in [0,1], and `density` can range widely).

**Trade-off:** Within-circuit normalization loses absolute circuit complexity. A 3-gate circuit and a 300-gate circuit look similar in both node and edge features if qubit interaction ratios are the same. However, the GNN learns to use the relative structure (which pairs interact more than others) for placement decisions, which is the primary signal for initial layout quality.

**Exception handling:** If standard deviation = 0 (all qubits identical), add ε = 1e-8 to denominator — node features become all-zero, which is uninformative but acceptable (uniform circuits have no strong placement preference).

**Multi-programming edge renormalization:** In multi-programming, each circuit's edge features are initially z-score normalized per-circuit. When merging multiple circuits, edge features are **re-normalized at the group level** (across all edges in all circuits) via `renormalize_group_edges()`. This ensures consistent edge feature scales across circuits in the group while node features remain per-circuit normalized.

**Known limitation — multi-programming cross-circuit node scale mismatch:** In multi-programming, node features remain per-circuit normalized (not group-level). This means a qubit that is "relatively the busiest" in a simple 3-gate circuit and one that is "relatively the busiest" in a complex 300-gate circuit will have similar node feature values after normalization, even though the latter should receive higher-priority physical placement. The Score Head (cross-attention) cannot distinguish their true absolute importance. Possible mitigations to consider if this proves impactful in practice:
- **Group-level node normalization**: normalize node features jointly across all circuits in a multi-programming group. Consistent scale across circuits, but breaks the unified single/multi pipeline.
- **Log transform**: apply log(1+x) before z-score to compress scale differences while retaining some absolute information.
- **Global summary re-introduction**: add circuit-level complexity features as a context signal (reverts the 4-dim unification decision).

### 3.2 Hardware Graph

**Node:** Each physical qubit = one node.

**Edge:** Undirected. Edges from FakeBackendV2 coupling map.

**Node Features (per physical qubit) — current 6-dim (5 z-scored + 1 raw):**

| Feature | Description | Normalization | Direction |
|---------|-------------|---------------|-----------|
| `readout_error` | Measurement error rate | z-score | Lower = better |
| `single_qubit_error` | Average single-qubit gate error rate (sx, x) | z-score | Lower = better |
| `degree` | Coupling map connectivity degree | z-score | Structural (optional: `exclude_degree: true`) |
| `t1_cx_ratio` | T1 / mean_cx_duration — 2Q gates fitting within T1 | z-score | Higher = better |
| `t2_cx_ratio` | T2 / mean_cx_duration — 2Q gates fitting within T2 | z-score | Higher = better |
| `t2_t1_ratio` | T2 / T1, clipped to [0, 2] — decoherence type indicator | **raw** | ≈2 relaxation-limited, ≈1 dephasing-dominated |

`t1_cx_ratio` and `t2_cx_ratio` are computed per qubit as T1 (or T2) divided by the mean cx_duration across all edges connected to that qubit. `t2_t1_ratio` is a dimensionless ratio with inherent physical meaning — z-score normalization would destroy this.

> **Feature selection analysis (2026-04-06):** T2/T1 ratio confirmed independent from all existing features (max |r| = 0.243 pooled across 60 backends). T1_raw and T2_raw rejected as redundant (r = 0.88–0.95 with ratio features under z-score). Measurement duration unavailable (ALL ZERO in FakeBackendV2). Frequency rejected (inconsistent across backends, crosstalk not modeled).

**Edge Features (per physical connection) — current 2-dim (1 z-scored + 1 raw):**

| Feature | Description | Normalization |
|---------|-------------|---------------|
| `2q_gate_error` | 2-qubit gate error rate, averaged over both directions | z-score |
| `edge_coherence_ratio` | `cx_duration / min(T1_u, T1_v, T2_u, T2_v)` — coherence budget consumed per gate | **raw** |

`edge_coherence_ratio` captures how much of the weakest endpoint's coherence budget a single 2Q gate operation consumes. Confirmed independent from `2q_gate_error` (r = 0.059). Edge error asymmetry analysis (2026-04-06) confirmed 99.3% of edges have symmetric error (<1% diff); ~10% duration asymmetry exists but is averaged for undirected representation.

**Hardware Feature Normalization (mixed strategy):**
- **Z-score within backend:** Error rates, coherence ratios, degree — scale varies significantly across backends (e.g., T1 can differ 10× between backends). Z-score preserves relative ordering.
- **Raw (not z-scored):** Dimensionless ratios with physical meaning (`t2_t1_ratio`, `edge_coherence_ratio`). Their absolute values carry information (e.g., T2/T1 ≈ 2.0 means relaxation-limited regardless of backend). Z-score would destroy this.

**Exception handling:** ε = 1e-8 for zero standard deviation. If a qubit has no connected edges (isolated), mean_cx_duration defaults to 1.0 to avoid division by zero in ratio features. T2/T1 ratio clipped to [0, 2] per theoretical bound T2 ≤ 2T1; 2.5% of qubits (mostly synthetic backends) hit the clip boundary.

---

## 4. Model Architecture

### 4.1 Overview

```
[Circuit Graph] → Circuit GNN (3-layer GATv2) → C (l×d)
[Hardware Graph] → Hardware GNN (3-layer GATv2) → H (h×d)

Cross-Attention Layer 1:
  C = LayerNorm(C + CrossAttn(Q=C, K=H, V=H))
  C = LayerNorm(C + FFN(C))
  H = LayerNorm(H + CrossAttn(Q=H, K=C, V=C))
  H = LayerNorm(H + FFN(H))

Cross-Attention Layer 2:
  C' = LayerNorm(C + CrossAttn(Q=C, K=H, V=H))
  C' = LayerNorm(C' + FFN(C'))
  H' = LayerNorm(H + CrossAttn(Q=H, K=C, V=C))
  H' = LayerNorm(H' + FFN(H'))

Score Head: S = (C'·W_q) × (H'·W_k)ᵀ / √d_k  → l×h
Row-wise Softmax(S/τ) → P (l×h) row-stochastic

[Training] P used directly for loss
[Inference] Hungarian(P) → discrete mapping
```

### 4.2 Dual GNN Encoder (GATv2)

Two independent GNN networks (no shared parameters) using the same architectural template.

**Architecture per GNN:**

```
Input Features → Linear(input_dim, d) → d-dimensional projection
→ GATv2 Layer 1 → BatchNorm → ELU → Residual Connection
→ GATv2 Layer 2 → BatchNorm → ELU → Residual Connection
→ GATv2 Layer 3 → BatchNorm → ELU → Residual Connection
→ Linear(d, d) → Final node embedding
```

**Edge Feature Integration in GATv2:**

Edge features are concatenated into attention score computation:

```
e_ij = LeakyReLU(a^T · [W·h_i || W·h_j || W_e·edge_feat_ij])
α_ij = softmax_j(e_ij)
h_i' = Σ_j α_ij · V·h_j
```

`W_e` is a separate learnable matrix that projects edge features into the attention space.

**Residual Connection:**

```python
h_out = GATv2Layer(h_in) + h_in  # Applied on layers 2, 3
```

**GNN Hyperparameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Embedding dimension (d) | 64 | Shared by both GNNs and cross-attention |
| GATv2 Layers | 3 | 3-hop information aggregation |
| Attention Heads | 4 | 16 dims per head |
| Activation | ELU | |
| Dropout | 0.1 | |
| BatchNorm | Per layer | |
| Residual Connections | Layers 2, 3 | |

**Input dimensions:**
- Circuit GNN input: `len(node_features) + rwpe_k` (current default: 4 + 2 = 6: gc, 2qc, sqr, cpf + RWPE2)
- Hardware GNN input: 6 (5 z-scored: readout_err, sq_err, degree, t1_cx_ratio, t2_cx_ratio + 1 raw: t2_t1_ratio)
- Circuit edge features: 5 (interaction_count, earliest, latest, span, density)
- Hardware edge features: 2 (1 z-scored: 2q_error + 1 raw: edge_coherence_ratio)

### 4.3 Cross-Attention Interaction Module

Inserted between GNN encoding and Score Head. Enables circuit and hardware embeddings to mutually reference each other.

**Purpose:** Simple dot-product between GNN outputs only captures similarity. Good mapping requires capturing **complementary** relationships (busy logical qubit → high-quality physical qubit). Cross-attention enables this.

**Structure (repeated 2 times):**

```python
# Layer n (n = 1, 2):
C = LayerNorm(C + MultiHeadCrossAttention(Q=C, K=H, V=H))
C = LayerNorm(C + FFN(C))

H = LayerNorm(H + MultiHeadCrossAttention(Q=H, K=C, V=C))
H = LayerNorm(H + FFN(H))
```

After 2 layers, output is C' (l×d) and H' (h×d).

**Cross-Attention Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Cross-Attention Layers | 2 |
| Attention Heads | 4 |
| FFN Hidden Dim | 128 (2×d) |
| Dropout | 0.1 |

### 4.4 Score Head

Learned projection from cross-attention output to mapping compatibility scores, with noise-aware bias:

```
S_ij = (C'_i · W_q)^T · (H'_j · W_k) / √d_k + bias_j
bias_j = Linear(hw_node_features_j)
```

- `W_q`: learnable matrix d → d_k (d_k = 64)
- `W_k`: learnable matrix d → d_k (d_k = 64)
- Scaled by `√d_k` for numerical stability before softmax
- `noise_bias_dim`: configurable (default 0, disabled). When enabled, adds per-physical-qubit bias from hardware features. Disabled by default to avoid gradient conflict with Hardware GNN (same features flowing through two paths to score matrix).

**Output:** S matrix of shape (l × h)

### 4.5 Score Normalization (Configurable)

Two normalization modes, selectable via `sinkhorn.score_norm` in YAML:

**SoftmaxNorm (`score_norm: softmax`, current default):**
Row-wise softmax → P (batch, l, h) row-stochastic. No dummy padding.
```python
P = F.softmax(S / tau, dim=-1)  # (batch, l, h)
```

**SinkhornLayer (`score_norm: sinkhorn`):**
Log-domain Sinkhorn with dummy padding l×h → h×h → doubly stochastic P.
- Pads h-l dummy zero rows to make matrix square
- Alternates row/column log-normalization for `max_iter` iterations
- Produces doubly stochastic P (rows AND columns sum to 1)
- Output sliced to top l rows: P[:, :l, :] for loss and Hungarian
- Log-domain prevents overflow at low τ

**Sinkhorn vs Softmax trade-offs:**
- Sinkhorn enforces one-to-one exclusivity (column sums ≈ 1), which may produce better discrete mappings
- Softmax is simpler and avoids dummy padding instability when l << h
- Historical best result (PST 0.3440) used Sinkhorn; later experiments switched to Softmax
- Both are available and selectable without code changes

**Temperature τ:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| τ_max (initial) | 1.0 | Soft distribution at training start |
| τ_min (final) | 0.05 | Near one-hot; minimizes train-inference gap |
| τ Schedule | Exponential decay | `τ(epoch) = τ_max · (τ_min/τ_max)^(epoch/total_epochs)` |

### 4.7 Hungarian Algorithm (Inference Only)

Applied to P matrix to produce discrete mapping. Only the top l rows (actual logical qubits) of the result are used. Dummy row assignments are discarded.

Use `scipy.optimize.linear_sum_assignment` with cost matrix `(1 - P)` (since Hungarian minimizes cost, and we want to maximize assignment probability).

---

## 5. Training Strategy: 2-Stage Curriculum

> **Note:** RL fine-tuning has been excluded from scope. The training consists of 2 stages only.

### 5.1 Stage 1: Supervised Pre-training

> **Status: Under investigation.** Early experiments suggest Stage 1 pretraining may not improve final PST over Stage 2 from scratch. See "Experimental Findings" below.

**Objective:** Learn the basic sense of what constitutes a good qubit mapping.

**Training Order:** MLQD + QUEKO (large quantity, existing labels) → QUEKO only (high quality true optimal, fine-tuning)

**Optimizer:** AdamW with cosine annealing LR scheduler.

**Loss Function: Cross-Entropy**

Ground truth layout π is converted to binary permutation matrix Y (h×h):
- Y[i, π(i)] = 1 for each logical qubit i
- All other entries = 0
- Dummy rows in Y are set to complete the permutation (each assigned to a remaining physical qubit)

```
L_sup = -Σ_i Σ_j Y_ij · log(P_ij)
```

**Softmax τ:** Annealed from τ_max=1.0 to τ_min=0.05 via exponential decay.

**MLQD + QUEKO → QUEKO-only Transition:**
- Criterion: Validation cross-entropy loss early stopping (patience = 15 epochs main, 10 epochs QUEKO)
- When switching to QUEKO: reduce learning rate to 1/10 of previous LR (lr_factor=0.1)
- QUEKO termination: same validation CE loss early stopping (patience = 10)

### 5.2 Stage 2: Noise-Aware Surrogate Metric Fine-tuning

**Objective:** Fine-tune toward actual NISQ PST correlation without running the full transpile pipeline.

**Data:** All available circuits including QASMBench and RevLib (no labels required). Labeled datasets (MQT Bench, MLQD, QUEKO) can also be included for surrogate fine-tuning.

**Softmax τ:** Fixed at 0.05.

**Optimizer:** AdamW with cosine annealing. Warm-up 5 epochs.

#### Loss Component Registry

Stage 2 uses a **modular loss registry** (`@register_loss()` in `training/losses.py`). Components are configured declaratively in YAML — no code changes needed to switch loss combinations. Each component receives P and all available kwargs (d_error, d_hw, hw_node_features, etc.).

```yaml
# configs/stage2_sinkhorn_adj.yaml (current best)
loss:
  type: surrogate
  components:
    - name: error_distance    # select registered components
      weight: 1.0             # multiplier in total loss
    - name: adjacency
      weight: 0.3
    # Components with constructor params use 'params' dict:
    # - name: soft_proximity
    #   weight: 0.3
    #   params:
    #     alpha: 2.0
```

**Available components:**

#### error_distance — L_surr: Error-Aware Edge Quality Loss (Default Primary)

Uses precomputed error-accumulated shortest path distances (Floyd-Warshall on 2Q gate error rates).

```
L_surr = (1/W) · Σ_{(i,j)∈E_circuit} f_ij · Σ_{p,q} P_ip · P_jq · d_error(p,q)
```

Where f_ij = 2Q gate count for pair (i,j), W = Σ f_ij. d_error(p,q) = error-weighted shortest path between physical qubits p and q. Gate-frequency weighting ensures frequently-interacting pairs contribute proportionally more. Fully differentiable w.r.t. P. Bounded in [0, ∞).

**Known limitations:** (1) Uses additive error sums via Floyd-Warshall, but quantum fidelity is multiplicative: F = Π(1-ε). Correct edge weight would be -log(1-ε). (2) Does not model SWAP 3× CX overhead: a hop=2 path costs ~4× a direct CX (1 SWAP + 1 CX = 4 CX), but d_error only sums 2 edge errors. (3) Raw error values (0.01-0.05) produce weak gradient signals → saturates by epoch 3. See `swap_count` for a physics-corrected alternative.

#### adjacency — L_adj: Adjacency Matching Loss (Gate-Frequency Weighted)

Directly measures whether interacting logical qubits are mapped to adjacent physical qubits.

```
A_hw(p, q) = 1 if (p, q) ∈ coupling_map, else 0
L_adj = -(1/W) · Σ_{(i,j)∈E_circuit} w_ij · Σ_{p,q} P_ip · P_jq · A_hw(p,q)
```

Where w_ij = number of 2-qubit gates on edge (i,j), W = Σ w_ij. Output bounded in [-1, 0].

**Known limitations:** (1) Binary signal — zero gradient for all non-adjacent pairs, regardless of distance (hop=2 and hop=10 contribute identically). (2) Coupling density is low on large backends (27Q: ~9%, 127Q: ~1.7%), so >98% of A_hw is 0 → extremely sparse gradient. (3) Ignores edge error rate differences — all adjacent pairs treated equally. See `soft_proximity` for a smooth alternative.

#### hop_distance — L_hop: Hop Distance Tiebreaker

Continuous distance signal for non-adjacent placements. Differentiates distance-2 from distance-10.

```
L_hop = (1/|E_circuit|) · Σ_{(i,j)∈E_circuit} Σ_{p,q} P_ip · P_jq · d_hop_norm(p,q)
```

Where d_hop_norm = d_hw / max(d_hw), normalized to [0, 1].

#### swap_count — L_swap: SWAP Count Estimation Loss

Directly estimates the number of additional CX gates from SWAP operations. Based on the observation that each SWAP decomposes into 3 CX gates, and adjacent qubits need 0 SWAPs.

```
L_swap = (1/W) · Σ_{(i,j)∈E_circuit} f_ij · Σ_{p,q} P_ip · P_jq · d_swap(p,q)
```

Where d_swap(p,q) = 3 · max(d_hop(p,q) - 1, 0). Adjacent qubits: d_swap=0 (no SWAP needed). 2-hop: d_swap=3 (1 SWAP = 3 CX). 3-hop: d_swap=6. Gate-frequency weighted like error_distance. Bounded in [0, ∞).

**Design rationale:** Addresses key L_surr limitations: (1) L_surr uses additive error sums, but fidelity is multiplicative — a hop=2 path costs ~4× (not 2×) a direct CX due to SWAP overhead. (2) Raw error rates (0.01-0.05) produce weak gradient signals. L_swap directly models SWAP cost with large dynamic range (0-60+ for 127Q backends). The 3× factor converts SWAP count to CX count (each SWAP = 3 CX gates). Inspired by colleague's MQM approach using D = max(hop-1, 0).

#### soft_proximity — L_soft: Exponential Decay Proximity Loss

Smooth replacement for adjacency that provides non-zero gradient for non-adjacent qubits. Solves adjacency's key limitation: binary A_hw gives exactly 0 gradient for all non-adjacent pairs. On large backends, coupling density is very low (27Q: ~9%, 127Q: ~1.7%), so >98% of A_hw is 0 → extremely sparse gradient. L_adj also cannot distinguish hop=2 from hop=10 (both contribute 0).

```
reward(p,q) = exp(-α · max(d_hop(p,q) - 1, 0))
L_soft = -(1/W) · Σ_{(i,j)∈E_circuit} f_ij · Σ_{p,q} P_ip · P_jq · reward(p,q)
```

Where α controls exponential decay rate (default 2.0). Adjacent: reward=1.0, 2-hop: reward=exp(-α)≈0.135, 3-hop: reward=exp(-2α)≈0.018. α→∞ recovers binary adjacency, α→0 gives uniform reward. Gate-frequency weighted. Bounded in [-1, 0].

**Constructor param:** `alpha` (float, default 2.0), configurable via YAML `params` dict.

#### node_quality — L_node: NISQ Node Quality Loss (deprecated)

Drives important logical qubits to high-quality physical qubits via learnable MLP.

```python
q_score(p) = sigmoid(MLP(hw_features_p))    # MLP: Linear(5→16) → ELU → Linear(16→1)
```

```
L_node = -Σ_i w_norm(i) · Σ_p P_ip · q_score(p)
```

Where w_norm(i) = importance of qubit i normalized to sum to 1. Bounded in [-1, 0]. QualityScore MLP trained jointly with model.

**Known issue — deprecated:** Collapses to trivial solution (-1.0) by epoch 1-2, providing zero gradient thereafter. Root cause: qubit mapping is fundamentally an edge problem (SWAP routing), not a node problem. The MLP learns a single circuit-agnostic ranking (low noise → high score) which is trivially solvable. Additionally, readout/1Q errors (what L_node optimizes) contribute far less to PST than 2Q gate errors from routing: 1Q error is 10-100× smaller than 2Q error. Can also conflict with edge losses when high-quality qubits are topologically distant. **Not recommended for use.**

#### separation — L_sep: Multi-Programming Separation Loss

Encourages physical distance between qubits of different circuits. Bounded in [-1, 0]. Automatically 0 for single-circuit scenarios.

```
L_sep = -(1/|E_cross|) · Σ_{(i,j)∈cross-circuit} Σ_{p,q} P_ip · P_jq · d_hw_norm(p,q)
```

#### Combined Stage 2 Loss

```
L_2 = Σ_k weight_k · component_k(P, ...)
```

**Current best configuration (Val PST 0.3588):**

| Component | Weight | Rationale |
|-----------|--------|-----------|
| error_distance | 1.0 | Primary: error-weighted distance (saturates early but provides initial signal) |
| adjacency | 0.3 | Only non-saturating loss; binary signal provides gradient throughout training |

**Under investigation (Phase 1):** `swap_count` replacing error_distance, `soft_proximity` replacing adjacency. See experiment plan in CLAUDE.md.

**Loss gradient analysis:** Both error_distance and adjacency gradients have the same structure: ∂L/∂P_ip depends on neighbor mapping P_jq and hardware structure only — circuit qubit i's properties never appear. This means no per-node circuit signal exists in current losses.

**CRITICAL: All terms are per-pair/per-qubit normalized** (divided by pair count or qubit count). This ensures scales are comparable regardless of circuit/hardware size.

**Verification:** Log each term during training. No single term should dominate by >10×.

**Experimenting with loss combinations:** Modify YAML only. Components with constructor parameters use `params` dict. CLI override: `--override loss.components.0.weight=2.0`. Each run's `config.yaml` records exact configuration.

### 5.3 Stage Transition Criteria

| Transition | Criterion | Details |
|------------|-----------|---------|
| MLQD + QUEKO → QUEKO-only | Validation CE loss early stopping | Patience 10 epochs; LR reduced to 1/10 |
| Stage 1 → Stage 2 | Validation PST convergence | Measure actual PST every 5–10 epochs on 50–100 representative circuits (Hungarian → transpile → noise sim); stop when 3 consecutive measurements improve < 0.5% |
| Stage 2 termination | No early stopping; train full max_epochs | Best checkpoint selected by val PST; val surrogate loss monitored but not used for selection |

**Validation surrogate loss procedure (monitoring only):**
1. Compute surrogate loss on 396 held-out val circuits every epoch
2. Same loss components as training (e.g., error_distance + adjacency)
3. Logged to metrics CSV for analysis; NOT used for checkpoint selection (saturates by epoch 4-13)

**Validation PST measurement procedure (used for best checkpoint selection):**
1. Take P matrix from model
2. Apply Hungarian → discrete layout
3. Run `qiskit.transpile(initial_layout=layout, routing_method='sabre')`
4. Run noise simulation on FakeBackendV2
5. Compute PST
6. Measured every `pst_validation.interval` epochs; best PST checkpoint saved

**Checkpoint strategy rationale (2026-04-05):**
Val surrogate loss reaches minimum by epoch 4-13 but PST best occurs at epoch 30-90. Controlled comparison (11 runs) showed PST-based checkpoint selection significantly outperforms val surrogate loss-based selection (eval avg 0.547-0.589 vs 0.395-0.517). No early stopping is used — models train for full max_epochs.

### 5.4 Experimental Findings (2026-04)

#### Feature Ablation

Original node features (gate_count, two_qubit_gate_count, degree, depth_participation) suffered from severe degeneracy:
- `gate_count` ↔ `depth_participation` correlation |r| ≈ 1.0 in 100% of circuits
- `degree` constant (→ zero after z-score) in 26% of benchmark circuits
- Effective dimensionality: ~2.1 / 4

Replacement features (gate_count, two_qubit_gate_count, single_qubit_gate_ratio, critical_path_fraction + RWPE k=2) improved effective dimensionality to 3.7 / 6 and eliminated perfect correlations (max |r| = 0.87).

Note: `weighted_degree` was found to be mathematically identical to `two_qubit_gate_count` (r = 1.0), providing no independent information.

#### Stage 1 Supervised Learning Results

| Experiment | Features | Stage 1 Best Val Loss (Phase 1 / Phase 2) |
|------------|----------|-------------------------------------------|
| 1-A | old (gc,2qc,deg,dp) | 35.70 / 55.36 |
| 1-C | new (gc,2qc,sqr,cpf+RWPE2) | 37.73 / 69.65 |

Old features performed better in Stage 1 supervised learning. This is expected — Stage 1 is essentially label memorization, and the original features (including degree, which directly reflects graph topology) may be more aligned with the specific label encoding scheme.

#### Stage 2 Surrogate Learning Results

**Historical runs (pre-filtering, 6,887 circuits):**

| Experiment | Score Norm | Loss | Features | Best Val PST | Notes |
|------------|-----------|------|----------|-------------|-------|
| scratch_error_dist_adj | **Sinkhorn** | err_dist + adj | old 4dim, HW 7dim, bias=7 | **0.3440** | Previous best |
| stage2_200epochs | Sinkhorn | err_dist + node_q | old 4dim | 0.3572 | Best but 200ep |
| scratch_default | Sinkhorn | err_dist + node_q | old 4dim | 0.3495 | |
| s2_old_feat_pretrained | Softmax | err_dist + node_q | old 4dim | 0.3044 | With Stage 1 |
| s2_new_feat_scratch | Softmax | err_dist + node_q | new 6dim | 0.2410 | New features |
| Baseline (QAP+NASSC) | — | — | — | **0.3785** | Target to beat |

**Current best (filtered 5,769 circuits, 2026-04-02):**

| Experiment | Score Norm | Loss | Features | Best Val PST |
|------------|-----------|------|----------|-------------|
| filtered_sinkhorn_adj | Sinkhorn | err_dist(1.0) + adj(0.3) | new 6dim, HW 5dim, bias=0 | **0.3588** |
| filtered_seed42_gpu0 | Softmax | err_dist + node_q | new 6dim, HW 5dim, bias=0 | 0.2727 |

**Phase 1: Edge loss optimization (2026-04-03):**

All use Sinkhorn, new 6dim features, HW 5dim, bias=0, filtered 5,769 circuits.

| Exp | Loss Config | Config File |
|-----|-------------|-------------|
| E1 | err_dist(1.0) + adj(**0.7**) | stage2_sinkhorn_adj.yaml + override |
| E2 | err_dist(1.0) + adj(**1.0**) | stage2_sinkhorn_adj.yaml + override |
| E3 | **swap(1.0)** + adj(0.3) | stage2_swap_adj.yaml |
| E4 | **swap(1.0)** standalone | stage2_swap_only.yaml |
| E5 | err_dist(1.0) + **soft(0.3, α=2)** | stage2_soft_proximity.yaml |
| E6 | **soft(1.0, α=2)** standalone | stage2_soft_only.yaml |

**Key observations:**
1. **Surrogate loss saturation**: `error_distance` saturates by epoch 3 (drops from ~0.14 to ~0.01). `node_quality` collapses to -1.0 by epoch 1-2. `adjacency` is the only loss providing meaningful gradient throughout 100 epochs (-0.10 → -0.28).
2. **Val PST oscillation**: PST fluctuates widely (0.12–0.36) across epochs rather than converging, suggesting poor correlation between surrogate loss and actual PST.
3. **Sinkhorn >> Softmax (confirmed)**: Controlled experiment: Sinkhorn 0.3588 vs Softmax 0.2727 (+0.086). All experiments now use Sinkhorn.
4. **New features >> Old features (confirmed)**: gc,2qc,sqr,cpf+RWPE2 → 0.3588 vs gc,2qc,deg,dp → 0.2473 (+0.111).
5. **HW 5dim > 7dim+noise_bias (confirmed)**: 0.3588 vs 0.2604. t1/t2 raw values + noise_bias hurt.
6. **node_quality harmful**: Without 0.3588 vs with 0.3474. Learned MLP collapses to trivial solution.
7. **Stage 1 no longer in use**: Supervised pretraining adds complexity without clear benefit. All current experiments run Stage 2 from scratch.
5. **Feature-indistinguishable filtering**: 1,118 circuits removed (16.2%) — primarily MQT Bench parametric circuits. Effect on training quality under evaluation.

---

## 6. Dataset Usage

### Available Datasets

| Dataset | Original | After Steps 1-6 | After Step 7 (Diversity) | Stage | Label Source | Notes |
|---------|:--------:|:---------------:|:------------------------:|-------|--------------|-------|
| **MQT Bench** | 1,219 | 433 | **305** | Stage 2 (Unsupervised) | None | 786 indist + 128 structural duplicates removed |
| **MLQD** | 4,443 | 4,159 | **267** | Stage 1 + 2 | OLSQ2 solver labels | 275 indist + 14 mid-measure + 3,892 structural duplicates removed |
| **QUEKO** | 900 | 894 | **245** | Stage 1 + 2 | τ⁻¹ (true optimal) | 6 indist + 649 random-seed variants removed |
| **QASMBench** | 94 | 52 | **39** | Stage 2 (Unsupervised) | None | 42 + 7 mid-measure/dup removed |
| **RevLib** | 231 | 219 | **113** | Stage 2 (Unsupervised) | None | 12 indist + 106 indexed-variants removed |
| **Total** | **6,887** | **5,757** | **969** | | | |

### Dataset Directory Structure

All circuit data is stored under `data/circuits/` with circuits, labels, and split definitions separated:

```
data/circuits/
├── qasm/                        # Raw .qasm files organized by source dataset
│   ├── mqt_bench/               # 1,219 circuits (29 algorithms, 2-127Q, no labels yet)
│   ├── mlqd/                    # 4,443 circuits (3,729 with OLSQ2 labels)
│   ├── queko/                   # 900 circuits (540 with τ⁻¹ labels, 360 without)
│   ├── qasmbench/               # 94 circuits (2Q-127Q, label-free)
│   └── revlib/                  # 231 circuits (3Q-127Q, converted from .real)
├── labels/                      # Label files — only for circuits with usable labels
│   ├── mqt_bench/               # (no labels — Stage 2 unsupervised only)
│   ├── mlqd/labels.json         # OLSQ2 solver labels (3,729 circuits)
│   └── queko/labels.json        # τ⁻¹ true optimal labels (540 circuits)
├── backends/                    # Synthetic backend definitions for non-Qiskit hardware
│   ├── queko_aspen4.json        # Rigetti Aspen-4 (16Q) — QUEKO + MLQD
│   ├── queko_tokyo.json         # IBM Tokyo (20Q) — QUEKO only
│   ├── queko_rochester.json     # IBM Rochester (53Q) — QUEKO only
│   ├── queko_sycamore.json      # Google Sycamore (54Q) — QUEKO + MLQD
│   └── mlqd_grid5x5.json       # 5x5 Grid (25Q) — MLQD only
└── splits/                          # Defines which circuits are used in each stage
    ├── stage1_supervised.json       # 288 labeled circuits → Stage 1 training
    ├── stage1_queko_only.json       # 73 QUEKO circuits → Stage 1 fine-tuning phase
    ├── stage1_unsupervised.json     # 653 unlabeled circuits → Stage 2 only
    ├── stage2_all.json              # 969 all circuits → Stage 2 surrogate loss
    ├── val.json                     # 28 labeled validation
    ├── val_queko_only.json          # 2 QUEKO validation
    ├── filter_log.json              # Combined removal log (indist + mid-measure + diversity)
    ├── diversity_filter_log.json    # Per-source K and per-cluster details
    ├── mid_measure_log.json         # Mid-circuit measurement scan log
    ├── dataset_quality.{md,csv}     # Per-category quality metrics
    ├── dataset_diversity.{md,csv}   # Per-category diversity / fingerprint analysis
    └── original/                    # Pre-filter backup of all splits
```

**Design rationale:**
- **Circuits and labels are decoupled.** Even within a labeled dataset (e.g., MQT Bench), only a subset may have labels compatible with our experimental setup. The remaining circuits are still valuable for Stage 2 unsupervised training.
- **Label format:** JSON mapping from circuit filename to layout: `{"circuit.qasm": {"backend": "manila", "layout": [0, 1, 3, 2, 4]}, ...}`
- **Split files control training behavior.** Adding new labels or circuits only requires updating `labels/*.json` and `splits/*.json` — no reorganization of circuit files.
- **QASMBench and RevLib** have no `labels/` directory entry (always unsupervised, Stage 2 only).
#### Dataset Preprocessing Pipeline

All raw circuit datasets undergo the following preprocessing before use in training. Each step is applied once and the results are stored in place.

**Step 1: Gate Normalization** (`scripts/normalize_gates.py`)
- All QASM files are transpiled to Qiskit standard basis gates `{cx, id, rz, sx, x}` via `transpile(circuit, basis_gates=..., optimization_level=0)` (pure decomposition, no gate optimization/merging)
- **Why:** Original datasets use incompatible gate sets — QUEKO uses only `x`/`cx`, MLQD uses `h`/`cx`/`sx`, while MQT Bench/RevLib contain 3+ qubit gates (`ccx`, `mcx`, `cswap`) that are invisible to the 2-qubit-only feature extraction in `circuit_graph.py` (`len(qubit_indices) == 2` condition). Without decomposition, multi-qubit gate interactions are completely missing from the circuit graph.
- Qubit counts are preserved (transpile does not add ancillas at optimization_level=0)

**Step 2: Untranspilable Circuit Removal**
- Circuits that cannot be transpiled within reasonable time/memory are removed
- Removed: `grover_n26.qasm`, `grover_n28.qasm` (MQT Bench) — 26/28-qubit custom gate wrapping entire circuit, transpile exceeds 10 min
- Removed: 32 circuits with QASM file size > 10 MB (24 from MQT Bench, 8 from RevLib) — Qiskit DAG parsing requires tens of GB memory, causing OOM

**Step 3: Evaluation Benchmark Deduplication**
- Circuits in `data/circuits/qasm/benchmarks/` (23 evaluation circuits) are checked against all training datasets for filename overlap
- Removed from training sets: 17 RevLib circuits + 2 MQT Bench circuits (`bv_n3`, `bv_n4`) that duplicate benchmark circuits
- **Why:** Training on evaluation circuits would make PST benchmarks unfair

**Step 4: Extreme Circuit Filtering** (edges > 1,000)
- Circuits with more than 1,000 unique 2Q qubit pairs (edges in the circuit interaction graph) are removed
- Removed: 182 MQT Bench + 1 QASMBench = 183 circuits
- **Why:** These are fully-connected circuits (QFT, QPE at 60-127Q with up to 8,001 edges) that cause GNN message passing memory/compute explosion and batch size imbalance. The labeled datasets (QUEKO/MLQD) have max 88/24 edges respectively — 1,000 provides 11× headroom while filtering extreme outliers.
- After filtering, max edges = 996

**Step 5: Feature-Indistinguishable Filtering** (`scripts/filter_indistinguishable.py`)
- For each circuit, compute pairwise cosine similarity between qubit feature vectors (current default features + RWPE k=2, after z-score normalization)
- Remove circuits where > 30% of qubit pairs have cosine similarity > 0.95 ("indistinguishable")
- These circuits provide zero/misleading gradient signal: the GNN cannot learn meaningful qubit-to-physical mappings when features don't distinguish qubits
- Removal log: `data/circuits/splits/filter_log.json`
- Original splits backed up to `data/circuits/splits/original/`

**Why this threshold:** At 50% threshold, only 607 circuits removed (8.8%), missing many "poor" quality circuits (30-50% indist rate). At 30%, 1,118 removed (16.2%) with acceptable labeled data loss (247 supervised, 27 val). The 30% threshold ensures training data has sufficiently differentiated features for GNN learning.

**Step 6: Mid-circuit Measurement Filtering** (`scripts/filter_mid_measure.py`)
- Scan all QASM files via `scripts/check_mid_measure.py` for any non-measure/barrier operation that follows a measure on the same qubit
- 21 circuits found: 7 unique algorithms (bb84, ipea, shor, cc×3, seca), replicated across MLQD backend variants. 9 already removed by Step 5, 12 additional removals
- **Why:** The GraphQMap circuit graph represents each logical qubit as a single node — mid-measure circuits where a qubit is measured and then reused for a different role cannot be represented correctly. Also, many solvers used to produce labels (including OLSQ2) assume unitary-only circuits, making label correctness dubious for such circuits.
- Removal log: `data/circuits/splits/mid_measure_log.json` (full scan results)

**Step 7: Strong Diversity Filtering** (`scripts/filter_diversity.py`, applied 2026-04-08)
- For each source, group circuits by fingerprint `(num_qubits, num_edges, sorted_degree_sequence)`. Keep K=1 representative per fingerprint (alphabetical first stem).
- **Why:** Pre-filter diversity analysis (`scripts/dataset_diversity.py`) revealed that nominal 5,757 training circuits contained only ~487 structurally distinct patterns (8.5%). Key duplication sources:
  - **QUEKO**: 215-circuit clusters of `{N}QBT_{depth}CYC_QSE_{seed}` random-seed variants with identical topology. 99% of QUEKO falls in "sparse" topology, only 4 distinct qubit counts (16/20/53/54).
  - **MLQD**: 359-circuit cluster at fingerprint (4Q, 3 edges, …) with mixed algorithm names (basis_trotter, dnn, ising, vqe_uccsd) — structurally identical despite semantic labels. 5×backend replication compounds the effect.
  - **RevLib**: indexed variants (`rd53_71`, `rd53_138`, ...) of the same base circuit.
  - **MQT Bench and QASMBench**: already diverse (singleton rate > 80%), mild K=1 compression only removes trivial 2Q pairs.
- Reduction: queko 894→245, mlqd 4,159→267, mqt_bench 433→305, qasmbench 52→39, revlib 219→113
- **Trade-off:** Validation set drops significantly (val 395→28, val_queko_only 52→2). Val metrics will have higher per-epoch variance but the number of structurally distinct val patterns is unchanged — the removed val entries were duplicates contributing no additional signal.
- Detailed log: `data/circuits/splits/diversity_filter_log.json`; analysis outputs: `dataset_diversity.{md,csv}`

**Effective unique count rationale:** Defining unique as matching `(num_qubits, num_edges, sorted_degree_sequence)` is an under-estimate of true structural uniqueness (two circuits with the same degree sequence could still differ in edge placement) but is a strong over-estimate of "what the GNN distinguishes" because feature extraction throws away even more detail. K=1 per fingerprint gives a conservative-but-actionable upper bound on the distinct patterns the current model architecture can learn from.

**Primary removal targets:**
- MQT Bench VQE/QNN/GHZ/W-state: parametric circuits with repeating layer structure → all qubits have identical features
- MLQD ising circuits: 38.4% bad rate due to uniform interaction patterns
- MLQD dnn (2Q): very small circuits with degenerate features

**Note on MLQD sqr=0:** 57% of MLQD circuits have constant `single_qubit_gate_ratio` = 0 (2Q-gate-only circuits). These are NOT removed because other features (gc, 2qc, cpf) still differentiate qubits — only 58 MLQD circuits exceed the 50% indist threshold.

**Preprocessing summary:**

| Step | Circuits Removed | Reason |
|------|:----------------:|--------|
| Gate normalization | 0 (in-place) | Standardize gate representation |
| Untranspilable | 34 | OOM / timeout during transpile |
| Benchmark dedup | 19 | Evaluation fairness |
| Extreme filtering | 183 | edges > 1,000, GNN scalability |
| Feature-indistinguishable | 1,118 | indist rate > 30%, no gradient signal |
| Mid-circuit measurement | 12 | mid-measure not representable in graph + dubious labels |
| Strong diversity filter (K=1) | 4,788 | structural near-duplicates by fingerprint |
| **Total removed** | **6,196** | |

**Original → Final circuit counts:**

| Dataset | Original | After Steps 1-4 | After Step 5 | After Step 6 | After Step 7 (Final) |
|---------|:--------:|:---------------:|:------------:|:------------:|:--------------------:|
| QUEKO | 900 | 900 | 894 | 894 | 245 |
| MLQD | 4,443 | 4,443 | 4,168 | 4,159 | 267 |
| MQT Bench | 1,448 | 1,219 | 433 | 433 | 305 |
| QASMBench | 111 | 94 | 55 | 52 | 39 |
| RevLib | 263 | 231 | 219 | 219 | 113 |
| **Total** | **7,165** | **6,887** | **5,769** | **5,757** | **969** |

#### QUEKO Backend Handling

QUEKO circuits are designed for 4 specific hardware topologies (Aspen-4, Tokyo, Rochester, Sycamore) that are not available as Qiskit FakeBackendV2. Since QUEKO only provides coupling maps without noise data, we generate **synthetic noise profiles** by sampling from distributions observed across real FakeBackendV2 hardware:

- Noise values (T1, T2, readout_error, sq_gate_error, cx_error, cx_duration) are sampled from clipped normal distributions fitted to 11 real FakeBackends. The JSON files also include `frequency`, but it is not used as a model feature.
- Generated once with fixed seed (42) for reproducibility, stored in `data/circuits/backends/`
- QUEKO's optimal layouts are topology-based (zero-SWAP), so synthetic noise does not affect label correctness
- The model learns topology-aware mapping from QUEKO in Stage 1; noise-aware optimization follows in Stage 2

See `scripts/generate_queko_noise.py` for the generation script.

#### MLQD Backend and Label Handling

MLQD provides OLSQ2-mapped result circuits but not explicit initial layouts. Layouts are extracted by:

1. Parsing measurement lines in the result circuit to obtain the **final mapping** (logical → physical after all SWAPs)
2. Detecting SWAP patterns (3-CNOT decomposition: `cx a,b; cx b,a; cx a,b`) and reversing them to recover the **initial layout**
3. Circuits where SWAP detection fails (pattern mismatch) are kept as unlabeled for Stage 2

**Backend mapping for MLQD:**
- **Melbourne, Rochester** → Qiskit FakeMelbourneV2 / FakeRochesterV2 (real noise data available)
- **Aspen-4, Sycamore** → reuse QUEKO synthetic noise profiles (identical topologies)
- **Grid 5x5** → dedicated synthetic noise profile (`mlqd_grid5x5.json`)

See `scripts/process_mlqd.py` for the extraction script.

### Training Strategy: Hybrid Supervised + Unsupervised

**Stage 1 (Supervised):** Use existing labels directly from MLQD (OLSQ2 solver labels) and QUEKO (τ⁻¹ true optimal). No self-generated label pipeline required — existing labels provide sufficient supervised signal for learning basic mapping quality, and any router-specific bias is corrected in Stage 2.

**Stage 2 (Unsupervised):** Fine-tune with configurable surrogate losses (default: L_surr + L_node) on all available circuits, including label-free datasets (MQT Bench, QASMBench, RevLib). Loss components are modular and configured via YAML registry. This stage aligns the model toward NISQ-aware PST optimization, compensating for any mismatch between existing labels and the evaluation pipeline. **Stage 2 uses only the 55 real Qiskit FakeBackendV2 backends** — synthetic backends are excluded. QUEKO/MLQD circuits (which were originally assigned to synthetic backends) are randomly re-assigned to real backends at data load time. This ensures the model's unsupervised fine-tuning generalizes to real hardware noise profiles.

### Rationale for Using Existing Labels

- **MLQD OLSQ2 labels** were optimized for the OLSQ2 routing pipeline, but still encode meaningful mapping quality signal (e.g., minimizing qubit interaction distance)
- **QUEKO τ⁻¹ labels** are true topology-optimal mappings (zero-SWAP overhead) — the highest quality supervised signal available
- **MQT Bench** was initially considered for pseudo-labels but excluded because mapped-level data is effectively unavailable from MQT Bench web/API
- **Stage 2 unsupervised fine-tuning corrects router-specific bias** — surrogate losses directly optimize layout quality metrics
- This approach **eliminates the massive computational cost** of self-generating labels while maintaining training effectiveness

---

## 7. Multi-Programming Training Data

### Design Principle

The model itself is designed to handle **any number of co-located circuits** without a fixed upper bound. The only hard constraint is that the total logical qubit count must not exceed the physical qubit count of the target backend. However, training on all possible circuit combinations is computationally infeasible, so multi-programming combinations are **pre-configured** for each training run.

### Circuit Combination Rules

- **Hard constraint:** Total logical qubits of combined circuits **must not exceed** physical qubit count of target backend
- **Occupancy limit:** Maximum **75%** of physical qubits
- **Occupancy range:** **30–75%** uniformly sampled for diversity
- **Combination method:** Random pairing with diverse circuit sizes

### Training Data Configuration

Multi-programming scenarios and their proportions are specified in the training config (YAML). Example:

```yaml
multi_programming:
  scenarios: [1, 2, 4]       # Number of circuits per scenario
  proportions: [0.5, 0.3, 0.2]  # Sampling ratio per scenario
```

The scenarios and proportions are fully configurable — not restricted to specific values. Users can add higher circuit counts (e.g., 3, 5, 8) or adjust proportions as needed. The model architecture imposes no limit on the number of co-located circuits.

### Multi-Programming Circuit Graph Construction

1. Construct individual circuit graphs for each circuit (4-dim node features, same as single-circuit; edge features z-score normalized per-circuit)
2. Re-normalize edge features at group level via `renormalize_group_edges()` (ensures consistent edge scales across circuits)
3. Merge into a single disconnected graph (no edges between circuits)
4. Total node count = l₁ + l₂ + ... (sum of all circuits' logical qubits)

---

## 8. Batching Strategy

### GNN Level

Use **PyTorch Geometric standard batching**: multiple graphs merged into a single disconnected graph with batch index tracking. Handles variable node counts without padding natively.

### Score Matrix Level: Backend-Based Bucketing

**Samples using the same hardware backend are grouped into the same mini-batch.** This ensures:
- All samples in a batch have identical h (physical qubit count)
- Score Matrix / Softmax / Loss computed as 3D tensor `(batch_size × l × h)` in parallel

### Dynamic Batch Size

Batch size determined by total physical qubit budget to maintain consistent GPU memory:

```python
max_total_nodes = 512  # Total physical qubits per batch (tune to GPU memory)

# Effective batch sizes:
# Manila (5Q):     ~102 samples
# Guadalupe (16Q): ~32 samples
# Montreal (27Q):  ~19 samples
```

### Backend Data Balance

- **Backend-level sampling weights:** All backends appear with roughly equal frequency per epoch
- Oversample data-scarce backends, undersample data-rich backends
- **Epoch ordering:** Backend group order randomized each epoch

---

## 9. Evaluation Protocol

### Hardware Split

| Role | Backends |
|------|----------|
| **Training** | **5Q:** Athens, Belem, Bogota, Burlington, Essex, Lima, London, Manila, Ourense, Quito, Rome, Santiago, Valencia, Vigo, Yorktown · **7Q:** Casablanca, Jakarta, Lagos, Nairobi, Oslo, Perth · **15-16Q:** Melbourne, Guadalupe · **20Q:** Almaden, Boeblingen, Johannesburg, Poughkeepsie, Singapore · **27-28Q:** Algiers, Auckland, Cairo, Cambridge, Geneva, Hanoi, Kolkata, Montreal, Mumbai, Paris, Peekskill, Sydney · **33Q:** Prague · **53Q:** Rochester · **65Q:** Manhattan · **127Q:** Brisbane, Cusco, Kawasaki, Kyiv, Kyoto, Osaka, Quebec, Sherbrooke, Washington |
| **Test (UNSEEN)** | **FakeToronto (27Q)**, **FakeBrooklyn (65Q)**, **FakeTorino (133Q)** — completely excluded from training |

This split enables rigorous evaluation of **hardware-agnostic generalization**.

### Test Scenarios

The model supports **arbitrary multi-programming** — any number of circuits can be compiled simultaneously, as long as the total logical qubit count does not exceed the physical qubit count of the target backend. Evaluation covers single-circuit and multi-circuit scenarios as configured.

### Metrics

| Type | Metric | Definition |
|------|--------|------------|
| **Primary** | PST (Probability of Successful Trials) | P(correct output) — probability of the ideal most-probable bitstring appearing in noisy execution. Standard definition used in QUEKO and multi-programming papers. |
| **Secondary** | SWAP count, circuit depth | Compilation quality metrics. |
| **Speed** | Inference latency (model forward + Hungarian), end-to-end time (+ transpile) | |

**PST computation pipeline:**
1. Create AerSimulator with `tensor_network` + GPU (cuQuantum) — handles any circuit size without OOM
2. Simulators created **once per backend**, reused across all circuits (ideal_sim + noisy_sim)
3. Transpile circuit with initial layout (via custom PassManager supporting layout×routing combinations)
4. Run ideal simulation (same transpiled circuit, no noise) to find most probable bitstring
5. Run noisy simulation (same transpiled circuit, backend noise model)
6. PST = noisy_counts[ideal_bitstring] / total_shots
7. Supports multi-register circuits (space-separated bitstrings → per-register PST averaged)

**Simulation method priority:** tensor_network+GPU → tensor_network+CPU (raises error if neither available)
**Default shots:** 8192

**Robustness:** On large backends (100Q+), model-generated layouts may produce very deep transpiled circuits that crash the tensor_network simulator (`CUTENSORNET_STATUS_INVALID_VALUE`). This corrupts GPU state for subsequent simulations. Mitigations:
- **Evaluation order:** Baselines run before model evaluation per circuit, so baseline results are never affected by model failures.
- **Simulator recovery:** On simulation failure, simulators are recreated and the failed run is recorded as NaN (excluded from averaging via `nanmean`).

### Transpilation

All evaluation paths (baselines and model) use a unified transpilation function (`transpile_with_timing()` in `evaluation/transpiler.py`). This ensures identical pass pipelines, noise-aware `UnitarySynthesis` (with `backend_props`), and consistent per-stage timing/metadata collection across all methods.

Custom PassManager builder (`evaluation/transpiler.py`) supporting all combinations:

| Layout Method | Routing Method | Description |
|---------------|---------------|-------------|
| `graphqmap` (model) | `sabre` | Our method + standard routing |
| `graphqmap` (model) | `nassc` | Our method + noise-aware routing |
| `sabre` | `sabre` | Qiskit default baseline |
| `sabre` | `nassc` | SABRE layout + noise-aware routing |
| `dense` | `sabre` | Dense layout baseline |
| `noise_adaptive` | `sabre` | Noise-adaptive layout baseline |
| `trivial` | `sabre` | Identity mapping baseline |
| `qap` | `sabre` | QAP-based layout (MQM) + standard routing |
| `qap` | `nassc` | QAP-based layout (MQM) + noise-aware routing |

Per-stage timing measured: init, layout, routing, optimization, scheduling. Metadata includes `map_cx` (2Q gate count after routing, before optimization) for SWAP count estimation.

### Benchmark Circuits

Standard evaluation set (shared with MQM colleague for direct comparison):
`toffoli_3`, `fredkin_3`, `3_17_13`, `4mod5-v1_22`, `mod5mils_65`, `alu-v0_27`, `decod24-v2_43`, `4gt13_92`

Extended set adds: `bv_n3`, `bv_n4`, `peres_3`, `xor5_254`

All benchmark circuits stored in `data/circuits/qasm/benchmarks/` (23 total .qasm files, gate-normalized to `{cx, id, rz, sx, x}`).
For fair comparison with MQM colleague, use original (non-normalized) circuits via `--circuit-dir references/colleague/tests2/benchmarks`. Gate normalization reduces basis translation overhead, which inflates PST relative to the colleague's results.

### Statistical Reliability

Multiple repetitions per experiment (matching baseline research), reporting mean and standard deviation.
Results presented as pandas DataFrame with per-circuit PST, depth, CX count, timing, and Avg row.

### Evaluation Output Pipeline

`evaluate.py` with `--output` automatically generates all evaluation artifacts in the run's `eval/` subdirectory:

```
<run_dir>/
├── eval_results.csv         # Raw results: circuit × backend × method × rep (all backends combined)
└── eval/
    ├── pst_summary.md       # Per-circuit PST table for each backend (Markdown)
    ├── pst_summary.csv      # Same in CSV format
    ├── pst_comparison_*.png # PST bar charts (method comparison per backend)
    └── pst_heatmap_*.png    # PST heatmaps (circuit × method per backend)
```

When evaluating multiple backends (e.g. `--backend toronto brooklyn torino`), all results are collected into a single combined CSV. Summary tables include per-backend sections with per-circuit breakdowns and AVG rows.

### Baselines

| Method | Description |
|--------|-------------|
| SABRE (layout+routing) | Qiskit default, standard baseline |
| NASSC (noise-aware routing) | SABRE layout + NASSC routing ([Li et al.](https://arxiv.org/abs/2305.06780)) |
| DenseLayout | Maps to densely connected subgraph |
| NoiseAdaptive | Noise-adaptive layout ([Murali et al., ASPLOS 2019](https://arxiv.org/abs/1901.11054)) |
| QAP (MQM) | QAP-based mathematical layout optimization (colleague's method) |
| Trivial | Identity mapping (qubit i → physical i) |
| Random | Uniformly random physical qubit assignment |
| Naive multi-prog | Independent SABRE per circuit (multi-programming only) |

---

## 10. Ablation Study Candidates

### Priority 1: Core Claims

| Ablation | Tests |
|----------|-------|
| Loss component ablation | error_distance vs swap_count; adjacency vs soft_proximity; standalone vs combined |
| Stage 1 only vs. Stage 1+2 | Surrogate fine-tuning contribution |
| Single-hardware vs. Multi-hardware training | Hardware-agnostic claim |

### Priority 2: Architecture

| Ablation | Variants |
|----------|----------|
| Embedding dim d | 32 vs. 64 vs. 128 |
| GNN layers | 2 vs. 3 vs. 4 |
| Directed vs. Undirected | Edge directionality effect |

### Priority 3: Training Details

| Ablation | Tests |
|----------|-------|
| τ annealing vs. fixed τ | Training-inference gap |
| L_node presence/absence | Node quality loss contribution |
| Noise features in Hardware GNN | With vs. without calibration features |
| Noise parameter perturbation | Robustness to calibration drift |

---

## 11. Open Items

These items are not yet finalized and need to be decided during implementation:

### Optimizer Details (Decided)
- **Confirmed:** AdamW + Cosine Annealing LR scheduler
- **Stage 1 LR:** 1e-3, weight decay 1e-4, cosine eta_min 1e-6
- **Stage 2 LR:** 5e-4, weight decay 1e-4, cosine eta_min 1e-5, warmup 2 epochs
- **Stage 2 Softmax τ:** exponential decay from 1.0 to 0.05 (starts softer for wider exploration with surrogate losses)
- **Gradient clipping (Stage 2):** max_norm 2.0

### Reproducibility Settings (Decided)
- Random seed: 42 (training), 43 (evaluation)
- Seed applied to: Python random, NumPy, PyTorch (CPU + CUDA)

---

## Appendix A: Key Implementation Notes

### Noise Data Extraction from FakeBackendV2

```python
from qiskit_ibm_runtime.fake_provider import FakeToronto  # example (V2)
from qiskit_ibm_runtime.fake_provider import FakeBrisbane  # example (non-V2, uses ecr)

backend = FakeToronto()

# Per-qubit properties:
# readout_error, single-qubit gate errors, degree, t1_cx_ratio, t2_cx_ratio
# Extract from backend.properties() or backend.target

# Per-edge properties:
# 2q_error (cx_error) from backend.target
# Native 2-qubit gate varies by backend: cx, ecr, or cz
# Use _get_two_qubit_gate_name() to detect automatically
```

### Error-Aware Distance Precomputation

```python
import numpy as np
from scipy.sparse.csgraph import floyd_warshall

# Build adjacency matrix with cx_error as weights
adj = np.full((h, h), np.inf)
np.fill_diagonal(adj, 0)
for (p, q), error in cx_errors.items():
    adj[p][q] = error
    adj[q][p] = error  # undirected

# Floyd-Warshall
d_error = floyd_warshall(adj)  # h×h matrix, precomputed once per backend
```

### q_score Implementation

```python
class QualityScore(nn.Module):
    def __init__(self, num_features=5, hidden_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features):
        """
        node_features: (h, 5) tensor
        Columns: [readout_error_norm, single_qubit_error_norm,
                  degree_norm, t1_cx_ratio_norm, t2_cx_ratio_norm]
        All pre-normalized within backend (z-score)
        """
        return torch.sigmoid(self.mlp(node_features).squeeze(-1))  # (h,), [0, 1]
```

### Score Normalization (SoftmaxNorm)

```python
def forward(self, S, num_logical, num_physical, tau):
    """
    S: (batch, l, h) raw score matrix
    Returns: P (batch, l, h) row-stochastic matrix
    """
    return F.softmax(S / tau, dim=-1)
```

### Multi-Programming Graph Merging

```python
from torch_geometric.data import Batch

def merge_circuits_for_multi_programming(circuits):
    """
    circuits: list of PyG Data objects (one per circuit)
    Returns: single merged PyG Data object
    """
    # Merge into disconnected graph (no summary features added — same 4-dim as single-circuit)
    merged = Batch.from_data_list(circuits)
    # merged.x: (l1+l2+...+lk, 4)
    # merged.edge_index: all edges with adjusted node indices
    # merged.batch: tracks which node belongs to which original circuit
    return merged
```

### Hungarian Algorithm for Inference

```python
from scipy.optimize import linear_sum_assignment

def hungarian_decode(P, l):
    """
    P: (h, h) doubly stochastic matrix
    l: number of actual logical qubits
    Returns: layout dict {logical_qubit: physical_qubit}
    """
    cost = 1 - P.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # Only take the first l assignments (actual logical qubits)
    layout = {}
    for i in range(l):
        layout[i] = col_ind[i]
    return layout
```

---

## Appendix B: Complete Hyperparameter Summary

| Category | Parameter | Value |
|----------|-----------|-------|
| **GNN** | Embedding dim (d) | 64 |
| | GATv2 Layers | 3 |
| | Attention Heads | 4 |
| | Activation | ELU |
| | Dropout | 0.1 |
| | BatchNorm | Per layer |
| | Residual | Layers 2, 3 |
| **Cross-Attention** | Layers | 2 |
| | Heads | 4 |
| | FFN Hidden | 128 |
| | Dropout | 0.1 |
| **Score Head** | d_k | 64 |
| | noise_bias_dim | 0 (disabled by default) |
| **QualityScore** | Architecture | MLP: 5 → 16 → 1 + sigmoid |
| | Hidden dim | 16 |
| **SoftmaxNorm** | τ_max | 1.0 |
| | τ_min | 0.05 |
| | Schedule | Exponential decay (Stage 1), Fixed (Stage 2) |
| **Stage 2 Loss** | Components | Configurable via YAML registry |
| | Current best | error_distance (1.0) + adjacency (0.3) |
| | Available | error_distance, adjacency, hop_distance, swap_count, soft_proximity, node_quality, separation, exclusion |
| **Batching** | Max total nodes | 512 (tune to GPU) |
| | large_backend_boost | 2.0 (oversample 50Q+ backends) |
| **Multi-prog** | Scenarios | Configurable (default: [1, 2, 4] with proportions [0.5, 0.3, 0.2]) |
| **Labels** | Stage 1 sources | MLQD (OLSQ2, 3,729), QUEKO (τ⁻¹, 540) |
| | Stage 2 (unsupervised) | MQT Bench, QASMBench, RevLib (+ all Stage 1 circuits) |
| **Optimizer** | Type | AdamW |
| | Weight Decay | 1e-4 |
| | LR (Stage 1) | 1e-3, reduced to 1e-4 for QUEKO fine-tuning (lr_factor=0.1) |
| | LR (Stage 2) | 5e-4 |
| | LR Scheduler | Cosine Annealing (eta_min=1e-5 for Stage 2) |
| | Warmup (Stage 2) | 2 epochs |
| | Grad Clip (Stage 2) | max_norm 2.0 |
| **Transitions** | MLQD+QUEKO→QUEKO | Val CE early stop, patience 15 |
| | QUEKO fine-tuning end | Val CE early stop, patience 10 |
| | Stage 1→2 | Manual (after Stage 1 completes) |
| | Stage 2 end | No early stopping; full max_epochs, best checkpoint by val PST |
| **Reproducibility** | Training seed | 42 |
| | Evaluation seed | 43 |
