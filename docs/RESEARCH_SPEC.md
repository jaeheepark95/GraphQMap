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
[Dummy Padding] → S_padded (h×h) by appending (h-l) dummy rows (zeros)
  ↓
[Log-domain Sinkhorn] → P (h×h) doubly stochastic matrix
  ↓
[Hungarian Algorithm] → Discrete one-to-one mapping (use top l rows only)
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
- **Sinkhorn temperature τ** is annealed during Stage 1

### Multi-Programming Pipeline

For multiple circuits (e.g., C₁ with l₁ qubits, C₂ with l₂ qubits):
1. Merge circuit graphs into a single **disconnected graph** (no edges between circuits) — node features stay at **4 dimensions** (same as single-circuit)
2. Circuit GNN output becomes (l₁+l₂)×d
3. Rest of pipeline is identical — Sinkhorn's doubly stochastic constraint automatically prevents mapping conflicts
4. **No architectural modifications required** — single-circuit is simply a special case of multi-programming with one circuit

---

## 3. Graph Representation

### 3.1 Circuit Graph

**Node:** Each logical qubit = one node.

**Edge:** Undirected. An edge exists between logical qubits i and j if any 2-qubit gate connects them. Multiple gates on the same pair are merged into a single edge.

**Node Features (per logical qubit):**

| Feature | Description |
|---------|-------------|
| `gate_count` | Total number of gates on this qubit (includes both 1Q and 2Q gates) |
| `two_qubit_gate_count` | Number of 2-qubit gates involving this qubit (each gate counted on both qubits symmetrically) |
| `degree` | Number of distinct qubits this qubit interacts with via 2-qubit gates |
| `circuit_depth_participation` | Fraction of circuit depth in which this qubit is active (not idle) |

**Node feature design notes:**
- `gate_count` includes `two_qubit_gate_count` (i.e., `gate_count` ≥ `two_qubit_gate_count`), so the two features are correlated. The incremental information from `gate_count` is the single-qubit gate count, which partially overlaps with `circuit_depth_participation`. This redundancy is tolerated — the GNN learns which combinations are informative, and 4 features is a small enough dimension that overfitting is not a concern.
- `degree` and `two_qubit_gate_count` are also correlated (more interactions → more unique neighbors), but degree specifically captures "hub" structure (fan-out) while `two_qubit_gate_count` captures total interaction load.
- 2-qubit gate counts are applied **symmetrically** to both control and target qubits. For initial layout, the interaction frequency (not the control/target role) is what drives placement decisions — both qubits experience the same decoherence during the gate.
- **Ablation candidate:** whether `gate_count` adds value beyond `two_qubit_gate_count` + `circuit_depth_participation`.

**Edge Features (per qubit pair):**

| Feature | Description |
|---------|-------------|
| `interaction_count` | Number of 2-qubit gates between this qubit pair |
| `earliest_interaction` | Normalized time (0~1) of the first 2-qubit gate between this pair |
| `latest_interaction` | Normalized time (0~1) of the last 2-qubit gate between this pair |

**Edge feature design notes:**
- `interaction_count` is the primary feature: pairs with high counts must be placed close on the hardware to minimize SWAP overhead.
- `earliest_interaction` and `latest_interaction` together encode the temporal span of interactions (span = latest − earliest). Temporal features are less critical for *static* initial layout (which is fixed before execution) than for routing, but provide circuit structure context. The GNN can implicitly derive interaction density (`interaction_count / span`) from these three features.
- **Ablation candidate:** whether temporal features (`earliest`, `latest`) contribute beyond `interaction_count` alone.

**Multi-Programming Note:**

Multi-programming uses the same 4 node features as single-circuit. Circuit graphs are merged into a disconnected graph without additional features. The GNN naturally distinguishes circuits through disconnected components, and Sinkhorn prevents mapping conflicts. No per-circuit global summary features are used — this keeps single-circuit and multi-programming unified under the same input dimensions.

**Circuit Node Feature Normalization:** Z-score normalized **within each circuit** (across qubits, dim=0). This captures *relative* qubit importance within the circuit — which qubit is busier than others — rather than absolute counts. This is intentional: hardware features are also normalized within-backend, so the model learns to match "relatively busy logical qubit → relatively good physical qubit" in a consistent scale.

**Circuit Edge Feature Normalization:** Edge features (`interaction_count`, `earliest_interaction`, `latest_interaction`) are **not normalized**. They retain absolute values, providing the GNN with absolute-scale circuit complexity information that is lost from node features after within-circuit normalization.

**Trade-off:** Within-circuit normalization loses absolute circuit complexity. A 3-gate circuit and a 300-gate circuit look similar in node features if qubit busyness ratios are the same. However, `interaction_count` in edge features compensates partially — larger circuits naturally have higher interaction counts.

**Exception handling:** If standard deviation = 0 (all qubits identical), add ε = 1e-8 to denominator — node features become all-zero, which is uninformative but acceptable (uniform circuits have no strong placement preference).

**Known limitation — multi-programming cross-circuit scale mismatch:** In multi-programming, each circuit is normalized independently before merging. This means a qubit that is "relatively the busiest" in a simple 3-gate circuit and one that is "relatively the busiest" in a complex 300-gate circuit will have similar node feature values after normalization, even though the latter should receive higher-priority physical placement. The Score Head (cross-attention) cannot distinguish their true absolute importance. Edge features (`interaction_count`) partially compensate, but do not fully resolve the issue. Possible mitigations to consider if this proves impactful in practice:
- **Group-level normalization**: normalize node features jointly across all circuits in a multi-programming group (instead of per-circuit). Consistent scale across circuits, but requires online normalization at batch creation time and breaks the unified single/multi pipeline.
- **Log transform**: apply log(1+x) before z-score to compress scale differences while retaining some absolute information.
- **Global summary re-introduction**: add circuit-level complexity features as a context signal (reverts the 4-dim unification decision).

### 3.2 Hardware Graph

**Node:** Each physical qubit = one node.

**Edge:** Undirected. Edges from FakeBackendV2 coupling map.

**Node Features (per physical qubit):**

| Feature | Description | Direction |
|---------|-------------|-----------|
| `T1` | Energy relaxation time | Higher = better |
| `T2` | Phase relaxation time | Higher = better |
| `readout_error` | Measurement error rate | Lower = better |
| `single_qubit_error` | Average single-qubit gate error rate (sx, x) | Lower = better |
| `degree` | Coupling map connectivity degree | Structural info |
| `t1_cx_ratio` | T1 / mean_cx_duration — number of 2Q gates fitting within T1 | Higher = better |
| `t2_cx_ratio` | T2 / mean_cx_duration — number of 2Q gates fitting within T2 | Higher = better |

`t1_cx_ratio` and `t2_cx_ratio` are computed per qubit as T1 (or T2) divided by the mean cx_duration across all edges connected to that qubit. These composite features capture the **physically meaningful** quantity for layout: how many 2-qubit gate cycles the qubit can sustain before decoherence. Raw T1/T2 and cx_duration alone require the model to learn this division implicitly.

**Edge Features (per physical connection):**

| Feature | Description |
|---------|-------------|
| `cx_error` | 2-qubit gate error rate on this connection |

`cx_duration` removed: it is correlated with `cx_error` (slower gates tend to have higher error) and its information is absorbed into node-level `t1_cx_ratio` / `t2_cx_ratio`.

**Hardware Feature Normalization:** All features are **z-score normalized within each backend independently**. This ensures the model learns relative quality rankings within a hardware, enabling cross-hardware generalization.

**Exception handling:** Same ε = 1e-8 for zero standard deviation. If a qubit has no connected edges (isolated), mean_cx_duration defaults to 1.0 to avoid division by zero in ratio features.

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
Dummy Padding → h×h
Log-domain Sinkhorn(S/τ) → P (h×h)

[Training] P used directly for loss
[Inference] Hungarian(P) → discrete mapping (top l rows only)
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
- Circuit GNN input: `num_circuit_node_features` (4: gate_count, 2q_count, degree, depth_participation — same for single and multi-programming)
- Hardware GNN input: `num_hardware_node_features` (7: T1, T2, readout_err, sq_err, degree, t1_cx_ratio, t2_cx_ratio)
- Circuit edge features: 3 (interaction_count, earliest, latest)
- Hardware edge features: 1 (cx_error)

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

Learned projection from cross-attention output to mapping compatibility scores:

```
S_ij = (C'_i · W_q)^T · (H'_j · W_k) / √d_k
```

- `W_q`: learnable matrix d → d_k (d_k = 64)
- `W_k`: learnable matrix d → d_k (d_k = 64)
- Scaled by `√d_k` for numerical stability before Sinkhorn

**Output:** S matrix of shape (l × h)

### 4.5 Dummy Padding

Since l < h, append (h - l) dummy rows of zeros to S, creating an h×h square matrix.

**Dummy row initial values:** 0 (or small constant)

**Rationale for dummy padding over rectangular Sinkhorn:**
1. Standard Sinkhorn convergence guarantees preserved
2. Hungarian algorithm directly compatible (works on square matrices)
3. Gradient flow unaffected (dummy rows not connected to learnable params)
4. Simple implementation: `torch.nn.functional.pad`

### 4.6 Log-Domain Sinkhorn

**CRITICAL: Use log-domain implementation for numerical stability at low τ.**

```python
def log_sinkhorn(log_alpha, max_iter=20, tol=1e-6):
    """
    log_alpha: h×h matrix (= S_padded / τ, already in log domain)
    """
    for i in range(max_iter):
        # Row normalization in log domain
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
        # Column normalization in log domain
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)
        
        # Early stopping check
        if (log_alpha.exp().sum(dim=1) - 1).abs().max() < tol:
            break
    
    return log_alpha.exp()  # Return P matrix
```

**Sinkhorn Hyperparameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| τ_max (initial) | 1.0 | Soft distribution at Stage 1 start |
| τ_min (final) | 0.05 | Near one-hot; minimizes train-inference gap |
| τ Schedule | Exponential decay | `τ(epoch) = τ_max · (τ_min/τ_max)^(epoch/total_epochs)` |
| Stage 1 | Annealing from τ_max to τ_min | |
| Stage 2 | Fixed at τ_min = 0.05 | |
| Max Iterations | 20 | |
| Early Stop Tolerance | 1e-6 | |
| Implementation | Log-domain | Prevents overflow at low τ |

### 4.7 Hungarian Algorithm (Inference Only)

Applied to P matrix to produce discrete mapping. Only the top l rows (actual logical qubits) of the result are used. Dummy row assignments are discarded.

Use `scipy.optimize.linear_sum_assignment` with cost matrix `(1 - P)` (since Hungarian minimizes cost, and we want to maximize assignment probability).

---

## 5. Training Strategy: 2-Stage Curriculum

> **Note:** RL fine-tuning has been excluded from scope. The training consists of 2 stages only.

### 5.1 Stage 1: Supervised Pre-training

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

**Sinkhorn τ:** Annealed from τ_max=1.0 to τ_min=0.05 via exponential decay.

**MLQD + QUEKO → QUEKO-only Transition:**
- Criterion: Validation cross-entropy loss early stopping (patience = 5–10 epochs)
- When switching to QUEKO: reduce learning rate to 1/5–1/10 of previous LR
- QUEKO termination: same validation CE loss early stopping

### 5.2 Stage 2: Noise-Aware Surrogate Metric Fine-tuning

**Objective:** Fine-tune toward actual NISQ PST correlation without running the full transpile pipeline.

**Data:** All available circuits including QASMBench and RevLib (no labels required). Labeled datasets (MQT Bench, MLQD, QUEKO) can also be included for surrogate fine-tuning.

**Sinkhorn τ:** Fixed at 0.05.

**Optimizer:** AdamW with cosine annealing. Warm-up 3–5 epochs recommended.

#### Loss Term 1: L_surr — Error-Aware Edge Quality Loss

Uses error-accumulated shortest path distance instead of simple hop count:

```
d_error(p, q) = shortest_path_weighted(coupling_map, weight=cx_error)
```

Precompute using Floyd-Warshall with cx_error as edge weights. Store as h×h lookup matrix per backend.

```
L_surr = (1/|E_circuit|) · Σ_{(i,j)∈E_circuit} Σ_{p,q} P_ip · P_jq · d_error(p,q)
```

Where E_circuit = set of logical qubit pairs connected by 2-qubit gates.

Fully differentiable w.r.t. P (d_error is a precomputed constant).

#### Loss Term 2: L_node — NISQ Node Quality Loss

Drives important logical qubits to high-quality physical qubits.

**q_score function (learnable weighted linear combination):**

```python
q_score(p) = sigmoid(w1*T1_norm + w2*T2_norm + w3*(1-readout_err_norm) + w4*(1-sq_err_norm) + w5*freq_norm + bias)
```

- `w1–w5` and `bias` are **learnable parameters** (initialized to small positive values, e.g., 0.2)
- Features are z-score normalized within backend
- (1 - error) inversion ensures all terms are "higher = better" direction
- Sigmoid bounds output to [0, 1] for scale stability
- After training, inspecting w1–w5 provides insight into which noise factors matter most for PST

**Logical qubit importance w(i):** Number of 2-qubit gates involving qubit i (from circuit graph).

```
L_node = (1/l_total) · (-Σ_i w(i) · Σ_p P_ip · q_score(p))
```

Negative sign: higher quality assignment → lower loss.

#### Loss Term 3: L_sep — Multi-Programming Separation Loss

Encourages physical distance between qubits of different circuits (crosstalk reduction).

```
L_sep = (1/|E_cross|) · (-Σ_{(i,j)∈cross-circuit} Σ_{p,q} P_ip · P_jq · d_hw(p,q))
```

Where E_cross = all pairs (i, j) where i and j belong to different circuits.

**Automatically equals 0 for single-circuit scenarios** (no cross-circuit pairs exist).

#### Combined Stage 2 Loss

```
L_2 = L_surr + α · L_node + λ · L_sep
```

| Weight | Value | Rationale |
|--------|-------|-----------|
| L_surr coefficient | 1.0 (fixed) | Primary objective, baseline reference |
| α (L_node) | 0.1 | Important for NISQ but secondary to distance |
| λ (L_sep) | 0.1 | Most indirect effect |

**CRITICAL: All three terms are per-pair/per-qubit normalized** (divided by pair count or qubit count). This ensures scales are comparable regardless of circuit/hardware size.

**Verification:** Log each term during training. No single term should dominate by >10×. Grid search candidates: α ∈ {0.1, 0.3, 0.5}, λ ∈ {0.05, 0.1, 0.2}.

### 5.3 Stage Transition Criteria

| Transition | Criterion | Details |
|------------|-----------|---------|
| MLQD + QUEKO → QUEKO-only | Validation CE loss early stopping | Patience 10 epochs; LR reduced to 1/10 |
| Stage 1 → Stage 2 | Validation PST convergence | Measure actual PST every 5–10 epochs on 50–100 representative circuits (Hungarian → transpile → noise sim); stop when 3 consecutive measurements improve < 0.5% |
| Stage 2 termination | Validation PST early stopping | Based on actual PST, NOT surrogate loss; surrogate loss improving while PST degrades = overfitting signal |

**Validation PST measurement procedure:**
1. Take P matrix from model
2. Apply Hungarian → discrete layout
3. Run `qiskit.transpile(initial_layout=layout, routing_method=configurable)`
4. Run noise simulation on FakeBackendV2
5. Compute PST

---

## 6. Dataset Usage

### Available Datasets

| Dataset | Volume | Stage | Label Source | Notes |
|---------|--------|-------|--------------|-------|
| **MQT Bench** | 1,219 circuits | Stage 2 (Unsupervised) | None | 29 algorithm types, 2-127Q. No mapped labels available. Generated via `scripts/generate_mqt_bench.py` |
| **MLQD** | 4,443 circuits (3,729 labeled) | Stage 1 (Supervised) | OLSQ2 solver labels (extracted) | 5 backends: Aspen-4, Grid5x5, Melbourne, Rochester, Sycamore |
| **QUEKO** | 900 circuits (540 labeled) | Stage 1 (Supervised, fine-tuning) | τ⁻¹ (true optimal) | 3 categories: BNTF (180), BSS (360, labeled), BIGD (360, no labels). 4 hardware topologies |
| **QASMBench** | 94 circuits | Stage 2 (Unsupervised) | None | 2Q-127Q filtered. No labels. Surrogate loss training |
| **RevLib** | 231 circuits | Stage 2 (Unsupervised) | None | Reversible circuits, 3Q-127Q. Converted from .real via Real2QASM |

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
    ├── stage1_supervised.json       # 3,846 labeled circuits → Stage 1 training
    ├── stage1_queko_only.json       # 486 QUEKO circuits → Stage 1 fine-tuning phase
    ├── stage1_unsupervised.json     # 2,618 unlabeled circuits → Stage 2 only
    ├── stage2_all.json              # 6,887 all circuits → Stage 2 surrogate loss
    ├── val.json                     # 423 labeled validation
    └── val_queko_only.json          # 54 QUEKO validation
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

**Preprocessing summary:**

| Step | Circuits Removed | Reason |
|------|:----------------:|--------|
| Gate normalization | 0 (in-place) | Standardize gate representation |
| Untranspilable | 34 | OOM / timeout during transpile |
| Benchmark dedup | 19 | Evaluation fairness |
| Extreme filtering | 183 | edges > 1,000, GNN scalability |
| **Total removed** | **236** | |

**Original → Final circuit counts:**

| Dataset | Original | Final | Removed |
|---------|:--------:|:-----:|:-------:|
| QUEKO | 900 | 900 | 0 |
| MLQD | 4,443 | 4,443 | 0 |
| MQT Bench | 1,448 | 1,219 | 229 |
| QASMBench | 111 | 94 | 17 |
| RevLib | 263 | 231 | 32 |
| **Total** | **7,165** | **6,887** | **278** |

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

**Stage 2 (Unsupervised):** Fine-tune with surrogate losses (L_surr, L_node, L_sep) on all available circuits, including label-free datasets (MQT Bench, QASMBench, RevLib). This stage aligns the model toward SABRE routing and NISQ-aware PST optimization, compensating for any mismatch between existing labels and the SABRE pipeline.

### Rationale for Using Existing Labels

- **MLQD OLSQ2 labels** were optimized for the OLSQ2 routing pipeline, but still encode meaningful mapping quality signal (e.g., minimizing qubit interaction distance)
- **QUEKO τ⁻¹ labels** are true topology-optimal mappings (zero-SWAP overhead) — the highest quality supervised signal available
- **MQT Bench** was initially considered for pseudo-labels but excluded because mapped-level data is effectively unavailable from MQT Bench web/API
- **Stage 2 unsupervised fine-tuning corrects router-specific bias** — L_surr directly optimizes error-weighted distance aligned with SABRE routing
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

1. Construct individual circuit graphs for each circuit (4-dim node features, same as single-circuit)
2. Merge into a single disconnected graph (no edges between circuits)
3. Total node count = l₁ + l₂ + ... (sum of all circuits' logical qubits)

---

## 8. Batching Strategy

### GNN Level

Use **PyTorch Geometric standard batching**: multiple graphs merged into a single disconnected graph with batch index tracking. Handles variable node counts without padding natively.

### Score Matrix Level: Backend-Based Bucketing

**Samples using the same hardware backend are grouped into the same mini-batch.** This ensures:
- All samples in a batch have identical h (physical qubit count)
- Score Matrix / Sinkhorn / Loss computed as 3D tensor `(batch_size × h × h)` in parallel
- Dummy padding amount varies per sample but matrix size is uniform

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

**Simulation method priority:** tensor_network+GPU → tensor_network+CPU → statevector (small circuits only)
**Default shots:** 8192

**Robustness:** On large backends (100Q+), model-generated layouts may produce very deep transpiled circuits that crash the tensor_network simulator (`CUTENSORNET_STATUS_INVALID_VALUE`). This corrupts GPU state for subsequent simulations. Mitigations:
- **Evaluation order:** Baselines run before model evaluation per circuit, so baseline results are never affected by model failures.
- **Simulator recovery:** On simulation failure, simulators are recreated and the failed run is recorded as NaN (excluded from averaging via `nanmean`).

### Transpilation

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

Per-stage timing measured: init, layout, routing, optimization, scheduling.

### Benchmark Circuits

Standard evaluation set (shared with MQM colleague for direct comparison):
`toffoli_3`, `fredkin_3`, `3_17_13`, `4mod5-v1_22`, `mod5mils_65`, `alu-v0_27`, `decod24-v2_43`, `4gt13_92`

Extended set adds: `bv_n3`, `bv_n4`, `peres_3`, `xor5_254`

All benchmark circuits stored in `data/circuits/qasm/benchmarks/` (23 total .qasm files, gate-normalized to `{cx, id, rz, sx, x}`).
For fair comparison with MQM colleague, use original (non-normalized) circuits via `--circuit-dir references/colleague/tests2/benchmarks`. Gate normalization reduces basis translation overhead, which inflates PST relative to the colleague's results.

### Statistical Reliability

Multiple repetitions per experiment (matching baseline research), reporting mean and standard deviation.
Results presented as pandas DataFrame with per-circuit PST, depth, CX count, timing, and Avg row.

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
| Hop count vs. Error-aware distance | L_surr definition; supports NISQ-aware claim |
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
| L_sep presence/absence | Multi-programming separation effect |
| Noise features in Hardware GNN | With vs. without calibration features |
| Noise parameter perturbation | Robustness to calibration drift |

---

## 11. Open Items

These items are not yet finalized and need to be decided during implementation:

### Optimizer Details (Decided)
- **Confirmed:** AdamW + Cosine Annealing LR scheduler
- **Stage 1 LR:** 1e-3, weight decay 1e-4, cosine eta_min 1e-6
- **Stage 2 LR:** 1e-4, weight decay 1e-4, cosine eta_min 1e-6, warmup 5 epochs
- **Gradient clipping (Stage 2):** max_norm 0.5

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
# T1, T2, readout_error, single-qubit gate errors, t1_cx_ratio, t2_cx_ratio
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
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(5) * 0.2)  # w1-w5
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, node_features):
        """
        node_features: (h, 5) tensor
        Columns: [T1_norm, T2_norm, (1-readout_err_norm), (1-sq_err_norm), freq_norm]
        All pre-normalized within backend (z-score), with errors inverted
        """
        score = torch.sigmoid((node_features * self.weights).sum(dim=-1) + self.bias)
        return score  # shape: (h,), values in [0, 1]
```

### Sinkhorn with Dummy Padding

```python
def forward(self, S, l, h, tau):
    """
    S: (batch, l, h) raw score matrix
    Returns: P (batch, h, h) doubly stochastic matrix
    """
    # Dummy padding
    dummy = torch.zeros(S.shape[0], h - l, h, device=S.device)
    S_padded = torch.cat([S, dummy], dim=1)  # (batch, h, h)
    
    # Log-domain Sinkhorn
    log_alpha = S_padded / tau
    P = log_sinkhorn(log_alpha, max_iter=20, tol=1e-6)
    return P
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
| **Sinkhorn** | τ_max | 1.0 |
| | τ_min | 0.05 |
| | Schedule | Exponential decay (Stage 1), Fixed (Stage 2) |
| | Max Iterations | 20 |
| | Tolerance | 1e-6 |
| **Stage 2 Loss** | L_surr weight | 1.0 |
| | α (L_node) | 0.1 |
| | λ (L_sep) | 0.1 |
| **Batching** | Max total nodes | 512 (tune to GPU) |
| **Multi-prog** | Scenarios | Configurable (default: [1, 2, 4] with proportions [0.5, 0.3, 0.2]) |
| **Labels** | Stage 1 sources | MLQD (OLSQ2, 3,729), QUEKO (τ⁻¹, 540) |
| | Stage 2 (unsupervised) | MQT Bench, QASMBench, RevLib (+ all Stage 1 circuits) |
| **Optimizer** | Type | AdamW |
| | Weight Decay | 1e-4 |
| | LR (Stage 1) | 1e-3, reduced to 1e-4 for QUEKO fine-tuning (lr_factor=0.1) |
| | LR (Stage 2) | 1e-4 |
| | LR Scheduler | Cosine Annealing (eta_min=1e-6) |
| | Warmup (Stage 2) | 5 epochs |
| | Grad Clip (Stage 2) | max_norm 0.5 |
| **Transitions** | MLQD+QUEKO→QUEKO | Val CE early stop, patience 10 |
| | QUEKO fine-tuning end | Val CE early stop, patience 5 |
| | Stage 1→2 | Manual (after Stage 1 completes) |
| | Stage 2 end | Val PST early stopping, patience 10, min_delta 0.5% |
| **Reproducibility** | Training seed | 42 |
| | Evaluation seed | 43 |
