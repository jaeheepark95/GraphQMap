# GraphQMap Model Architecture (with Iterative Refinement)

QAP Mirror Descent refinement enabled. Based on `stage2_toronto_qap_refine.yaml` with `iterations: 5`.

---

## Full Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT                                          │
├──────────────────────────────┬──────────────────────────────────────────────┤
│     Circuit Graph (G_c)      │        Hardware Graph (G_hw)                 │
│                              │                                              │
│  Nodes: l logical qubits     │  Nodes: h physical qubits                   │
│    x_c: 6dim                 │    x_hw: 5dim                               │
│    ├─ gate_count        (1)  │    ├─ readout_error      (1) z-score        │
│    ├─ two_qubit_gate_ct (1)  │    ├─ single_qubit_error (1) z-score        │
│    ├─ single_qubit_ratio(1)  │    ├─ degree             (1) z-score        │
│    ├─ critical_path_frac(1)  │    ├─ t1_cx_ratio        (1) z-score        │
│    └─ RWPE [M², M³]    (2)  │    └─ t2_cx_ratio        (1) z-score        │
│        (not z-scored)        │                                              │
│                              │  Edges: physical couplings                   │
│  Edges: 2Q gate interactions │    e_hw: 1dim                               │
│    e_c: 3dim (z-scored)      │    └─ 2q_gate_error      (1) z-score        │
│    ├─ interaction_count (1)  │                                              │
│    ├─ earliest_interact (1)  │  Precomputed matrices:                       │
│    └─ latest_interact   (1)  │    C_eff (h×h): Floyd-Warshall, 3×ε₂ edges │
│                              │    ε_r   (h):   readout error per qubit      │
│  Precomputed:                │                                              │
│    Ã_c (l×l): gate-weighted  │                                              │
│      circuit adjacency       │                                              │
└──────────────┬───────────────┴──────────────────┬───────────────────────────┘
               │                                  │
               ▼                                  ▼
┌──────────────────────────────┐  ┌──────────────────────────────┐
│      Circuit GNN Encoder     │  │     Hardware GNN Encoder      │
│      (independent params)    │  │     (independent params)      │
│                              │  │                                │
│  Linear(6 → 64)             │  │  Linear(5 → 64)               │
│  EdgeProj: Linear(3 → 64)   │  │  EdgeProj: Linear(1 → 64)    │
│         │                    │  │         │                      │
│  ┌──────▼──────┐             │  │  ┌──────▼──────┐              │
│  │ GATv2 L1    │             │  │  │ GATv2 L1    │              │
│  │ 4 heads     │             │  │  │ 4 heads     │              │
│  │ head_dim=16 │             │  │  │ head_dim=16 │              │
│  └──────┬──────┘             │  │  └──────┬──────┘              │
│    BatchNorm → ELU           │  │    BatchNorm → ELU            │
│         │                    │  │         │                      │
│  ┌──────▼──────┐             │  │  ┌──────▼──────┐              │
│  │ GATv2 L2    │◄── residual │  │  │ GATv2 L2    │◄── residual  │
│  └──────┬──────┘             │  │  └──────┬──────┘              │
│    BatchNorm → ELU + skip    │  │    BatchNorm → ELU + skip     │
│         │                    │  │         │                      │
│  ┌──────▼──────┐             │  │  ┌──────▼──────┐              │
│  │ GATv2 L3    │◄── residual │  │  │ GATv2 L3    │◄── residual  │
│  └──────┬──────┘             │  │  └──────┬──────┘              │
│    BatchNorm → ELU + skip    │  │    BatchNorm → ELU + skip     │
│         │                    │  │         │                      │
│  Linear(64 → 64)             │  │  Linear(64 → 64)              │
└──────────┬───────────────────┘  └──────────┬────────────────────┘
           │                                  │
           ▼                                  ▼
      C (B, l, 64)                       H (B, h, 64)
           │                                  │
           └──────────────┬───────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              Bidirectional Cross-Attention (×2 layers)           │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Layer k (k = 1, 2):                                      │  │
│  │                                                            │  │
│  │  ① Circuit attends to Hardware:                           │  │
│  │     attn_c = MHA(Q=C, K=H, V=H)   [4 heads, d=64]       │  │
│  │     C = LayerNorm(C + attn_c)                             │  │
│  │     C = LayerNorm(C + FFN(C))      [64→128→64, ELU]      │  │
│  │                                                            │  │
│  │  ② Hardware attends to Circuit:                           │  │
│  │     attn_h = MHA(Q=H, K=C, V=C)   [4 heads, d=64]       │  │
│  │     H = LayerNorm(H + attn_h)                             │  │
│  │     H = LayerNorm(H + FFN(H))      [64→128→64, ELU]      │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
                  C' (B, l, 64),  H' (B, h, 64)
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Score Head                                │
│                                                                  │
│   Q = W_q · C'    (B, l, 64)     W_q: Linear(64→64, no bias)   │
│   K = W_k · H'    (B, h, 64)     W_k: Linear(64→64, no bias)   │
│                                                                  │
│   S = (Q · K^T) / √64            (B, l, h)                     │
│                                                                  │
│   noise_bias_dim = 0  →  no noise bias added                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
                      S (B, l, h)    ← "raw GNN score"
                           │
                           ▼
┌═════════════════════════════════════════════════════════════════════════════┐
║           ITERATIVE REFINEMENT  (QAP Mirror Descent, T=5 steps)           ║
║                                                                            ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │  Step 0: Z-score normalize raw scores                               │  ║
║  │                                                                      │  ║
║  │  S_init = (S - μ_S) / σ_S          μ_S, σ_S detached (no grad)     │  ║
║  │                                                                      │  ║
║  │  Purpose: GNN produces S with mean~38, std~24, range [11,149].      │  ║
║  │  Feedback has range [0, 0.4]. Without normalization the ratio is    │  ║
║  │  ~300× → feedback has zero effect. After: S_init ~ N(0,1),         │  ║
║  │  feedback/score ratio ~ 1:3 → meaningful refinement.                │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                           │                                                ║
║                           ▼                                                ║
║  ┌──────────────────────────────────────────────────────────────────────┐  ║
║  │  Temperature schedule:                                               │  ║
║  │                                                                      │  ║
║  │  τ_start = τ_epoch / β^(T-1)     (high → allows exploration)       │  ║
║  │  τ_t *= β  each iteration         (β=0.9, anneals toward τ_epoch)  │  ║
║  │                                                                      │  ║
║  │  Example (τ_epoch=0.5, T=5, β=0.9):                                │  ║
║  │    t=0: τ=0.762  →  t=1: τ=0.686  →  t=2: τ=0.617                 │  ║
║  │    t=3: τ=0.556  →  t=4: τ=0.500  (= τ_epoch)                     │  ║
║  └──────────────────────────────────────────────────────────────────────┘  ║
║                           │                                                ║
║                           ▼                                                ║
║            S_current = S_init                                              ║
║                           │                                                ║
║  ╔════════════════════════╧═══════════════════════════════════════════╗    ║
║  ║  for t = 0, 1, ..., T-1:        (mirror descent loop)            ║    ║
║  ║                                                                    ║    ║
║  ║  ┌─────────────────────────────────────────────────────────────┐  ║    ║
║  ║  │  ① Soft assignment at current temperature                  │  ║    ║
║  ║  │                                                             │  ║    ║
║  ║  │  P^(t) = Sinkhorn(S_current, τ_t)        (B, l, h)        │  ║    ║
║  ║  │          ├─ pad to (B,h,h), log-domain, max 20 iter        │  ║    ║
║  ║  │          └─ slice back to (B,l,h)                          │  ║    ║
║  ║  └─────────────────────────────────────────────────────────────┘  ║    ║
║  ║                          │                                        ║    ║
║  ║                          ▼                                        ║    ║
║  ║  ┌─────────────────────────────────────────────────────────────┐  ║    ║
║  ║  │  ② Compute expected cost landscape                         │  ║    ║
║  ║  │                                                             │  ║    ║
║  ║  │  Z = P^(t) @ C_eff                        (B, l, h)       │  ║    ║
║  ║  │                                                             │  ║    ║
║  ║  │  Z_ij = Σ_k P^(t)_ik · C_eff_kj                          │  ║    ║
║  ║  │  = "expected cost from qubit i to reach physical qubit j   │  ║    ║
║  ║  │    given current soft assignment P^(t)"                    │  ║    ║
║  ║  └─────────────────────────────────────────────────────────────┘  ║    ║
║  ║                          │                                        ║    ║
║  ║                          ▼                                        ║    ║
║  ║  ┌─────────────────────────────────────────────────────────────┐  ║    ║
║  ║  │  ③ QAP gradient feedback                                   │  ║    ║
║  ║  │                                                             │  ║    ║
║  ║  │  feedback = Ã_c @ Z                        (B, l, h)      │  ║    ║
║  ║  │                                                             │  ║    ║
║  ║  │  feedback_ij = Σ_k Ã_c_ik · Z_kj                         │  ║    ║
║  ║  │  = "total interaction-weighted cost of placing qubit i at  │  ║    ║
║  ║  │    physical qubit j, considering ALL neighbors' current    │  ║    ║
║  ║  │    placements"                                             │  ║    ║
║  ║  │                                                             │  ║    ║
║  ║  │  This equals ½ · ∂L_QAP/∂P — the analytical QAP gradient  │  ║    ║
║  ║  └─────────────────────────────────────────────────────────────┘  ║    ║
║  ║                          │                                        ║    ║
║  ║                          ▼                                        ║    ║
║  ║  ┌─────────────────────────────────────────────────────────────┐  ║    ║
║  ║  │  ④ Update score (subtract cost → high score = low cost)    │  ║    ║
║  ║  │                                                             │  ║    ║
║  ║  │  S_current = S_init - λ · feedback         (B, l, h)      │  ║    ║
║  ║  │                                                             │  ║    ║
║  ║  │  λ: learnable scalar (init=1.0, nn.Parameter)             │  ║    ║
║  ║  │  Always resets from S_init (not S_current) — stable       │  ║    ║
║  ║  └─────────────────────────────────────────────────────────────┘  ║    ║
║  ║                          │                                        ║    ║
║  ║                          ▼                                        ║    ║
║  ║                    τ_t *= β  (cool temperature)                   ║    ║
║  ║                          │                                        ║    ║
║  ╚══════════════════════════╧════════════════════════════════════════╝    ║
║                             │                                              ║
║                             ▼                                              ║
║                     P = P^(T-1)   (final iteration's output)              ║
║                                                                            ║
║  Performance floor guarantee:                                              ║
║    Even if GNN's S_init is random noise, the analytical feedback           ║
║    term (Ã_c · P · C_eff) still optimizes QAP → valid layout.            ║
╚═══════════════════════════════╤════════════════════════════════════════════╝
                                │
                   P (B, l, h)  │
                                │
               ┌────────────────┴────────────┐
               │                             │
          [Training]                   [Inference]
               │                             │
               ▼                             ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│    QAP Fidelity Loss      │  │   Hungarian Algorithm     │
│                            │  │                            │
│  L_qap = L_edge + L_read  │  │  cost = 1 - P              │
│                            │  │  scipy.linear_sum_assign.  │
│  L_edge:                   │  │         ↓                  │
│    tr(Ã_c·P·C_eff·P^T)   │  │  {logical → physical}      │
│    ─────────────────       │  │  discrete 1-to-1 mapping   │
│          Σg                │  └──────────────────────────┘
│                            │
│  L_readout:                │
│    1^T · P · ε_r           │
│    ────────────            │
│        l                   │
│                            │
│  Ã_c: gate-weighted        │
│    circuit adjacency (l×l) │
│  C_eff: Floyd-Warshall     │
│    3×ε₂ SWAP cost (h×h)   │
│  ε_r: readout errors (h)   │
└──────────────────────────┘
```

---

## Iterative Refinement: Conceptual View

```
          GNN + CrossAttn                    Analytical QAP Gradient
         (learned component)              (precomputed, no parameters)
               │                                     │
               ▼                                     ▼
          S_init (B,l,h)             feedback = Ã_c @ (P^(t) @ C_eff)
          "where does the                  "where SHOULD qubit i go,
           GNN think qubit i               given current P and the
           should go?"                     QAP cost structure?"
               │                                     │
               └──────────────┬──────────────────────┘
                              ▼
                  S = S_init  -  λ · feedback
                              │
                              ▼
                   Sinkhorn(S/τ_t) → P^(t+1)
                              │
                              ▼
                      repeat T times
                      (τ_t decreasing)
```

**Key insight**: The refinement loop interprets each Sinkhorn step as **mirror descent
on the QAP relaxation**. The GNN provides an initial guess; the analytical gradient
corrects it iteratively. Even a poor GNN can produce good layouts because the
gradient term alone solves the QAP.

---

## Configuration Comparison

| Component | No Refinement | With Refinement |
|-----------|:-------------:|:---------------:|
| **GNN Encoders** | same | same |
| **Cross-Attention** | same | same |
| **Score Head** | same | same |
| **Score Normalization** | Sinkhorn ×1 | z-norm → Sinkhorn ×T |
| **Sinkhorn calls** | 1 | T (one per iteration) |
| **Extra learnable params** | 0 | 1 (λ scalar) |
| **Extra precomputed data** | - | C_eff (h×h), Ã_c (l×l) |
| **Temperature** | τ = 0.5 fixed | τ anneals: τ/β^(T-1) → τ |
| **Iterations (T)** | 0 | 5 |
| **β (temp decay)** | - | 0.9 |
| **λ_init** | - | 1.0 (learnable) |

---

## Precomputed Matrices

```
C_eff (h × h) — Effective cost matrix
┌─────────────────────────────────────────────────────────────┐
│  Floyd-Warshall shortest path on hardware graph              │
│  Edge weight = 3 × ε₂(p,q)    (each SWAP = 3 CX gates)    │
│  Adjacent pairs: overwritten with raw ε₂(p,q)               │
│  C_eff[i][j] = total -log(fidelity) cost to route i↔j      │
│  Precomputed once per backend, reused across all circuits    │
└─────────────────────────────────────────────────────────────┘

Ã_c (l × l) — Gate-count weighted circuit adjacency
┌─────────────────────────────────────────────────────────────┐
│  Ã_c[i][j] = number of 2Q gates between logical qubit i,j  │
│  Built from circuit_edge_pairs + circuit_edge_weights        │
│  Dense matrix, constructed in collation per batch            │
│  Ã_c[i][j] = 0  if no interaction between qubits i and j   │
└─────────────────────────────────────────────────────────────┘

ε_r (h) — Readout error vector
┌─────────────────────────────────────────────────────────────┐
│  ε_r[j] = measurement error rate of physical qubit j        │
│  From backend noise model (FakeBackendV2 / synthetic JSON)  │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow Dimensions (with Refinement)

```
Circuit:  x_c (l, 6) ──→ Linear ──→ (l, 64) ──→ GATv2×3 ──→ (l, 64) ──→ C
                                                                              │
Hardware: x_hw(h, 5) ──→ Linear ──→ (h, 64) ──→ GATv2×3 ──→ (h, 64) ──→ H │
                                                                              │
Cross-Attn ×2:  C(B,l,64), H(B,h,64)  ──→  C'(B,l,64), H'(B,h,64)         │
                                                                              │
Score:   Q=W_q·C'(B,l,64), K=W_k·H'(B,h,64)  ──→  S = QK^T/√64 (B,l,h)   │
                                                                              │
Z-norm:  S_init = (S - μ) / σ                       (B,l,h)  ~N(0,1)        │
                                                                              │
Refine (×T):                                                                  │
  P^(t) = Sinkhorn(S_current, τ_t)                  (B,l,h)                  │
  Z     = P^(t) @ C_eff                             (B,l,h)                  │
  fb    = Ã_c @ Z                                   (B,l,h)  [Ã_c broadcast]│
  S_cur = S_init - λ·fb                             (B,l,h)                  │
                                                                              │
Output:  P = P^(T-1)                                 (B,l,h)                  │
                                                                              │
Inference: P(B,l,h)  →  Hungarian  →  {0→p₀, 1→p₁, ..., (l-1)→p_{l-1}}    │
```
