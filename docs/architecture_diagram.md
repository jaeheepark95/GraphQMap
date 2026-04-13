# GraphQMap Model Architecture

Latest experiment settings: `v7_norefine_tau05_s42` (2026-04-13)

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
│    ├─ earliest_interact (1)  │                                              │
│    └─ latest_interact   (1)  │                                              │
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
                      S (B, l, h)
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Sinkhorn Normalization (τ = 0.5)                    │
│                                                                  │
│  1. Pad: S (B,l,h) → S_pad (B,h,h)  [dummy rows = h-l zeros]  │
│  2. Scale: log_α = S_pad / τ                                    │
│  3. Log-domain Sinkhorn (max 20 iter, tol 1e-6):               │
│     repeat:                                                      │
│       log_α -= logsumexp(log_α, dim=-1)   [row normalize]      │
│       log_α -= logsumexp(log_α, dim=-2)   [col normalize]      │
│     until converged                                              │
│  4. P = exp(log_α)  →  doubly stochastic (B, h, h)             │
│  5. Slice: P[:, :l, :]  →  (B, l, h)                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
         [Training]               [Inference]
              │                         │
              ▼                         ▼
┌──────────────────────┐  ┌──────────────────────────┐
│    QAP Fidelity Loss │  │   Hungarian Algorithm     │
│                      │  │                            │
│  L_qap = L_edge      │  │  cost = 1 - P             │
│        + L_readout   │  │  scipy.linear_sum_assign.  │
│                      │  │         ↓                  │
│  L_edge:             │  │  {logical → physical}      │
│    tr(Ã_c·P·C_eff·  │  │  discrete 1-to-1 mapping   │
│       P^T) / Σg     │  └──────────────────────────┘
│                      │
│  L_readout:          │
│    1^T·P·ε_r / l    │
│                      │
│  Ã_c: gate-weighted  │
│    circuit adjacency │
│  C_eff: Floyd-Warsh. │
│    3×ε₂ SWAP cost    │
│  ε_r: readout errors │
└──────────────────────┘
```

---

## Training Configuration Summary

| Component | Setting |
|-----------|---------|
| **Circuit features** | gc, 2qc, sqr, cpf + RWPE k=2 = **6dim** |
| **Circuit edges** | interaction_count, earliest, latest = **3dim** |
| **HW node features** | readout, sq_err, degree, t1_cx, t2_cx = **5dim** |
| **HW edge features** | 2q_gate_error = **1dim** |
| **Embedding dim** | 64 |
| **GNN** | GATv2, 3 layers, 4 heads (16dim/head), residual on L2-L3 |
| **Cross-Attention** | 2 layers, 4 heads, FFN 128dim |
| **Score Head** | d_k=64, no noise bias |
| **Score Norm** | Sinkhorn (log-domain, 20 iter, dummy-padded) |
| **Temperature** | τ = 0.5 (fixed, no annealing) |
| **Iterative Refinement** | disabled (iterations=0) |
| **Loss** | QAP Fidelity (per-term normalized) |
| **Optimizer** | AdamW, lr=5e-4, wd=1e-4 |
| **Scheduler** | CosineAnnealing, T_max=100, η_min=1e-5 |
| **Warmup** | 2 epochs |
| **Grad clip** | 2.0 |
| **Max epochs** | 100 |
| **Checkpoint** | best val PST (every 5 epochs) |
| **Backend (train)** | FakeToronto only |
| **Backend (val)** | FakeMumbai, FakeManhattan |
| **Dataset** | curated split (stage2_curated.json) |

---

## Data Flow Dimensions

```
Circuit:  x_c (l, 6) ──→ Linear ──→ (l, 64) ──→ GATv2×3 ──→ (l, 64) ──→ C
                                                                              │
Hardware: x_hw(h, 5) ──→ Linear ──→ (h, 64) ──→ GATv2×3 ──→ (h, 64) ──→ H │
                                                                              │
Cross-Attn ×2:  C(B,l,64), H(B,h,64) ──→ C'(B,l,64), H'(B,h,64)           │
                                                                              │
Score:   Q=W_q·C'(B,l,64), K=W_k·H'(B,h,64) ──→ S = QK^T/√64 (B,l,h)     │
                                                                              │
Sinkhorn: S(B,l,h) → pad(B,h,h) → doubly stochastic → slice → P(B,l,h)     │
                                                                              │
Inference: P(B,l,h) → Hungarian → layout {0→p₀, 1→p₁, ..., (l-1)→p_{l-1}} │
```
