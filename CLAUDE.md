# CLAUDE.md

## Project Overview
**GraphQMap** — Hardware-agnostic quantum qubit mapping ML model.
A single GNN-based model that outputs initial layouts for quantum circuit compilation on NISQ hardware.
Supports arbitrary multi-programming (any number of co-located circuits, total logical qubits ≤ physical qubits).

## Tech Stack
- Python 3.10+, PyTorch, PyTorch Geometric (PyG)
- Qiskit >= 1.0, qiskit-ibm-runtime (FakeBackendV2), qiskit-aer
- scipy (Hungarian algorithm, Floyd-Warshall)
- networkx

## Project Structure
- `models/` — GNN encoders, cross-attention, score head, score normalization (SoftmaxNorm / SinkhornLayer), hungarian
- `data/` — dataset loaders, graph construction, label generation, normalization
- `training/` — training loops, loss functions, tau scheduler, early stopping
- `evaluation/` — PST measurement, metrics, baselines, transpiler, benchmarks
  - `evaluation/transpiler.py` — custom PassManager builder (layout×routing combinations, per-stage timing, noise-aware UnitarySynthesis)
  - `evaluation/benchmark.py` — structured benchmark runner (DataFrame output, simulator reuse)
  - `evaluation/prev_methods/` — baseline routing/layout methods (NASSC, NoiseAdaptive, QAP)
- `configs/` — hyperparameter configs (YAML) and config loader (`config_loader.py`)
- `runs/` — experiment outputs (timestamped, gitignored): checkpoints, metrics CSV, config snapshots, note.md, EXPERIMENTS.md
- `scripts/` — dataset generation/processing, visualization, diagnostics
  - `visualize.py` — training/eval plotting (auto-generated + manual comparison)
  - `diagnose_features.py` — circuit node feature quality diagnostics (effective dim, cosine sim, correlations)
  - `visualize_layouts.py` — initial layout visualization on backend topology (SABRE vs model vs QAP)
  - `visualize_backends.py` — test backend topology/error map visualization
  - `compare_pst.py` — PST comparison across layout methods
  - `benchmark_feature_analysis.py` — per-circuit feature analysis (variance, correlation, dimensionality)
- `docs/` — research specification and documentation
- `tests/` — unit and integration tests (151 tests)

## Dataset Structure
All circuit data lives under `data/circuits/` with circuits, labels, backends, and splits separated.

```
data/circuits/
├── qasm/                           # 6,910 total .qasm files (all normalized to basis gates)
│   ├── queko/                      # 900 circuits (540 labeled, 360 unlabeled)
│   ├── mlqd/                       # 4,443 circuits (3,729 labeled)
│   ├── mqt_bench/                  # 1,219 circuits (no labels)
│   ├── qasmbench/                  # 94 circuits (no labels, 2Q-127Q)
│   ├── revlib/                     # 231 circuits (no labels, 3Q-127Q)
│   └── benchmarks/                 # 23 evaluation benchmark circuits (3Q-13Q)
├── labels/                         # 4,269 total labels
│   ├── queko/labels.json           # 540 τ⁻¹ true optimal labels
│   └── mlqd/labels.json            # 3,729 OLSQ2 solver labels
├── backends/                       # 5 synthetic noise profiles (JSON)
│   ├── queko_aspen4.json           # Rigetti Aspen-4 (16Q)
│   ├── queko_tokyo.json            # IBM Tokyo (20Q)
│   ├── queko_rochester.json        # IBM Rochester (53Q)
│   ├── queko_sycamore.json         # Google Sycamore (54Q)
│   └── mlqd_grid5x5.json          # 5x5 Grid (25Q)
└── splits/                         # Train/val split definitions
    ├── stage1_supervised.json      # 3,599 labeled circuits for Stage 1
    ├── stage1_queko_only.json      # 483 QUEKO circuits for fine-tuning
    ├── stage1_unsupervised.json    # 1,774 unlabeled circuits
    ├── stage2_all.json             # 5,769 all circuits for Stage 2
    ├── val.json                    # 396 validation (labeled)
    ├── val_queko_only.json         # 52 QUEKO validation
    ├── filter_log.json             # Feature-indistinguishable removal log
    └── original/                   # Pre-filter backup of all splits
```

### Dataset Sources & Labels
| Dataset | Circuits | Labels | Stage | Label Source | Backend Mapping |
|---------|:--------:|:------:|-------|--------------|-----------------|
| QUEKO | 900 | 540 | Stage 1 + 2 | τ⁻¹ optimal (zero-SWAP) | 4 synthetic backends |
| MLQD | 4,443 | 3,729 | Stage 1 + 2 | OLSQ2 solver (extracted) | melbourne, rochester (Qiskit) + 3 synthetic |
| MQT Bench | 1,219 | 0 | Stage 2 | None | Assigned randomly to training backends |
| QASMBench | 94 | 0 | Stage 2 | None | Assigned randomly to training backends |
| RevLib | 231 | 0 | Stage 2 | None | Assigned randomly to training backends |

### Dataset Preprocessing
Raw datasets are preprocessed before training (details in `docs/RESEARCH_SPEC.md`):
1. **Gate normalization** — all QASM files transpiled to `{cx, id, rz, sx, x}` basis via `scripts/normalize_gates.py`
2. **Untranspilable removal** — 34 circuits removed (OOM/timeout during transpile or QASM parsing)
3. **Benchmark deduplication** — 19 circuits removed from training sets (overlap with evaluation benchmarks)
4. **Extreme circuit filtering** — 183 circuits with edges > 1,000 removed (GNN scalability)
5. **Feature-indistinguishable filtering** — 1,118 circuits removed where node features cannot distinguish qubits (cosine similarity > 0.95 in > 30% of qubit pairs). MQT Bench: 786 (VQE/QNN/GHZ), MLQD: 275 (ising/dnn/bv), QASMBench: 39, RevLib: 12, QUEKO: 6. See `data/circuits/splits/filter_log.json` for details. Original splits backed up in `data/circuits/splits/original/`.
- Original 7,165 → Post-preprocessing 6,887 → **Final 5,769 training circuits** (1,396 removed total, 19.5%)
- Primarily unlabeled removals; QUEKO nearly unaffected (6 removed, 0.7%)

### Synthetic Backends
QUEKO/MLQD circuits use hardware topologies not available as Qiskit FakeBackendV2. Synthetic noise profiles are generated by sampling from real FakeBackend noise distributions (seed=42/43, fixed). MLQD's Melbourne/Rochester use real FakeBackendV2 noise.

## Key Commands
```bash
# Training (each run creates a timestamped directory under runs/)
# Training plots are auto-generated in <run_dir>/plots/ on completion
python train.py --config configs/stage1.yaml --name baseline_v1
python train.py --config configs/stage2.yaml --name baseline_v1 \
  --override pretrained_checkpoint=runs/stage1/<STAGE1_RUN>/checkpoints/best.pt

# Config overrides (can stack multiple --override flags)
python train.py --config configs/stage1.yaml --name lr_test \
  --override training.optimizer.lr=0.0005 \
  --override training.mlqd_queko.max_epochs=50

# Evaluation (model + baselines on all 3 test backends, auto-saved to runs/eval/<RUN>/)
python evaluate.py --config configs/stage2.yaml \
  --checkpoint runs/stage2/<RUN>/checkpoints/best.pt \
  --backend toronto brooklyn torino --reps 3

# Benchmark (baselines only, no model)
python evaluate.py --benchmark --backend toronto brooklyn torino

# Manual visualization (for comparing runs or standalone eval CSV)
python scripts/visualize.py runs/stage1/<RUN> runs/stage2/<RUN>
python scripts/visualize.py --eval runs/eval/<RUN>/eval_results.csv

# Tests
pytest tests/                                       # 151 tests

# Feature diagnostics (run before training to verify feature quality)
python scripts/diagnose_features.py --config configs/stage1.yaml
python scripts/diagnose_features.py --features gate_count two_qubit_gate_count single_qubit_gate_ratio --rwpe-k 4

# Dataset scripts
python scripts/generate_queko_noise.py             # Generate synthetic noise profiles
python scripts/process_mlqd.py                     # Process MLQD (extract labels)
python scripts/generate_mqt_bench.py               # Generate MQT Bench circuits
python scripts/normalize_gates.py                  # Normalize all QASM to basis gates {cx,id,rz,sx,x}
```

## Experiment Management
Each `train.py` run creates a timestamped directory with config snapshot, metrics, and note template.
Training plots are auto-generated on completion. Evaluation outputs go to `runs/eval/<RUN>/`.
```
runs/
├── EXPERIMENTS.md               # Central experiment log (all runs, results, changes)
├── stage1/
│   └── 20260323_221150_baseline_after_refactor/
│       ├── config.yaml          # Actual config used (with overrides applied)
│       ├── source_config.txt    # Original config file path
│       ├── note.md              # Auto-generated: what changed, hypothesis, result
│       ├── metrics.csv          # Per-epoch: epoch, phase, tau, lr, train_loss, val_loss
│       ├── plots/               # Auto-generated training visualization
│       │   └── stage1_training.png
│       └── checkpoints/
│           ├── mlqd_queko_best.pt
│           ├── queko_best.pt
│           └── best.pt
├── stage2/
│   └── 20260323_223946_baseline_after_refactor/
│       ├── config.yaml
│       ├── source_config.txt
│       ├── note.md
│       ├── metrics.csv          # Per-epoch: epoch, lr, l_total, <active_components...>, val_surrogate_loss, val_pst
│       ├── plots/               # Auto-generated training visualization
│       │   └── stage2_training.png
│       └── checkpoints/
│           ├── best.pt
│           └── final.pt
└── eval/                        # Evaluation outputs (auto-derived from checkpoint run name)
    └── 20260323_223946_baseline_after_refactor/
        ├── eval_results.csv     # Raw evaluation results (all backends, all reps)
        ├── pst_summary.md       # Per-circuit PST table + per-backend comparison (Markdown)
        ├── pst_summary.csv      # Same as above in CSV format
        ├── pst_comparison_*.png # PST bar charts per backend
        └── pst_heatmap_*.png    # PST heatmaps per backend
```
- `--name` flag appends a label to the timestamp — **use descriptive names reflecting what changed** (e.g. `ablation_no_cross_attn`, `loss_alpha0.5`, `gnn_6layer`, `feat_no_t1t2_ratio`)
- `note.md` is auto-generated with template (What changed / Hypothesis / Result) — fill in after each run
- `runs/EXPERIMENTS.md` is the central log — add one row per experiment with key results
- Previous runs are never overwritten; each run gets its own directory
- `runs/` is gitignored
- `checkpoint_dir`/`log_dir` in YAML configs are fallback values; overridden at runtime by `_setup_run_dir()`
- Stage 2 `pretrained_checkpoint` must be specified via `--override` (no default path)
- Stage 1→2 checkpoint loading uses `strict=False` (Stage 2 adds QualityScore layers not in Stage 1)

### Visualization
Training and evaluation plots are auto-generated. Manual visualization is for comparing runs or standalone use:
```bash
# Training curves (loss only — LR/tau schedules excluded as they are deterministic)
python scripts/visualize.py runs/stage1/<RUN>    # Stage 1: train/val CE loss
python scripts/visualize.py runs/stage2/<RUN>    # Stage 2: loss components + Val PST

# Compare multiple runs (plots saved to each run's plots/ directory)
python scripts/visualize.py runs/stage1/RUN_A runs/stage1/RUN_B

# Evaluation results (PST bar chart + heatmap)
python scripts/visualize.py --eval runs/stage2/<RUN>/eval_results.csv

# Headless (save PNG only, no display)
python scripts/visualize.py runs/stage1/<RUN> --no-show
```
- Stage 1 plots: Train/Val Loss (with phase boundaries)
- Stage 2 plots: L_total + active components, Val PST (with best annotation)
- Plots saved to each run's own `plots/` directory (not shared across stages)

## Hardware Backends
- **Stage 1 Training (55 Qiskit + 5 synthetic = 60 backends)**:
  - Qiskit FakeBackendV2: 5Q×15, 7Q×6, 15-16Q×2, 20Q×5, 27-28Q×12, 33Q×1, 53Q×1, 65Q×1, 127Q×9
  - Synthetic: queko_aspen4(16Q), queko_tokyo(20Q), queko_rochester(53Q), queko_sycamore(54Q), mlqd_grid5x5(25Q)
- **Stage 2 Training (55 Qiskit backends only)**:
  - Synthetic backends excluded; QUEKO/MLQD circuits randomly re-assigned to real backends at data load time
- **Test (UNSEEN)**: FakeToronto(27Q), FakeBrooklyn(65Q), FakeTorino(133Q)
- Native 2-qubit gates: cx, ecr, or cz (auto-detected via `_get_two_qubit_gate_name()`)

## Circuit Node Feature System
Circuit node features are **configurable via YAML** — no code changes needed to experiment with feature combinations. All candidate features are pre-computed during preprocessing (`scripts/preprocess_circuits.py`); feature selection happens at dataset load time.

**Available features:** `gate_count`, `two_qubit_gate_count`, `degree`, `depth_participation`, `weighted_degree`, `single_qubit_gate_ratio`, `critical_path_fraction`
**Positional encoding:** RWPE (Random Walk PE, configurable k steps)

**Current default** (configs/stage1.yaml, stage2.yaml):
```yaml
node_features: [gate_count, two_qubit_gate_count, single_qubit_gate_ratio, critical_path_fraction]
rwpe_k: 2    # node_input_dim = 4 + 2 = 6
```

**Cache format:** `data/circuits/cache/{source}/{filename}.pt` stores raw `node_features_dict` (all 7 features) + `edge_features`. Feature selection at load time means changing features does NOT require re-preprocessing. Old cache format (pre-built PyG Data) is detected and handled via fallback in `dataset.py`.

**Feature diagnostics** (run before training):
```bash
python scripts/diagnose_features.py --config configs/stage1.yaml
python scripts/diagnose_features.py --features gate_count two_qubit_gate_count single_qubit_gate_ratio --rwpe-k 2
```

**Known degeneracy in original features (gc, 2qc, deg, dp):**
- `gate_count` ↔ `depth_participation`: |r| ≈ 1.0 in 100% of circuits (redundant)
- `degree` constant in small fully-connected circuits → z-score makes it all-zero
- Effective dimensionality: ~2.1 / 4; mean cosine similarity > 0.95 in 83-93% of circuits
- Current features (gc, 2qc, sqr, cpf + RWPE2): eff dim 3.7 / 6, max |r| = 0.87

## Training Strategy
- **Stage 1**: Supervised CE loss on labeled data — **currently not in use**
  - Was: MLQD + QUEKO → QUEKO fine-tuning with τ annealing
  - Decision: Stage 2 from scratch achieves comparable results; Stage 1 pretrained checkpoint confirmed harmful (negative transfer, -0.07 PST in controlled test)
- **Stage 2**: Unsupervised surrogate losses on 5,769 circuits (filtered dataset)
  - Loss components configured via YAML registry pattern (see Loss Registry below)
  - Score normalization: configurable via `sinkhorn.score_norm` ("softmax" or "sinkhorn")
  - τ annealing (1.0→0.05, exponential), warm-up 2 epochs
  - Large backend (50Q+) oversampling via `large_backend_boost`
  - Runs from scratch (no Stage 1 pretrained checkpoint)
  - **No early stopping**: trains for full max_epochs (100)
  - **Best checkpoint**: selected by val PST (measured every `pst_validation.interval` epochs)
  - **Val surrogate loss**: computed every epoch on 396 val circuits, logged to CSV (monitoring only, not used for checkpoint selection)
  - **Val PST**: measured every 10 epochs on benchmark circuits via SABRE routing — used for best checkpoint selection

### Loss Registry (Stage 2)
Loss components are modular and configured declaratively in YAML. Each component is registered via `@register_loss()` decorator in `training/losses.py`.

**Available components:**
| Name | Loss | Description | Bounds |
|------|------|-------------|--------|
| `error_distance` | L_surr | Gate-frequency-weighted error distance (Floyd-Warshall on cx_error) | [0, ∞) |
| `adjacency` | L_adj | Binary adjacency matching with gate-frequency weighting | [-1, 0] |
| `hop_distance` | L_hop | Normalized hop distance penalty | [0, 1] |
| `swap_count` | L_swap | SWAP count estimation: 3·max(hop-1,0), gate-freq weighted | [0, ∞) |
| `soft_proximity` | L_soft | Exponential decay proximity: exp(-α·max(hop-1,0)), gate-freq weighted. Params: `alpha` (default 2.0) | [-1, 0] |
| `node_quality` | L_node | Learnable MLP qubit quality score | [-1, 0] |
| `separation` | L_sep | Multi-programming circuit separation | [-1, 0] |
| `exclusion` | L_excl | Column-wise one-to-one mapping penalty (penalizes shared physical qubits) | [l/h, l] |

**YAML config format:**
```yaml
loss:
  type: surrogate
  components:
    - name: error_distance
      weight: 1.0
    - name: soft_proximity
      weight: 0.3
      params:
        alpha: 2.0          # components with constructor args use params dict
```

**Experimenting with loss combinations:**
- Add/remove/reorder components in YAML — no code changes needed
- Weight override via CLI: `--override loss.components.0.weight=2.0` (list index supported)
- Components with constructor params use `params` dict in YAML
- Each run's `config.yaml` records exact loss configuration for reproducibility
- Metrics CSV columns are dynamic — reflect active components

### Loss Design Rationale (Physics Analysis)

**Why edge losses dominate**: PST is dominated by SWAP routing overhead, not individual qubit quality. Each SWAP = 3 CX gates; 2Q gate error (0.5-5%) is 10-100× larger than 1Q error (0.01-0.1%). Optimizing qubit pair placement (edge losses) is far more impactful than individual qubit selection (node losses).

**L_surr limitations**: Floyd-Warshall on raw error rates uses additive sum, but fidelity is multiplicative: F = Π(1-ε). More critically, d_error doesn't account for SWAP 3× CX overhead — hop=2 costs ~4× (not 2×) a direct CX. The error rate values (0.01-0.05) produce weak gradients that saturate quickly.

**L_adj limitations**: Binary A_hw has zero gradient for all non-adjacent pairs. On large backends, coupling density is very low (27Q: ~9%, 127Q: ~1.7%), so >98% of the A_hw matrix is 0 → extremely sparse gradient signal. Cannot distinguish hop=2 from hop=10.

**L_swap rationale**: Directly models SWAP routing cost: d_swap = 3·max(hop-1, 0). Natural step from 0→3 at adjacency boundary captures the most important binary decision. Large dynamic range (0-60+ for 127Q) prevents gradient vanishing.

**L_soft rationale**: Exponential decay provides non-zero gradient everywhere while preserving strong adjacency preference. α parameter controls L_adj↔L_hop spectrum.

**L_node deprecated**: Qubit mapping is fundamentally an edge problem, not a node problem. MLP collapses to trivial solution (rank qubits by noise, circuit-agnostic) in 1-2 epochs. Can conflict with edge losses (best-quality qubits may be far apart on topology).

## Score Normalization
Two modes available, selectable via `sinkhorn.score_norm` in YAML config:
- **`sinkhorn`** (recommended): Log-domain Sinkhorn with dummy padding l×h → h×h → doubly stochastic P. Enforces both row and column sum constraints. **Confirmed best** (+0.086 PST over softmax in controlled test).
- **`softmax`**: Row-wise softmax → P (batch, l, h) row-stochastic. Simple, no dummy padding.

```yaml
sinkhorn:
  score_norm: sinkhorn    # or "softmax"
  tau_max: 1.0
  tau_min: 0.05
  max_iter: 20            # Sinkhorn iterations (ignored for softmax)
```

See `configs/stage2_sinkhorn_adj.yaml` for Sinkhorn + adjacency loss config.

## Experiment History & Best Configurations

### Current Best (2026-04-02)
Run `20260402_004812_filtered_sinkhorn_adj` — **Eval 3-backend avg OURS+SABRE PST 0.589** (Val PST 0.3588, epoch 94)
- Score norm: **Sinkhorn** (confirmed: Sinkhorn >> Softmax, +0.086 in controlled test)
- Loss: error_distance(1.0) + adjacency(0.3)
- Features: new 4-feature (gc, 2qc, sqr, cpf) + RWPE k=2 = 6dim
- HW features: **5dim** (readout_err, sq_err, degree, t1_cx_ratio, t2_cx_ratio)
- noise_bias_dim: **0** (disabled; 7dim+bias tested worse: 0.2604)
- Dataset: Filtered 5,769 circuits
- Eval breakdown: Brooklyn 0.697, Toronto 0.717, Torino 0.353
- Baseline QAP+NASSC: Brooklyn 0.753, Toronto 0.768, Torino 0.386 (avg **0.636**)

### Confirmed Findings (controlled comparisons, 13 runs)
| Variable | Winner | Val PST | Loser | Val PST | Delta |
|----------|--------|---------|-------|---------|-------|
| Score norm | Sinkhorn | **0.3588** | Softmax | 0.2727 | +0.086 |
| Circuit features | new(gc,2qc,sqr,cpf)+RWPE2 | **0.3588** | old(gc,2qc,deg,dp) | 0.2473 | +0.111 |
| HW features | 5dim, no bias | **0.3588** | 7dim+noise_bias | 0.2604 | +0.098 |
| node_quality | without | **0.3588** | with | 0.3474 | +0.011 |
| exclusion | without | 0.2727 | with | 0.2346 | +0.038 |

### Reproducibility & Checkpoint Strategy (2026-04-05)
8 runs with sinkhorn_adj config, varying seed/pretrained/checkpoint strategy:

| Strategy | Runs | Eval OURS+SABRE 3-backend avg |
|----------|:----:|-------------------------------|
| PST best checkpoint, scratch (04/02) | 3 | 0.547, 0.569, **0.589** |
| PST best checkpoint, pretrained | 2 | 0.484, 0.569 |
| PST best checkpoint, scratch (04/05) | 2 | 0.464, 0.506 |
| Val loss best checkpoint (ep4-13) | 2 | 0.395, 0.435 |
| Val loss best, no ES (ep13-23) | 2 | 0.457, 0.517 |

**Conclusions:**
- PST-based checkpoint selection >> val surrogate loss-based (surrogate loss saturates too early)
- No early stopping — train full max_epochs, select best PST checkpoint
- Stage 1 pretrained: not reliably better, sometimes harmful
- High variance across runs (0.395-0.589) — need multiple seeds

### Loss Gradient Analysis (2026-04-02)
Both `error_distance` and `adjacency` gradients depend only on neighbor mapping + hardware structure — **no per-node circuit signal**. `error_distance` saturates by epoch 3 (~0.01); `adjacency` is the only non-saturating loss.

### Phase 1: Edge Loss Optimization (in progress, 2026-04-03)
New loss components implemented: `swap_count` (L_swap), `soft_proximity` (L_soft). Six experiments running:
| Exp | Loss Config | Config |
|-----|-------------|--------|
| E1 | err_dist(1.0) + adj(**0.7**) | stage2_sinkhorn_adj.yaml + override |
| E2 | err_dist(1.0) + adj(**1.0**) | stage2_sinkhorn_adj.yaml + override |
| E3 | **swap(1.0)** + adj(0.3) | stage2_swap_adj.yaml |
| E4 | **swap(1.0)** standalone | stage2_swap_only.yaml |
| E5 | err_dist(1.0) + **soft(0.3, α=2)** | stage2_soft_proximity.yaml |
| E6 | **soft(1.0, α=2)** standalone | stage2_soft_only.yaml |

## Known Issues & Active Investigation
- **Score matrix row collapse**: Circuit information collapses through GNN→cross-attention, making score matrix rows indistinguishable. Partially addressed by feature registry + RWPE. Feature-indistinguishable circuit filtering removes worst cases.
- **No per-node circuit signal in loss**: Both `error_distance` and `adjacency` gradients (∂L/∂P_ip) depend only on neighbor mapping P_j and hardware structure — circuit qubit i's properties (gate count, degree) never appear. This is the fundamental limitation of current edge-pair losses.
- **error_distance saturates by epoch 3**: Drops from ~0.14 to ~0.01 and provides negligible gradient thereafter. `adjacency` is the only loss providing meaningful gradient throughout training.
- **node_quality collapse**: Learned MLP reaches trivial solution (-1.0) by epoch 1-2, zero gradient thereafter. **Do not use** — replaced by `swap_count` and `soft_proximity` in Phase 1 experiments.
- **Val PST oscillation**: PST validation fluctuates 0.12-0.36 across epochs without converging. Best PST often occurs early/mid-training then degrades. Caused by weak correlation between surrogate loss and actual PST, compounded by SABRE routing non-determinism. No early stopping used — train for full max_epochs and select best PST checkpoint.
- **Val surrogate loss saturates early**: Surrogate loss reaches minimum by epoch 4-13, but model continues improving (PST best at epoch 30-90). Val surrogate loss is NOT suitable for checkpoint selection or early stopping — only for monitoring.
- **Stage 1 pretrained checkpoint harmful**: Controlled test (2 pretrained vs 4 scratch runs) confirmed negative transfer. Pretrained avg eval PST 0.527, scratch avg 0.522 at best but with wider variance. Stage 1 CE-trained weights interfere with Stage 2 surrogate loss optimization.
- **High run-to-run variance**: Same config + same seed produces eval PST range of 0.395-0.589 across runs. Non-deterministic CUDA ops, dataloader shuffle, and multi-programming random assignment contribute. Single-run results are unreliable — always run 2+ seeds.
- **Sinkhorn >> Softmax (resolved)**: Controlled experiment confirmed Sinkhorn is decisively better (+0.086 PST). All future experiments use Sinkhorn.
- **Feature-indistinguishable circuits**: 16% of original training data (VQE, QNN, GHZ parametric circuits) had >30% indistinguishable qubit pairs. Removed via filtering. MLQD sqr=0 (57% of MLQD) retained — other features still differentiate qubits.

## Critical Rules
- All quantum circuits are pre-normalized to basis gates {cx, id, rz, sx, x} via `scripts/normalize_gates.py`
- All quantum circuits loaded from .qasm files (OPENQASM 2.0)
- Hardware noise features MUST be z-score normalized WITHIN each backend
- Circuit node features MUST be z-score normalized WITHIN each circuit (RWPE is NOT z-score normalized)
- Circuit edge features MUST be z-score normalized WITHIN each circuit (multi-programming: group-level via `renormalize_group_edges`)
- Score normalization: configurable via `sinkhorn.score_norm` in YAML
  - `softmax` (default): row-wise softmax → P (batch, l, h)
  - `sinkhorn`: log-domain Sinkhorn with dummy padding → P (batch, h, h), sliced to (batch, l, h)
- Stage 1 uses existing labels: QUEKO (τ⁻¹), MLQD (OLSQ2)
- Stage 2 uses unsupervised surrogate losses on all circuits
- Score Matrix uses Cross-Attention + learned projection, NOT simple dot product (noise_bias disabled by default)
- All hyperparameters configurable via YAML
- Batching groups samples by (backend, num_logical) for uniform tensor shapes
- PST measurement: P(correct output) = primary metric
- PST simulation: tensor_network + GPU (cuQuantum) as default; simulators created once per backend, reused for all circuits
- PST simulation: on tensor_network failure (large/deep circuits on 100Q+ backends), simulators are recreated to recover GPU state
- Evaluation order: baselines run before model to prevent GPU state corruption from model-generated deep circuits
- PST measurement: optimization_level configurable (default 3), 8192 shots
- Transpilation: all evaluation paths (baselines + model) use unified `transpile_with_timing()` from `evaluation/transpiler.py`
- Transpilation: custom PassManager with noise-aware UnitarySynthesis (`backend_props`) for all methods
- Transpilation: supported layout×routing combinations (sabre, nassc, dense, noise_adaptive, trivial, qap)
- Benchmark circuits: 23 standard circuits (3Q-13Q), stored in `data/circuits/qasm/benchmarks/`, deduplicated from training sets
- Multi-programming: model handles arbitrary circuit count; training scenarios configured via YAML (no fixed limit on circuit count)

## Dependencies
- torch >= 2.0
- torch-geometric >= 2.4
- qiskit >= 1.0
- qiskit-ibm-runtime
- qiskit-aer (with cuQuantum for tensor_network GPU)
- scipy
- networkx
- pyyaml
- pandas (benchmark reporting)

## Code Conventions
- Type hints on all function signatures
- Docstrings on all public classes and functions
- Config via YAML files, not hardcoded values

## Full Research Specification
For complete architecture details, loss functions, all hyperparameters, training strategy,
and design rationale, see `docs/RESEARCH_SPEC.md`.
**Read this file before implementing any new component.**
