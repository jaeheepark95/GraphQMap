# CLAUDE.md

## Project Overview
**GraphQMap** — Hardware-agnostic quantum qubit mapping ML model.
A single GNN-based model that outputs initial layouts for quantum circuit compilation on NISQ hardware.
Supports arbitrary multi-programming (any number of co-located circuits, total logical qubits ≤ physical qubits).

## Tech Stack
- Python 3.10+, PyTorch, PyTorch Geometric (PyG)
- Qiskit >= 1.0 (< 2.0), qiskit-ibm-runtime (FakeBackendV2), qiskit-aer-gpu
- cuQuantum (tensor_network GPU simulation for PST evaluation)
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
  - `analyze_circuit_features.py` — 7-phase circuit graph feature analysis (completeness, raw stats, CoV, correlation, normalization, RWPE, size-dependent)
- `docs/` — research specification and documentation
- `tests/` — unit and integration tests (152 tests)

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

## Environment Setup
```bash
# 1. Create conda environment
conda create -n graphqmap python=3.10 -y && conda activate graphqmap

# 2. Install PyTorch (CUDA index URL required — adjust for your CUDA version)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 3. Install all dependencies (pinned versions)
pip install -r requirements-lock.txt

# 4. Transfer data (from existing server — cache included to skip preprocessing)
#    On source: tar czf circuits.tar.gz data/circuits/
#    On target: tar xzf circuits.tar.gz  (run from project root)

# 5. Verify
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pytest tests/ -x -q  # expect 152 passed
```

**Important:**
- PyTorch MUST be installed separately before `requirements-lock.txt` (CUDA index URL needed)
- NVIDIA sub-packages (nvidia-cublas-cu12, etc.) are managed by torch — never pin them manually
- `qiskit-aer-gpu` (not `qiskit-aer`) is required for tensor_network GPU simulation
- `qiskit >= 2.0` breaks `qiskit-aer-gpu==0.15.1` — keep qiskit < 2.0
- `cuquantum-cu12 >= 25.0` breaks `qiskit-aer-gpu==0.15.1` — keep cuquantum 24.x

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
pytest tests/                                       # 152 tests

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
- **Stage 1 Training (49 Qiskit + 5 synthetic = 54 backends)**:
  - Qiskit FakeBackendV2: 5Q×15, 7Q×6, 15-16Q×2, 20Q×5, 27-28Q×11, 33Q×1, 53Q×1, 127Q×8
  - Synthetic: queko_aspen4(16Q), queko_tokyo(20Q), queko_rochester(53Q), queko_sycamore(54Q), mlqd_grid5x5(25Q)
- **Stage 2 Training (49 Qiskit backends only)**:
  - Synthetic backends excluded; QUEKO/MLQD circuits randomly re-assigned to real backends at data load time
- **Validation (held-out from training, UNSEEN by training data)**: FakeMumbai(27Q, Falcon r5.11), FakeManhattan(65Q, Hummingbird r2), FakeWashington(127Q, Eagle cx)
  - Used for PST checkpoint selection in Stage 2 (every `pst_validation.interval` epochs)
  - Size-matched to test backends but never appear in training; eliminates val=test leakage
- **Test (UNSEEN by both training and validation)**: FakeToronto(27Q), FakeBrooklyn(65Q), FakeTorino(133Q)
  - Evaluated **once** at the end via `evaluate.py`; never used for checkpoint/model selection
- Native 2-qubit gates: cx, ecr, or cz (auto-detected via `_get_two_qubit_gate_name()`)
- **History note**: Pre-2026-04-07 runs used test backends for PST validation (data leakage via checkpoint selection). All historical eval numbers (e.g. 0.589 from `20260402_004812_filtered_sinkhorn_adj`) are upper-bound estimates contaminated by this leakage and not directly comparable to post-2026-04-07 runs.

## Circuit Node Feature System
Circuit node features are **configurable via YAML** — no code changes needed to experiment with feature combinations. All candidate features are pre-computed during preprocessing (`scripts/preprocess_circuits.py`); feature selection happens at dataset load time.

**Available features:** `gate_count`, `two_qubit_gate_count`, `degree`, `depth_participation`, `weighted_degree`, `single_qubit_gate_ratio`, `critical_path_fraction`, `interaction_entropy`
**Positional encoding:** RWPE (Random Walk PE, configurable k steps, start_step=2)

**Current default** (configs/stage1.yaml, stage2.yaml):
```yaml
node_features: [gate_count, two_qubit_gate_count, single_qubit_gate_ratio, critical_path_fraction]
rwpe_k: 2    # node_input_dim = 4 + 2 = 6
```

**RWPE:** `compute_rwpe(start_step=2)` skips step 1 (structurally zero for graphs without self-loops). With `rwpe_k=2`, outputs `[M^2, M^3]` (both non-trivial). Changed 2026-04-06 from `[M^1(dead), M^2]`.

**Cache format:** `data/circuits/cache/{source}/{filename}.pt` stores raw `node_features_dict` (all 8 features) + `edge_features`. Feature selection at load time means changing features does NOT require re-preprocessing. Old cache format (pre-built PyG Data) is detected and handled via fallback in `dataset.py`.

**Feature diagnostics** (run before training):
```bash
python scripts/diagnose_features.py --config configs/stage1.yaml
python scripts/diagnose_features.py --features gate_count two_qubit_gate_count single_qubit_gate_ratio --rwpe-k 2
python scripts/analyze_circuit_features.py --num-samples 500   # 7-phase comprehensive analysis
```

**Circuit feature analysis (2026-04-06, 520 circuits via `scripts/analyze_circuit_features.py`):**
- Current features (gc, 2qc, sqr, cpf + RWPE2): eff dim 3.68/6, indist rate 16.73%, max |r| = 0.753 (gc↔2qc)
- Normalization: all z-score confirmed better than mixed (mixed: indist +7.03pp worse)
- RWPE k=2 confirmed optimal (k=3: +0.23 eff_dim but RWPE[2]↔2qc r=0.768, dead dims increase)
- **sqr problem**: 37.7% all-zero, CoV<0.1 in 53.8%, medium(6-10Q) 54% constant
- **cpf problem**: median 0.956 (saturated), xlarge(21Q+) 67% constant
- **xlarge indist 41%**: worst size bucket, cpf constant + gc CoV 0.069

**Rejected features (with evidence):**
- `depth_participation`: gc와 r=1.000 (100%)
- `weighted_degree`: 2qc와 r=1.000 (100%)
- `degree`: GNN learns topology + 2qc와 r>0.9 (64.5%)
- `interaction_entropy`: degree와 r=0.976 (95.3% >0.9) — H ≈ log(degree)

**Circuit Edge Features (5dim, configurable via `circuit_gnn.edge_input_dim`):**

| Feature | Description | Const% | CoV<0.1 |
|---------|-------------|--------|---------|
| `interaction_count` | Number of 2Q gates between this qubit pair | 27.4% | 27.7% |
| `earliest_interaction` | Normalized time (0~1) of first 2Q gate | 2.3% | 1.2% |
| `latest_interaction` | Normalized time (0~1) of last 2Q gate | 1.9% | 10.4% |
| `interaction_span` | `latest - earliest` (temporal duration) | 17.0% | 31.0% |
| `interaction_density` | `count / (span + eps)` (burstiness) | 17.2% | 28.9% |

All z-score normalized within each circuit. First 3 are established; span/density added 2026-04-06 but **rejected after ablation** (edge 5dim eval PST 0.486 vs 3dim 0.572). `edge_dim` parameter controls slicing (default 3). Old 3-dim caches are auto-extended to 5-dim via `_extend_edge_features()` when `edge_dim=5` is requested.

## Hardware Node/Edge Feature System
Hardware features are extracted from FakeBackendV2 (or synthetic JSON) per backend. Configurable via `hardware_gnn.node_input_dim`, `hardware_gnn.edge_input_dim`, and `hardware_gnn.exclude_degree` in YAML configs.

**Node Features (6dim default, configurable):**

| Feature | Description | Normalization | Direction |
|---------|-------------|---------------|-----------|
| `readout_error` | Measurement error rate | z-score | Lower = better |
| `single_qubit_error` | Avg 1Q gate error (sx, x) | z-score | Lower = better |
| `degree` | Coupling map connectivity | z-score | Structural (optional, `exclude_degree: true` to remove) |
| `t1_cx_ratio` | T1 / mean_cx_duration | z-score | Higher = better |
| `t2_cx_ratio` | T2 / mean_cx_duration | z-score | Higher = better |
| `t2_t1_ratio` | T2 / T1, clipped to [0, 2] | **raw** (not z-scored) | Decoherence type: ≈2 relaxation-limited, ≈1 dephasing-dominated |

**Edge Features (2dim):**

| Feature | Description | Normalization |
|---------|-------------|---------------|
| `2q_gate_error` | 2Q gate error rate (cx/ecr/cz), averaged over both directions | z-score |
| `edge_coherence_ratio` | `cx_duration / min(T1_u, T1_v, T2_u, T2_v)` — fraction of coherence budget consumed per gate | **raw** (not z-scored) |

**Normalization strategy (mixed):**
- **Z-scored within backend**: error rates, coherence ratios, degree — scale varies across backends
- **Raw (no normalization)**: dimensionless ratios with inherent physical meaning (T2/T1, edge_coherence) — absolute value carries information, z-score would destroy it

**Config options:**
```yaml
hardware_gnn:
  node_input_dim: 6        # 5 z-scored + 1 raw (default)
  edge_input_dim: 2        # 1 z-scored + 1 raw
  exclude_degree: false    # set true for ablation (-1 dim → node_input_dim: 5)
```

**Data availability notes:**
- `single_qubit_error`: ALL ZERO on 10 older backends (Burlington, Essex, London, Almaden, etc.) — FakeBackendV2 doesn't provide sx/x gate error
- `2q_gate_error`: ALL ZERO on fake_kyoto (ecr gate error not provided)
- Edge asymmetry: 99.3% of edges have symmetric error (<1% diff), ~10% duration asymmetry — averaged for undirected edges

**Feature independence verified (2026-04-06):**
- Node: max |r| = 0.41 (t1_cx_ratio ↔ t2_cx_ratio), all pairs |r| < 0.7
- Edge: r = 0.059 (2q_error ↔ edge_coherence_ratio) — near-independent

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
| HW degree (node) | without | 0.3478 | with | 0.3448 | +0.003 (negligible) |

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

### Phase 3: Circuit Feature Analysis & Edge Ablation (2026-04-06)
Comprehensive 7-phase analysis of circuit graph features using `scripts/analyze_circuit_features.py`.

**Analysis framework (mirroring HW feature analysis):**
1. Data completeness — NaN/Inf/all-zero/constant check
2. Raw statistics — pre-z-score distributions
3. Within-circuit CoV — qubit-differentiating power
4. Correlation — node, edge, node-edge cross
5. Normalization strategy — all z-score vs mixed
6. RWPE quality — k=2,3,4 comparison
7. Size-dependent analysis — per-size-bucket metrics

**Key findings:** see Circuit Node Feature System section above.

**Changes implemented:**
- RWPE `start_step=2`: skip dead step 1, output `[M^2, M^3]` instead of `[M^1(dead), M^2]`
- Edge features expanded 3→5dim: added `interaction_span`, `interaction_density`
- `edge_dim` parameter threaded through `build_circuit_graph` → `build_circuit_graph_from_raw` → `dataset.py` → `train.py` → `evaluate.py`
- Backward compat: `_extend_edge_features()` auto-extends old 3-dim caches to 5-dim
- `interaction_entropy` node feature computed but rejected (degree와 r=0.976)

**Edge ablation results (2026-04-07):**
| Exp | Edge Dim | Best Epoch | Eval OURS+SABRE avg | Eval OURS+NASSC avg |
|-----|:--------:|:----------:|:-------------------:|:-------------------:|
| C1 | 3 (baseline) | 64 | **0.572** | **0.613** |
| C2 | 5 (+span, density) | 19 | 0.486 | 0.495 |

**Conclusion:** Edge 5dim significantly worse (-0.086 OURS+SABRE). span/density features hurt rather than help. **Keep edge 3dim.** C2 best epoch=19 suggests unstable training with 5dim edges.

### Phase 2: Hardware Feature Enhancement (2026-04-06)
Added T2/T1 ratio (node, raw) and edge_coherence_ratio (edge, raw) to hardware graph features. Mixed normalization: z-score for scale-dependent features, raw for dimensionless ratios.

**Degree ablation results:**
| Exp | HW Node Dim | Degree | Val PST | Eval OURS+SABRE avg | Eval OURS+NASSC avg |
|-----|:-----------:|:------:|:-------:|:-------------------:|:-------------------:|
| A1 | 6 | included | 0.3448 | 0.554 | 0.584 |
| A2 | 5 | excluded | 0.3478 | 0.522 | 0.558 |

**Conclusions:**
- Degree inclusion/exclusion: with_degree slightly better (+0.032 OURS+SABRE) but both below current best
- Both A1 (0.554) and A2 (0.522) < current best (0.589) — new HW features (t2_t1_ratio, edge_coherence_ratio) did not improve over original 5dim baseline
- Single-run results; high variance caveat applies

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
- **HW feature gaps in older backends**: `single_qubit_error` ALL ZERO on 10 backends (burlington, essex, london, almaden, boeblingen, johannesburg, poughkeepsie, singapore, cambridge, rochester); `2q_gate_error` ALL ZERO on kyoto. Total 11/55 training backends (20%). These are FakeBackendV2 data gaps, not code bugs. Model learns to handle zero-variance features via z-score → all-zero column. Future experiment: exclude these backends and measure impact.
- **Feature-indistinguishable circuits**: 16% of original training data (VQE, QNN, GHZ parametric circuits) had >30% indistinguishable qubit pairs. Removed via filtering. MLQD sqr=0 (57% of MLQD) retained — other features still differentiate qubits.

## Critical Rules
- All quantum circuits are pre-normalized to basis gates {cx, id, rz, sx, x} via `scripts/normalize_gates.py`
- All quantum circuits loaded from .qasm files (OPENQASM 2.0)
- Hardware noise features: z-scored features MUST be z-score normalized WITHIN each backend; raw features (T2/T1 ratio, edge_coherence_ratio) are NOT z-scored
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
