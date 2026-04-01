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
- `models/` — GNN encoders, cross-attention, score head, SoftmaxNorm (+ legacy sinkhorn), hungarian
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
    ├── stage1_supervised.json      # 3,846 labeled circuits for Stage 1
    ├── stage1_queko_only.json      # 486 QUEKO circuits for fine-tuning
    ├── stage1_unsupervised.json    # 2,618 unlabeled circuits
    ├── stage2_all.json             # 6,887 all circuits for Stage 2
    ├── val.json                    # 423 validation (labeled)
    └── val_queko_only.json         # 54 QUEKO validation
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
- Original 7,165 → Final 6,887 training circuits (278 removed, 3.9%)
- QUEKO and MLQD (labeled data) are unaffected; all removals are from unlabeled datasets

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
│       ├── metrics.csv          # Per-epoch: epoch, lr, l_total, <active_components...>, val_pst
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
- **Stage 1**: Supervised CE loss on labeled data (MLQD + QUEKO → QUEKO fine-tuning)
  - Phase 1: Train on all 3,846 labeled circuits with τ annealing (1.0→0.05)
  - Phase 2: Fine-tune on 486 QUEKO circuits with reduced LR (1/10)
  - **Note**: Stage 1 effectiveness is under investigation — may be removed in favor of Stage 2 only
- **Stage 2**: Unsupervised surrogate losses on all 6,887 circuits
  - Loss components configured via YAML registry pattern (see Loss Registry below)
  - Default: L_surr (error-weighted distance) + α·L_node (node quality MLP)
  - τ annealing (1.0→0.05, exponential), warm-up 2 epochs
  - Large backend (50Q+) oversampling via `large_backend_boost`
  - Can run with or without Stage 1 pretrained checkpoint

### Loss Registry (Stage 2)
Loss components are modular and configured declaratively in YAML. Each component is registered via `@register_loss()` decorator in `training/losses.py`.

**Available components:**
| Name | Loss | Description | Bounds |
|------|------|-------------|--------|
| `error_distance` | L_surr | Gate-frequency-weighted error distance (Floyd-Warshall on cx_error) | [0, ∞) |
| `adjacency` | L_adj | Binary adjacency matching with gate-frequency weighting | [-1, 0] |
| `hop_distance` | L_hop | Normalized hop distance penalty | [0, 1] |
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
    - name: node_quality
      weight: 0.3
```

**Experimenting with loss combinations:**
- Add/remove/reorder components in YAML — no code changes needed
- Weight override via CLI: `--override loss.components.0.weight=2.0`
- Each run's `config.yaml` records exact loss configuration for reproducibility
- Metrics CSV columns are dynamic — reflect active components

## Known Issues & Active Investigation
- **Score matrix row collapse**: Circuit information collapses through GNN→cross-attention, making score matrix rows indistinguishable. Root cause: insufficient node feature differentiation at GNN input. Addressed by feature registry + RWPE, but not fully resolved.
- **Stage 2 surrogate loss saturation**: Both `error_distance` and `node_quality` saturate early (epoch ~10), leaving 90+ epochs without meaningful gradient signal. `node_quality` reaches -1.0 (its bound) almost immediately.
- **Val PST oscillation**: PST validation fluctuates widely across epochs (0.12-0.30) rather than converging, suggesting surrogate loss does not correlate well with actual PST.
- **Stage 1 effectiveness**: Stage 1 supervised pretraining may not improve final PST over Stage 2 from scratch. Under investigation — Stage 2 from scratch with new features achieved best Val PST 0.2410 vs 0.3044 with Stage 1 pretrain, but both show high variance.

## Critical Rules
- All quantum circuits are pre-normalized to basis gates {cx, id, rz, sx, x} via `scripts/normalize_gates.py`
- All quantum circuits loaded from .qasm files (OPENQASM 2.0)
- Hardware noise features MUST be z-score normalized WITHIN each backend
- Circuit node features MUST be z-score normalized WITHIN each circuit (RWPE is NOT z-score normalized)
- Circuit edge features MUST be z-score normalized WITHIN each circuit (multi-programming: group-level via `renormalize_group_edges`)
- Score normalization: SoftmaxNorm (row-wise softmax) is the primary method → P (batch, l, h) row-stochastic
  - SinkhornLayer (log-domain, dummy padding l×h → h×h → doubly stochastic P (batch, h, h)) is kept in `models/sinkhorn.py` for future experimentation
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
