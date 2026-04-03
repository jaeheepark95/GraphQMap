# CLAUDE.md

## Project Overview

**Project name:** GraphQMap / KMW canonical branch (`kmw1`)

This repository currently has **two conceptual branches**:

- `src/kmw/` = older learned-reindexer v1.4 branch
- `src/kmw1/` = current **canonical-hardware branch** that removed the learned reindexer and now serves as the main research branch for the current experiments

The current research goal is:

> Learn a **label-free initial qubit mapper** for a **fixed 27-qubit IBM BackendV2 target** using a **U-Net + cross-attention** architecture, where the circuit is the main spatial input and the hardware is injected as conditioning tokens.

This branch is **not** doing full routing during training.  
It predicts an **initial logical-to-physical mapping** that is later evaluated under real transpilation/routing.

### What this branch is trying to do

The current canonical branch is designed to test whether the performance drop in the learned-reindexer v1.4 branch was primarily caused by the learned reindexer. The canonical branch therefore:

- removes the learned reindexer entirely
- keeps **hardware canonicalization only**
- keeps the streamlined v1.4 mapper design:
  - single circuit matrix as U-Net input
  - hardware-only token conditioning
  - shallow conditional U-Net backbone
- uses corrected **Option A** runtime semantics:

```text
native -> canonical reindexing -> mapper -> decode -> native-frame loss
```

The current branch now also supports **real-circuit label-free training**:
- Stage 0: smoke test
- Stage 1: synthetic warmup
- Stage 2: mixed synthetic + real training
- Stage 3: real-focused calibration

### What has already been done

The current branch has already incorporated the following major changes:

1. **Removed learned reindexing**
   - no learned `R_L`
   - no learned `R_H`
   - no `tau_r`, locality loss, consistency loss, reindexer entropy diagnostics, or related machinery

2. **Restored fixed hardware canonicalization**
   - hardware only
   - logical order remains identity
   - canonical frame is used for the mapper
   - logits are decoded back to native frame before assignment/loss

3. **Kept the streamlined v1.4 model**
   - no rollback to full v1_1 representation
   - no 5-channel spatial input
   - no logical tokens
   - no old lambda-weighted PST/SWAP/depth loss family

4. **Fixed three major performance-risk issues**
   - source-balanced sampling is required
   - padded dummy logical rows must not participate in Sinkhorn/Hungarian
   - route term is explicitly reweighted by `lambda_route`, with mandatory loss decomposition logging

5. **Integrated real-circuit training**
   - QASMBench and RevLib are now part of training
   - no supervised stage
   - per-source splits are explicit
   - validation now uses both synthetic and real pools
   - checkpoint selection is driven primarily by real validation

---

## Tech Stack

### Language / core libraries
- Python 3.10+
- PyTorch
- Qiskit >= 1.0
- `qiskit-ibm-runtime`
- `qiskit-aer`
- SciPy
- NetworkX
- PyYAML
- pandas

### Environment
Use this Conda environment:

```bash
conda activate graphqmap_pascal
```

Recommended project-root startup:

```bash
cd ~/KMWs_workspace/GraphQMap/KMW
export PYTHONPATH="$PWD/src"
conda activate graphqmap_pascal
```

### Main computational ingredients
- U-Net-style 2D mapper
- cross-attention from circuit feature maps to hardware tokens
- Sinkhorn for training-time soft assignment
- Hungarian for inference-time hard assignment
- deterministic hardware canonicalization
- routed final evaluation with PST / SWAP / depth / compile-time outputs

---

## Project Structure

The current canonical branch must follow this exact `src/kmw1` structure:

```text
src/
└── kmw1/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   └── dataset.py
    ├── preprocessing/
    │   ├── __init__.py
    │   └── pipeline.py
    ├── models/
    │   ├── __init__.py
    │   └── model.py
    ├── losses/
    │   ├── __init__.py
    │   └── loss.py
    ├── training/
    │   ├── __init__.py
    │   └── trainer.py
    ├── evaluation/
    │   ├── __init__.py
    │   └── evaluate.py
    ├── cli/
    │   ├── __init__.py
    │   └── main.py
    └── utils.py
```

### Important rule
These files must **not** exist in `src/kmw1`:

```text
extractor.py
featurizer.py
canonical_indexer.py
layers.py
```

Their contents must be merged into:

- `preprocessing/pipeline.py`
- `models/model.py`

### One-line responsibilities

- `dataset.py` — manifests, split selection, per-sample loading, source-balanced sampling support
- `pipeline.py` — backend extraction, circuit featurization, canonicalization, normalization, validation, decode helpers
- `model.py` — hardware token encoder, shallow conditional U-Net, cross-attention, assignment helpers
- `loss.py` — v1.4.1 execution-surrogate loss in native frame after decode, active-row-only support, `lambda_route`
- `trainer.py` — staged training orchestration, checkpointing, logging, samplers, validation
- `evaluate.py` — inference, decode, native-frame assignment, proxy metrics, routed final evaluation, CSV/JSON summaries
- `main.py` — CLI entrypoint
- `utils.py` — shared helpers

---

## Dataset Structure

Current data root:

```text
data/
├── circuits_v2/
│   └── qasm/
│       ├── queko/
│       ├── mlqd/
│       ├── mqt_bench/
│       ├── qasmbench/
│       ├── revlib/
│       └── benchmarks/
├── manifests/
│   └── full/
│       ├── source_manifests/
│       ├── recipes/
│       └── catalog.json
└── cache/
```

### Important distinction
In the **current canonical branch**, the project is centered on a **fixed 27-qubit IBM backend target**.  
So the words **synthetic** and **real** mainly refer to **circuit datasets**, not to switching many hardware backends during training.

---

## Dataset Sources

### Synthetic training sources
- `queko` — 900 circuits
- `mlqd` — 4,443 circuits
- `mqt_bench` — 1,219 circuits

### Real training sources
- `qasmbench` — 94 circuits
- `revlib` — 231 circuits

### External-only evaluation source
- `benchmarks/`

### Current split contract

#### Synthetic
- `queko`: train 810 / val 90
- `mlqd`: train 3999 / val 444
- `mqt_bench`: train 1097 / val 122

#### Real
- `qasmbench`: train 60 / val 10 / test 24
- `revlib`: train 145 / val 30 / test 56

#### Real test pools
- `test_real_core20` — curated subset of `test_real_broad80`
- `test_real_broad80` — all held-out real test circuits
- `test_benchmarks_ext` — all `benchmarks/` circuits

### Leakage rules
- splits are per-source, not post-concatenation
- no train/val/test overlap
- no benchmark leakage into training
- keep related variants grouped if possible
- do not allow near-duplicates to cross splits

---

## Dataset Preprocessing

### Circuit-side preprocessing
For each circuit:
- parse QASM
- reject circuits with `K > 27`
- build weighted logical interaction matrix `A`
- build logical-valid mask `m`
- build count vectors:
  - `n1Q`
  - `nmeas`
- pad all logical tensors to 27

### Hardware-side preprocessing
For the fixed backend:
- extract native hardware tensors:
  - `B_nat`
  - `c1_nat`
  - `c2_nat`
  - `D_nat`
  - `D_raw_nat`
  - `e1q_nat`
  - `ero_nat`
  - `e2q_nat`
- compute deterministic hardware permutation `p`
- build:
  - `B_can`
  - `c1_can`
  - `c2_can`
  - `D_can`
  - `D_raw_can`
  - `e1q_can`
  - `ero_can`
  - `e2q_can`

### Canonicalization rule
Canonicalization is **hardware only**:

```text
logical order: unchanged
hardware order: canonicalized
```

### Decode rule
Mapper outputs canonical-frame logits `S_can`, then deterministically decodes to native-frame logits:

```text
S_can -> S_nat
```

using the inverse permutation.

### Finite-value policy
- no `inf`
- no `-inf`
- no giant sentinels
- all tensors must be finite
- fail loudly on NaN / Inf

---

## Synthetic Backends

### Important clarification
The **current kmw1 canonical branch** is **not** the older multi-backend synthetic-noise GraphQMap training pipeline.

So for this branch:

- the main backend target is a fixed **27-qubit IBM BackendV2**
- the main synthetic/real distinction applies to **circuit sources**
- historical synthetic backend profiles from the older GNN project are not the main current path

### Hardware backends used in the current branch
- default fixed target: `ibm_backendv2_fixed_27q`
- evaluation examples often use: `fake_toronto_v2`

If you see references to older synthetic backends from the historical CLAUDE file, treat them as **legacy GraphQMap context**, not the primary current kmw1 branch contract.

---

## Hardware Backends

### Current canonical branch
- fixed 27-qubit IBM BackendV2 target for training / canonical branch logic
- native backend tensors are always extracted before canonicalization

### Current evaluation examples
Known working evaluation examples use:

```text
fake_toronto_v2
```

### Native-frame rule
Even though the mapper operates in canonical hardware space, the official evaluation contract is still:

```text
native -> canonical reindexing -> mapper -> decode -> native-frame loss
```

and final reported mappings are native-ID mappings.

---

## Important Concepts

## 1. U-Net + attention
The mapper is a **shallow conditional U-Net**:
- spatial input = circuit matrix `A`
- hardware context = token sequence
- cross-attention injects hardware into circuit-side feature maps

## 2. Tokenization
Each canonical hardware slot becomes one token:

```text
x_hw_can[j] = concat(B_can[j,:], c2_can[j,:], c1_can[j])
```

These are passed through an MLP to produce hardware embeddings used as keys/values.

## 3. Tensor organization
### Circuit tensors
- `A` — weighted logical interaction matrix
- `m` — logical-valid mask
- `n1Q` — per-logical-qubit 1Q count
- `nmeas` — per-logical-qubit measurement count

### Native hardware tensors
- `B_nat`
- `c1_nat`
- `c2_nat`
- `D_nat`
- `D_raw_nat`
- `e1q_nat`
- `ero_nat`
- `e2q_nat`

### Canonical hardware tensors
- `B_can`
- `c1_can`
- `c2_can`
- `D_can`
- `D_raw_can`
- `e1q_can`
- `ero_can`
- `e2q_can`

### Assignment/logit tensors
- `S_can` — canonical-frame logits
- `S_nat` — native-frame logits after deterministic decode
- `P_nat_act` — active-row soft assignment in native frame
- `M_nat_act` — active-row hard assignment in native frame

## 4. Option A semantics
This is the locked runtime:

```text
native -> canonical reindexing -> mapper -> decode -> native-frame loss
```

Not:
- learned reindexer
- canonical-frame loss
- logical canonicalization
- both-sides canonicalization

## 5. Active-row-only assignment
Padded dummy logical rows must **not** participate in:
- Sinkhorn during training
- Hungarian during inference

Only active logical rows should compete for hardware columns.

## 6. Staged real-circuit training
Current branch stages:

- **Stage 0** — smoke test
- **Stage 1** — synthetic warmup
- **Stage 2** — mixed synthetic + real training
- **Stage 3** — real-focused calibration

## 7. Source-balanced / group-balanced sampling
### Stage 1
Uniform source-balanced:
- QUEKO = 1/3
- MLQD = 1/3
- MQT Bench = 1/3

### Stage 2
Two-level group-balanced:
- synthetic group = 70%
- real group = 30%

Within synthetic:
- uniform over `queko / mlqd / mqt_bench`

Within real:
- proportional over `qasmbench / revlib`

### Stage 3
Two-level group-balanced:
- synthetic group = 50%
- real group = 50%

Within synthetic:
- uniform over `queko / mlqd / mqt_bench`

Within real:
- proportional over `qasmbench / revlib`

## 8. Deterministic hardware canonicalization
The learned reindexer was removed.
The current branch instead computes a deterministic hardware permutation `p` from topology and cost tensors, and uses that to canonicalize hardware once per backend snapshot.

---

## Training Strategy

The current branch is fully **label-free**.

There is **no supervised stage**.

### Stage 0 — Smoke Test
Purpose:
- verify preprocessing
- verify canonicalization
- verify decode path
- verify assignment path
- verify logging
- verify routed evaluation

Recommended source:
- small MQT smoke subset

### Stage 1 — Synthetic Warmup
Sources:
- QUEKO
- MLQD
- MQT Bench

Sampling:
- choose source first
- sample uniformly within source bucket

Purpose:
- stabilize training on larger synthetic sets before real-circuit exposure

### Stage 2 — Mixed Synthetic + Real Training
Sources:
- QUEKO
- MLQD
- MQT Bench
- QASMBench
- RevLib

Sampling:
- synthetic group 70%
- real group 30%

Purpose:
- introduce real-circuit training signal without letting tiny real sets dominate

### Stage 3 — Real-Focused Calibration
Start from best Stage-2 checkpoint.

Sampling:
- synthetic group 50%
- real group 50%

Purpose:
- bias the model more strongly toward real-circuit behavior at lower LR

### Validation strategy
Two validation pools:
- `val_synth_macro`
- `val_real_macro`

Checkpointing:
- primary = `val_real_macro`
- secondary guardrail = `val_synth_macro`

### Epoch semantics
Do **not** treat epoch as raw concatenated pass.

Recommended default:
- `1 epoch = 6000 balanced draws`

---

## Loss Function

The current branch uses the **v1.4.1 execution-surrogate loss family**.

### Core rule
Loss is evaluated in the **native frame after deterministic decode**.

### Structure
- `L_1Q`
- `L_RO`
- `L_2Q`
- `L_native = L_1Q + L_RO + L_2Q`
- `L_route`
- `L_task = L_native + lambda_route * L_route`

### Required diagnostics
- `L_1Q`
- `L_RO`
- `L_2Q`
- `L_native`
- `L_route`
- `L_task`
- `route_to_native_ratio = L_route / max(L_native, eps)`

### Why this matters
This explicit route weighting was added because the route term had become too weak in practice relative to native terms.

### Important non-goals
Do **not** silently revert to:
- the old v1_1 lambda-weighted PST/SWAP/depth family
- canonical-frame loss
- learned decode

---

## YAML Config Format

Use a main YAML config with at least these current-branch keys:

```yaml
backend: ibm_backendv2_fixed_27q
batch_size: 1
optimizer: adamw

lr_stage1: 1.0e-4
lr_stage2: 1.0e-4
lr_stage3: 5.0e-5

weight_decay: 1.0e-4
grad_clip_norm: 1.0

sinkhorn_tau: 0.5
sinkhorn_iters: 30

use_learned_reindexer: false
use_logical_canonicalizer: false
use_hardware_canonicalizer: true

loss_family: v1.4.1_execution_surrogate
loss_frame: native_after_decode
lambda_route: 1.0
log_loss_decomposition: true

stage0_recipe: smoke_mqt

stage1_train_sources: [queko, mlqd, mqt_bench]
stage2_train_sources: [queko, mlqd, mqt_bench, qasmbench, revlib]
stage3_train_sources: [queko, mlqd, mqt_bench, qasmbench, revlib]

stage1_sampling: synth_uniform
stage2_sampling:
  synthetic_group_prob: 0.70
  real_group_prob: 0.30
  synthetic_within_group: uniform
  real_within_group: proportional_train_count

stage3_sampling:
  synthetic_group_prob: 0.50
  real_group_prob: 0.50
  synthetic_within_group: uniform
  real_within_group: proportional_train_count

epoch_draws: 6000

primary_checkpoint_metric: val_real_macro
secondary_guardrail_metric: val_synth_macro

real_train_enabled: true
external_only_eval_sources: [benchmarks]
detailed_real_eval_split: test_real_core20
headline_real_eval_split: test_real_broad80

project_namespace: kmw1
```

### Sidecar JSON configs commonly used in practice
Loss/eval configs are often stored as sidecar JSONs for runs.

Example loss-config shape:

```json
{
  "sinkhorn_tau": 0.50,
  "sinkhorn_iters": 30,
  "eps": 1e-8,
  "eps_surv": 1e-12,
  "route_step_2q_mult": 3.0,
  "route_step_1q_mult": 2.0
}
```

Example eval-config shape:

```json
{
  "print_console_summary": true,
  "fail_fast": false,
  "use_qiskit_default_mapper": true,
  "routing_method": null,
  "transpile_optimization_level": 0,
  "seed_transpiler": 20260325,
  "include_readout_in_pst": true
}
```

---

## Key Commands

### Environment
```bash
cd ~/KMWs_workspace/GraphQMap/KMW
export PYTHONPATH="$PWD/src"
conda activate graphqmap_pascal
```

### Evaluate a checkpoint on QUEKO test with routed PST
```bash
python -m kmw1.cli.main eval   --project-root "$PWD"   --manifest data/manifests/full/source_manifests/queko/test.jsonl   --checkpoint "$CKPT"   --backend-name fake_toronto_v2   --device cuda   --seed 20260331   --num-workers 0   --sinkhorn-tau 0.50   --sinkhorn-iters 30   --loss-config "$RUN_DIR/loss_config_kmw1.json"   --eval-config "$RUN_DIR/eval_config_queko_pst.json"   --per-circuit-csv "$RUN_DIR/queko_test_per_circuit_pst.csv"   --summary-json "$RUN_DIR/queko_test_summary_pst.json"   --eval-split queko_test   --route-final-eval
```

### Inspect PST-inclusive results quickly
```bash
python - <<'PY'
import pandas as pd
p = "artifacts/fullrun_queko_mqt_kmw1_current_seed20260331/queko_test_per_circuit_pst.csv"
df = pd.read_csv(p)
cols = [
    "id",
    "real_pst_gate_readout",
    "routing_compile_time_s",
    "swap_inserted_count",
    "swap_overhead_ratio",
    "added_2q_ops",
    "depth_increase_abs",
    "depth_increase_ratio",
]
print(df[cols].head(20))
print("\n=== means ===")
for c in cols[1:]:
    if c in df.columns:
        print(f"{c}: {df[c].mean()}")
PY
```

### Smoke / main / eval helpers
```bash
bash scripts/run_smoke.sh
bash scripts/run_main.sh
bash scripts/run_eval.sh
```

---

## Experiment Management

Current run directories live under:
- `runs/`
- `artifacts/`

Use the following discipline:
- keep one run directory per experiment
- save config snapshots
- save per-circuit CSVs
- save summary JSONs
- save routed-eval outputs
- log the stage schedule and split recipe used

### Mandatory official comparison outputs
For a run to count as complete, routed evaluation outputs must include at minimum:

- `real_pst_gate_readout`
- `swap_inserted_count`
- `added_2q_ops`
- `routing_compile_time_s`
- `routing_total_eval_time_s`
- `original_depth`
- `routed_depth`
- `depth_increase_abs`

If `real_pst_gate_readout` is missing, the routed-metrics addon is not fully active.

---

## Critical Rules

1. **Read `Canonical Design Plan v1.44.md` before making architectural changes.**
2. The current canonical branch is **kmw1**, not the older `kmw` learned-reindexer branch.
3. Runtime is locked to:

```text
native -> canonical reindexing -> mapper -> decode -> native-frame loss
```

4. Hardware canonicalization only.
5. Logical order remains identity.
6. Circuit tensor is **not** canonically reordered.
7. Do not reintroduce learned `R_L / R_H` unless explicitly asked.
8. Do not restore logical tokens or 5-channel spatial input unless explicitly asked.
9. Dummy logical rows must **not** participate in Sinkhorn/Hungarian.
10. Route term must stay explicitly reweighted by `lambda_route`.
11. Routed hard metrics are mandatory for official comparison.
12. `benchmarks/` is external-only evaluation, not training.
13. Keep `src/kmw1` at one main file per subsystem.
14. Do not add `extractor.py`, `featurizer.py`, `canonical_indexer.py`, or `layers.py` back into `kmw1`.
15. No infinities in tensors or loss.
16. Fail fast on NaN / Inf.
17. Keep imports from the `kmw1` package root.
18. Preserve native-ID output compatibility for downstream transpilation/routing.

---

## Dependencies / Environment Setup

Required packages:
- torch
- qiskit
- qiskit-ibm-runtime
- qiskit-aer
- scipy
- networkx
- pyyaml
- pandas

Recommended environment:
- Conda env: `graphqmap_pascal`

Project-root setup:
```bash
cd ~/KMWs_workspace/GraphQMap/KMW
export PYTHONPATH="$PWD/src"
conda activate graphqmap_pascal
```

---

## Code Conventions

- keep one main `.py` file per subsystem in `src/kmw1`
- use clear type hints
- keep public functions/classes documented
- prefer config-driven behavior over hardcoding
- do not silently change runtime semantics
- preserve Option A frame semantics explicitly in code
- keep evaluation outputs reproducible and machine-readable
- when adding logic, update both:
  - the design plan
  - the evaluation / logging contract if outputs change

### Special convention for Claude
When working on this repo:
- first restate the task briefly
- identify any ambiguity that would change architecture or metrics
- then propose the minimal patch
- then implement

---

## Files Claude Should Read First

### Minimal core set for almost all tasks
Read these **first**, in this exact order:

1. `Canonical Design Plan v1.44.md`
2. `src/kmw1/preprocessing/pipeline.py`
3. `src/kmw1/models/model.py`
4. `src/kmw1/losses/loss.py`
5. `src/kmw1/training/trainer.py`
6. `src/kmw1/evaluation/evaluate.py`
7. `src/kmw1/data/dataset.py`
8. `src/kmw1/cli/main.py`

This set is usually enough to understand:
- architecture
- tensors
- canonicalization
- loss
- training flow
- eval flow
- data flow
- CLI entrypoints

### Read next only if needed
9. `scripts/build_manifest_full.py`
10. `configs/train_main.yaml`
11. `configs/eval.yaml`
12. `scripts/run_main.sh`
13. `scripts/run_eval.sh`

### Read only for run reproduction / PST debugging
14. `Canonical Reindexer Implementation.txt`
15. the exact run’s:
   - `loss_config_kmw1.json`
   - `eval_config_*.json`
   - `*_per_circuit*.csv`
   - `*_summary*.json`

### Files Claude should avoid reading by default
Do **not** expand token usage on these unless the task explicitly requires historical comparison:
- old learned-reindexer `src/kmw/*`
- old v1_1 documents
- older canonical revisions (`v1.41`, `v1.42`, `v1.43`) if `v1.44` already answers the question

---

## Full Research Spec

The authoritative research spec for the current branch is:

```text
Canonical Design Plan v1.44.md
```

This is the main architecture / dataset / training / evaluation authority for the current canonical branch.

If there is any conflict between implementation guesses and the design plan, follow the design plan first, then inspect code.

---

## Practical Summary for Claude

If you need the shortest correct mental model:

- this is the **kmw1 canonical-hardware branch**
- hardware only is canonicalized
- logical order stays unchanged
- the mapper is still the streamlined v1.4 U-Net + hardware-token attention model
- no learned reindexer exists
- logits decode back to native order before assignment/loss
- loss is v1.4.1 execution-surrogate in native frame
- assignment is active-row-only
- real-circuit label-free staged training is now part of the main branch
- routed hard metrics are mandatory for real comparisons
- `Canonical Design Plan v1.44.md` is the main source of truth
