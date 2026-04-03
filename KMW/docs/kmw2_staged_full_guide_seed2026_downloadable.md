# =============================================================================
# KMW2 STAGED FULL RUN GUIDE (EXISTING SPLITS / FLEXIBLE CLI / REAL PST ENABLED)
# =============================================================================
#
# Purpose
# -------
# This run sheet is the current staged-training guide for the kmw2 branch.
#
# - training side:
#     Stage 1: QUEKO(train)
#     Stage 2: QUEKO(train) + QASMBench(train)
#     Stage 3: QUEKO(train) + QASMBench(train)
#
# - held-out evaluation side:
#     QASMBench + RevLib
#
# - final evaluation mode:
#     routed downstream evaluation enabled
#
# Current kmw2 assumptions
# ------------------------
# - package namespace: src/kmw2
# - CLI entrypoint: python -m kmw2.cli.main
# - you copied only src/kmw2 into the repo
# - you are using the already-existing split manifests under:
#     data/manifests/full/source_manifests/...
# - do NOT rebuild manifests for this setup
#
# Locked/default experiment target used here
# ------------------------------------------
# - run directory:
#     artifacts/fullrun_queko_qasm_kmw2_staged_seed2026
# - backend_name = FakeTorontoV2
# - global batch size = 8
# - staged schedule = 15 / 35 / 10 epochs
# - Stage 3 lower LR = 5e-5
# - PST callable = kmw2.metrics.pst:pst_overlap_percent
#
# Important notes
# ---------------
# - The existing source_manifests/.../*.jsonl files are the source of truth.
# - Do NOT run build_manifests.py for this setup.
# - Do NOT pass --epochs during train-staged unless you want to overwrite all
#   stage epoch counts.
# - Do NOT pass --lr during train-staged unless you want to overwrite Stage 3's
#   lower LR too.
# - Because only src/kmw2 was copied, this guide creates the needed YAML config
#   files locally before training/evaluation.
#
# =============================================================================


## Assumptions

These commands assume:

- repo root: `~/KMWs_workspace/GraphQMap/KMW`
- package code exists at: `~/KMWs_workspace/GraphQMap/KMW/src/kmw2`
- existing manifests already exist under:
  - `data/manifests/full/source_manifests/queko/train.jsonl`
  - `data/manifests/full/source_manifests/qasmbench/train.jsonl`
  - `data/manifests/full/source_manifests/qasmbench/val.jsonl`
  - `data/manifests/full/source_manifests/qasmbench/test.jsonl`
  - `data/manifests/full/source_manifests/revlib/val.jsonl`
  - `data/manifests/full/source_manifests/revlib/test.jsonl`

> Do **not** run `build_manifests.py` for this setup. The existing split files are the source of truth.

---

## 1) Environment setup

```bash
conda activate graphqmap_pascal
cd ~/KMWs_workspace/GraphQMap/KMW

export PYTHONPATH="$PWD/src:$PYTHONPATH"

# environment setup
python -m pip install -U pyyaml numpy pandas qiskit qiskit-ibm-runtime qiskit-aer
```

> If `torch` is not already installed in that environment, install the correct PyTorch build for your CUDA setup first.

### Quick sanity checks

```bash
python - <<'PY'
import kmw2
import yaml
print("kmw2 import ok")
PY

python - <<'PY'
from qiskit_aer import AerSimulator
print("qiskit_aer import ok")
PY
```

---

## 2) Create the YAML config files

```bash
mkdir -p configs/overrides
mkdir -p artifacts/fullrun_queko_qasm_kmw2_staged_seed2026
```

### `configs/base.yaml`

```bash
cat > configs/base.yaml <<'YAML'
backend: FakeTorontoV2

model:
  in_channels: 5
  token_dim: 128

loss:
  lambda_p: 1.0
  lambda_s: 0.1
  lambda_d: 0.1
  kappa: 1.0

training:
  epochs: 50
  batch_size: 8
  num_workers: 0
  lr: 1.0e-4
  weight_decay: 0.0
  scheduler_patience: 5
  scheduler_factor: 0.5
  grad_clip_norm: 1.0
  sinkhorn_tau: 0.5
  sinkhorn_iters: 30
  seed: 20260402
  save_epoch_checkpoints: true

paths:
  run_dir: artifacts/fullrun_queko_qasm_kmw2_staged_seed2026
  checkpoint_name: model_final.pth
  eval_dir: artifacts/fullrun_queko_qasm_kmw2_staged_seed2026/eval

runtime:
  device: cuda

dataset:
  manifest_path: data/manifests/full/source_manifests/queko/train.jsonl
YAML
```

### `configs/train_staged.yaml`

```bash
cat > configs/train_staged.yaml <<'YAML'
paths:
  run_dir: artifacts/fullrun_queko_qasm_kmw2_staged_seed2026
  checkpoint_name: model_final.pth

validation:
  manifest_path: data/manifests/full/source_manifests/qasmbench/val.jsonl

staged_training:
  final_checkpoint: last
YAML
```

### `configs/overrides/trainer_default.yaml`

```bash
cat > configs/overrides/trainer_default.yaml <<'YAML'
epochs: 60 
batch_size: 8
num_workers: 0
lr: 1.0e-4
weight_decay: 0.0
scheduler_patience: 5
scheduler_factor: 0.5
grad_clip_norm: 1.0
sinkhorn_tau: 0.5
sinkhorn_iters: 30
save_epoch_checkpoints: true
YAML
```

### `configs/overrides/loss_default.yaml`

```bash
cat > configs/overrides/loss_default.yaml <<'YAML'
lambda_p: 1.0
lambda_s: 0.1
lambda_d: 0.1
kappa: 1.0
YAML
```

### `configs/overrides/stages_existing_splits.yaml`

```bash
cat > configs/overrides/stages_existing_splits.yaml <<'YAML'
final_checkpoint: last
stages:
  - name: stage1_synthetic_warmup
    epochs: 22
    lr: 1.0e-4
    batch_size: 8
    num_workers: 0
    dataset:
      manifest_path: data/manifests/full/source_manifests/queko/train.jsonl
      allowed_sources: [queko]
    sampler:
      kind: source_balanced
      epoch_samples: 1024

  - name: stage2_mixed_synthetic_real
    epochs: 53
    lr: 1.0e-4
    batch_size: 8
    num_workers: 0
    load_from: previous_best
    dataset:
      manifest_paths:
        - data/manifests/full/source_manifests/queko/train.jsonl
        - data/manifests/full/source_manifests/qasmbench/train.jsonl
      allowed_sources: [queko, qasmbench]
    sampler:
      kind: group_balanced
      epoch_samples: 1024
      groups:
        synthetic: [queko]
        real: [qasmbench]
      group_weights:
        synthetic: 0.7
        real: 0.3

  - name: stage3_real_calibration
    epochs: 15
    lr: 5.0e-5
    batch_size: 8
    num_workers: 0
    load_from: stage2_mixed_synthetic_real:best
    dataset:
      manifest_paths:
        - data/manifests/full/source_manifests/queko/train.jsonl
        - data/manifests/full/source_manifests/qasmbench/train.jsonl
      allowed_sources: [queko, qasmbench]
    sampler:
      kind: group_balanced
      epoch_samples: 1024
      groups:
        synthetic: [queko]
        real: [qasmbench]
      group_weights:
        synthetic: 0.5
        real: 0.5
YAML
```

### `configs/overrides/eval_qasmbench_val.yaml`

```bash
cat > configs/overrides/eval_qasmbench_val.yaml <<'YAML'
manifest_path: data/manifests/full/source_manifests/qasmbench/val.jsonl
pst_callable: kmw2.metrics.pst:pst_overlap_percent
YAML
```

### `configs/overrides/eval_qasmbench_test.yaml`

```bash
cat > configs/overrides/eval_qasmbench_test.yaml <<'YAML'
manifest_path: data/manifests/full/source_manifests/qasmbench/test.jsonl
pst_callable: kmw2.metrics.pst:pst_overlap_percent
YAML
```

### `configs/overrides/eval_revlib_val.yaml`

```bash
cat > configs/overrides/eval_revlib_val.yaml <<'YAML'
manifest_path: data/manifests/full/source_manifests/revlib/val.jsonl
pst_callable: kmw2.metrics.pst:pst_overlap_percent
YAML
```

### `configs/overrides/eval_revlib_test.yaml`

```bash
cat > configs/overrides/eval_revlib_test.yaml <<'YAML'
manifest_path: data/manifests/full/source_manifests/revlib/test.jsonl
pst_callable: kmw2.metrics.pst:pst_overlap_percent
YAML
```

> These configs match the current manifest-based staged workflow, with the run directory fixed to `artifacts/fullrun_queko_qasm_kmw2_staged_seed2026`, global batch size fixed to `8`, and PST enabled through `kmw2.metrics.pst:pst_overlap_percent`.

---

## 3) Sanity-check that the manifest files actually exist

```bash
ls data/manifests/full/source_manifests/queko/train.jsonl
ls data/manifests/full/source_manifests/qasmbench/train.jsonl
ls data/manifests/full/source_manifests/qasmbench/val.jsonl
ls data/manifests/full/source_manifests/qasmbench/test.jsonl
ls data/manifests/full/source_manifests/revlib/val.jsonl
ls data/manifests/full/source_manifests/revlib/test.jsonl
```

---

## 4) Run staged training

```bash
python -m kmw2.cli.main train-staged \
  --config configs/base.yaml \
  --config configs/train_staged.yaml \
  --trainer-config configs/overrides/trainer_default.yaml \
  --loss-config configs/overrides/loss_default.yaml \
  --stage-config configs/overrides/stages_existing_splits.yaml \
  --run-dir artifacts/fullrun_queko_qasm_kmw2_staged_seed2026 \
  --checkpoint-name model_final.pth \
  --batch-size 8 \
  --num-workers 0 \
  --backend-name FakeTorontoV2 \
  --device cuda
```

### Important notes

- Do **not** pass `--epochs` here unless you want to overwrite **all stage epoch counts**.
- Do **not** pass `--lr` here unless you want to overwrite **Stage 3's lower LR too**.
- With the command above:
  - Stage 1 stays **15 epochs**
  - Stage 2 stays **35 epochs**
  - Stage 3 stays **10 epochs**
  - Stage 3 LR stays **5e-5**
- Only batch size is globally overridden to **8**.

---

## 5) Run evaluation

### QASMBench validation

```bash
python -m kmw2.cli.main eval \
  --config configs/base.yaml \
  --eval-config configs/overrides/eval_qasmbench_val.yaml \
  --run-dir artifacts/fullrun_queko_qasm_kmw2_staged_seed2026 \
  --checkpoint-name model_final.pth \
  --backend-name FakeTorontoV2 \
  --device cuda \
  --per-circuit-csv artifacts/fullrun_queko_qasm_kmw2_staged_seed2026/eval/qasmbench_val/per_circuit.csv \
  --per-circuit-json artifacts/fullrun_queko_qasm_kmw2_staged_seed2026/eval/qasmbench_val/per_circuit.json \
  --summary-json artifacts/fullrun_queko_qasm_kmw2_staged_seed2026/eval/qasmbench_val/summary.json
```

### QASMBench test

```bash
python -m kmw2.cli.main eval \
  --config configs/base.yaml \
  --eval-config configs/overrides/eval_qasmbench_test.yaml \
  --run-dir artifacts/fullrun_queko_qasm_kmw2_staged_seed2026 \
  --checkpoint-name model_final.pth \
  --backend-name FakeTorontoV2 \
  --device cuda \
  --per-circuit-csv artifacts/fullrun_queko_qasm_kmw2_staged_seed2026/eval/qasmbench_test/per_circuit.csv \
  --per-circuit-json artifacts/fullrun_queko_qasm_kmw2_staged_seed2026/eval/qasmbench_test/per_circuit.json \
  --summary-json artifacts/fullrun_queko_qasm_kmw2_staged_seed2026/eval/qasmbench_test/summary.json
```

### RevLib validation

```bash
python -m kmw2.cli.main eval \
  --config configs/base.yaml \
  --eval-config configs/overrides/eval_revlib_val.yaml \
  --run-dir artifacts/fullrun_queko_qasm_kmw2_staged_seed2026 \
  --checkpoint-name model_final.pth \
  --backend-name FakeTorontoV2 \
  --device cuda \
  --per-circuit-csv artifacts/fullrun_queko_qasm_kmw2_staged_seed2026/eval/revlib_val/per_circuit.csv \
  --per-circuit-json artifacts/fullrun_queko_qasm_kmw2_staged_seed2026/eval/revlib_val/per_circuit.json \
  --summary-json artifacts/fullrun_queko_qasm_kmw2_staged_seed2026/eval/revlib_val/summary.json
```

### RevLib test

```bash
python -m kmw2.cli.main eval \
  --config configs/base.yaml \
  --eval-config configs/overrides/eval_revlib_test.yaml \
  --run-dir artifacts/fullrun_queko_qasm_kmw2_staged_seed2026 \
  --checkpoint-name model_final.pth \
  --backend-name FakeTorontoV2 \
  --device cuda \
  --per-circuit-csv artifacts/fullrun_queko_qasm_kmw2_staged_seed2026/eval/revlib_test/per_circuit.csv \
  --per-circuit-json artifacts/fullrun_queko_qasm_kmw2_staged_seed2026/eval/revlib_test/per_circuit.json \
  --summary-json artifacts/fullrun_queko_qasm_kmw2_staged_seed2026/eval/revlib_test/summary.json
```

---

## 6) Optional: save routed QASM/QPY too

Add these flags to any eval command:

```bash
--save-routed-qasm-dir artifacts/fullrun_queko_qasm_kmw2_staged_seed2026/routed_qasm/<split_name> \
--save-routed-qpy-dir artifacts/fullrun_queko_qasm_kmw2_staged_seed2026/routed_qpy/<split_name>
```

### Example: QASMBench test with routed outputs saved

```bash
python -m kmw2.cli.main eval \
  --config configs/base.yaml \
  --eval-config configs/overrides/eval_qasmbench_test.yaml \
  --run-dir artifacts/fullrun_queko_qasm_kmw2_staged_seed2026 \
  --checkpoint-name model_final.pth \
  --backend-name FakeTorontoV2 \
  --device cuda \
  --per-circuit-csv artifacts/fullrun_queko_qasm_kmw2_staged_seed2026/eval/qasmbench_test/per_circuit.csv \
  --per-circuit-json artifacts/fullrun_queko_qasm_kmw2_staged_seed2026/eval/qasmbench_test/per_circuit.json \
  --summary-json artifacts/fullrun_queko_qasm_kmw2_staged_seed2026/eval/qasmbench_test/summary.json \
  --save-routed-qasm-dir artifacts/fullrun_queko_qasm_kmw2_staged_seed2026/routed_qasm/qasmbench_test \
  --save-routed-qpy-dir artifacts/fullrun_queko_qasm_kmw2_staged_seed2026/routed_qpy/qasmbench_test
```

---

## 7) Practical note about PST

With the YAMLs above:

```yaml
pst_callable: kmw2.metrics.pst:pst_overlap_percent
```

So eval should produce:

- `pst`
- `compile_seconds`
- `swap_overhead`
- `depth_increase`

If `pst` still comes out `null`, the likely causes are:

- `qiskit-aer` is not installed
- `kmw2.metrics.pst` is not importable
- `src/kmw2/evaluation/evaluate.py` was not patched correctly to use the PST callable on a measured circuit

---

# Quick overview: how to change experiment settings

## Change the save directory

Edit the command-line `--run-dir`.

Example:

```bash
--run-dir artifacts/my_new_run_name
```

Also update the eval output paths if you want reports saved under the same new directory.

You can also change the default inside:

- `configs/base.yaml`
- `configs/train_staged.yaml`

but the CLI flag takes precedence.

---

## Change global batch size

Easiest way: change the CLI flag:

```bash
--batch-size 8
```

That applies to **all stages** in `train-staged`.

You can also edit batch sizes inside:

- `configs/overrides/trainer_default.yaml`
- `configs/overrides/stages_existing_splits.yaml`

### Recommendation

Use the CLI for quick changes, and YAML if you want the setting permanently recorded in the config files.

---

## Change stage epochs

Edit:

```text
configs/overrides/stages_existing_splits.yaml
```

Specifically:

- Stage 1 `epochs`
- Stage 2 `epochs`
- Stage 3 `epochs`

> Avoid passing `--epochs` to `train-staged` unless you intentionally want to overwrite every stage with the same epoch count.

---

## Change learning rate

### If you want stage-specific learning rates

Edit them inside:

```text
configs/overrides/stages_existing_splits.yaml
```

### If you want one global LR everywhere

Use:

```bash
--lr ...
```

> For staged training, this overwrites all stage LRs, including Stage 3's lower LR.

---

## Change backend

Use:

```bash
--backend-name FakeTorontoV2
```

or edit:

```text
configs/base.yaml
```

under:

```yaml
backend: ...
```

---

## Change device

Use:

```bash
--device cuda
```

or:

```bash
--device cpu
```

---

## Change held-out eval split

Edit the corresponding eval YAML:

- `configs/overrides/eval_qasmbench_val.yaml`
- `configs/overrides/eval_qasmbench_test.yaml`
- `configs/overrides/eval_revlib_val.yaml`
- `configs/overrides/eval_revlib_test.yaml`

by changing:

```yaml
manifest_path: ...
```

---

## Change loss weights

Edit:

```text
configs/overrides/loss_default.yaml
```

for:

- `lambda_p`
- `lambda_s`
- `lambda_d`
- `kappa`

---

## Change sampler behavior

Edit:

```text
configs/overrides/stages_existing_splits.yaml
```

This is where you control:

- stage dataset manifests
- `source_balanced` vs `group_balanced`
- `group_weights`
- `epoch_samples`

---

## Rule of thumb

Use **CLI flags** for:

- run directory
- batch size
- device
- backend
- output paths

Use **YAML edits** for:

- staged schedule
- stage-specific LR
- stage-specific epochs
- sampler/group weights
- manifest definitions
- loss settings
