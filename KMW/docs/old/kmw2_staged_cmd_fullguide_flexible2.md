# These commands assume:

>repo root: ~/KMWs_workspace/GraphQMap/KMW
>package code exists at: ~/KMWs_workspace/GraphQMap/KMW/src/kmw2
>existing manifests already exist under:
  >data/manifests/full/source_manifests/queko/train.jsonl
  >data/manifests/full/source_manifests/qasmbench/train.jsonl
  >data/manifests/full/source_manifests/qasmbench/val.jsonl
  >data/manifests/full/source_manifests/qasmbench/test.jsonl
  >data/manifests/full/source_manifests/revlib/val.jsonl
  >data/manifests/full/source_manifests/revlib/test.jsonl

# Do not run build_manifests.py for this setup.

# 1) Environment setup

conda activate graphqmap_pascal
cd ~/KMWs_workspace/GraphQMap/KMW

export PYTHONPATH="$PWD/src:$PYTHONPATH"

python -m pip install -U pyyaml numpy pandas qiskit qiskit-ibm-runtime qiskit-aer

# If torch is not already installed in that env, install the correct build for your CUDA setup first.

# Quick sanity check:

python - <<'PY'
import kmw2
import yaml
print("kmw2 import ok")
PY

# 2) Create the YAML config files
mkdir -p configs/overrides
mkdir -p runs/kmw2

# configs/base.yaml
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
  batch_size: 16
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
  run_dir: runs/kmw2/train_main
  checkpoint_name: model_final.pth
  eval_dir: runs/kmw2/eval

runtime:
  device: cuda

dataset:
  manifest_path: data/manifests/full/source_manifests/queko/train.jsonl
YAML

# configs/train_staged.yaml

cat > configs/train_staged.yaml <<'YAML'
paths:
  run_dir: runs/kmw2/real_full_run_staged
  checkpoint_name: model_final.pth

validation:
  manifest_path: data/manifests/full/source_manifests/qasmbench/val.jsonl

staged_training:
  final_checkpoint: last
YAML

# configs/overrides/trainer_default.yaml
cat > configs/overrides/trainer_default.yaml <<'YAML'
epochs: 60
batch_size: 16
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

# configs/overrides/loss_default.yaml
cat > configs/overrides/loss_default.yaml <<'YAML'
lambda_p: 1.0
lambda_s: 0.1
lambda_d: 0.1
kappa: 1.0
YAML

# configs/overrides/stages_existing_splits.yaml
cat > configs/overrides/stages_existing_splits.yaml <<'YAML'
final_checkpoint: last
stages:
  - name: stage1_synthetic_warmup
    epochs: 15
    lr: 1.0e-4
    batch_size: 16
    num_workers: 0
    dataset:
      manifest_path: data/manifests/full/source_manifests/queko/train.jsonl
      allowed_sources: [queko]
    sampler:
      kind: source_balanced
      epoch_samples: 1024

  - name: stage2_mixed_synthetic_real
    epochs: 35
    lr: 1.0e-4
    batch_size: 16
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
    epochs: 10
    lr: 5.0e-5
    batch_size: 16
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

# configs/overrides/eval_qasmbench_val.yaml
cat > configs/overrides/eval_qasmbench_val.yaml <<'YAML'
manifest_path: data/manifests/full/source_manifests/qasmbench/val.jsonl
pst_callable: kmw2.metrics.pst:pst_overlap_percent
YAML

# configs/overrides/eval_qasmbench_test.yaml
cat > configs/overrides/eval_qasmbench_test.yaml <<'YAML'
manifest_path: data/manifests/full/source_manifests/qasmbench/test.jsonl
pst_callable: kmw2.metrics.pst:pst_overlap_percent
YAML

# configs/overrides/eval_revlib_val.yaml
cat > configs/overrides/eval_revlib_val.yaml <<'YAML'
manifest_path: data/manifests/full/source_manifests/revlib/val.jsonl
pst_callable: kmw2.metrics.pst:pst_overlap_percent
YAML

# configs/overrides/eval_revlib_test.yaml
cat > configs/overrides/eval_revlib_test.yaml <<'YAML'
manifest_path: data/manifests/full/source_manifests/revlib/test.jsonl
pst_callable: kmw2.metrics.pst:pst_overlap_percent
YAML

# 3) Sanity-check that the manifest files actually exist
ls data/manifests/full/source_manifests/queko/train.jsonl
ls data/manifests/full/source_manifests/qasmbench/train.jsonl
ls data/manifests/full/source_manifests/qasmbench/val.jsonl
ls data/manifests/full/source_manifests/qasmbench/test.jsonl
ls data/manifests/full/source_manifests/revlib/val.jsonl
ls data/manifests/full/source_manifests/revlib/test.jsonl

# 4) Run staged training

# Create the artifact root first
mkdir -p artifacts/fullrun_queko_qasm_kmw2_staged_seed2026

# Use this command for training
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

# Important notes
# - Do NOT pass --epochs here unless you want to overwrite all stage epoch counts.
# - Do NOT pass --lr here unless you want to overwrite Stage 3's lower LR too.
# - With the command above:
#   > Stage 1 stays 15 epochs
#   > Stage 2 stays 35 epochs
#   > Stage 3 stays 10 epochs
#   > Stage 3 LR stays 5e-5
# - Only batch size is globally overridden to 8.


# 5) Run evaluation

# QASMBench validation
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

# QASMBench test
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

# RevLib validation
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

# RevLib test
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


# 6) Optional: save routed QASM/QPY too

# Add these flags to any eval command:
# --save-routed-qasm-dir artifacts/fullrun_queko_qasm_kmw2_staged_seed2026/routed_qasm/<split_name> \
# --save-routed-qpy-dir artifacts/fullrun_queko_qasm_kmw2_staged_seed2026/routed_qpy/<split_name>

# Example: QASMBench test with routed outputs saved
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


# 7) One practical note about PST

# With the YAMLs above:

pst_callable: kmw2.metrics.pst:pst_overlap_percent

# So eval should produce:

- pst
- compile_seconds
- swap_overhead
- depth_increase

# If pst is still null, check:
- qiskit-aer installation
- importability of kmw2.metrics.pst
- whether evaluate.py includes the measured-circuit PST patch