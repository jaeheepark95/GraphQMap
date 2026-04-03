# =============================================================================
# REAL FULL RUN (ROUTED FINAL EVAL): train on QUEKO + MQT, evaluate on QASMBench + RevLib
# =============================================================================
#
# Purpose
# -------
# This run sheet is the "real" full-run version for the current codebase:
#
# - training side:
#     QUEKO + MQT Bench
#   (MLQD intentionally excluded here by project decision)
#
# - held-out evaluation side:
#     QASMBench + RevLib
#
# - final evaluation mode:
#     routed downstream evaluation enabled
#     with real post-routing metrics:
#         PST (gate + readout)
#         compile time
#         SWAP overhead
#         depth increase
#
# Important note about old checkpoints
# ------------------------------------
# A checkpoint trained BEFORE the routing update is still valid for this file,
# as long as:
# - the model architecture in src/kmw/models/model.py is unchanged, and
# - the checkpoint contains the usual "model" state_dict.
#
# Routing was added on the evaluation side only.
# So you can skip the training section below and start from the evaluation
# sections using your already-trained checkpoint.
#
# Locked routed-eval settings
# ---------------------------
# - --route-final-eval
# - --routing-method sabre
# - --transpile-optimization-level 0
# - --seed-transpiler 20260324
# - readout error included in real PST
#
# Stable defaults kept from the current project phase
# ---------------------------------------------------
# - batch size = 1
# - num_workers = 0
# - pin_memory on CUDA
# - tau_r = 1.0
# - tau_m = 0.10
# - lambda_p = 1.0
# - lambda_s = 1.0
# - lambda_d = 0.25
# - kappa_depth = 1.0
# - alpha_loc = 0.0
# - beta_cons = 0.0
#
# =============================================================================


# -----------------------------------------------------------------------------
# 0) environment
# -----------------------------------------------------------------------------

conda activate graphqmap_pascal
cd ~/KMWs_workspace/GraphQMap/KMW
export PYTHONPATH="$PWD/src"


# -----------------------------------------------------------------------------
# 1) manifest paths
# -----------------------------------------------------------------------------
# Intentional project choice:
# training uses QUEKO + MQT only
# held-out eval uses QASMBench + RevLib
# -----------------------------------------------------------------------------

TRAIN_MANIFEST=data/manifests/full/recipes/train_queko_mqt_bench/train.jsonl

QASMBENCH_VAL=data/manifests/full/recipes/heldout_qasmbench/val.jsonl
QASMBENCH_TEST=data/manifests/full/recipes/heldout_qasmbench/test.jsonl

REVLIB_VAL=data/manifests/full/recipes/heldout_revlib/val.jsonl
REVLIB_TEST=data/manifests/full/recipes/heldout_revlib/test.jsonl


# -----------------------------------------------------------------------------
# 2) run directory and checkpoint selection
# -----------------------------------------------------------------------------
# Choose ONE of the following:
#
# A) fresh full run:
#    use RUN_DIR below and run the training section
#
# B) evaluate an already-trained checkpoint from before routing was added:
#    set CKPT to that old checkpoint path and SKIP the training section
# -----------------------------------------------------------------------------

RUN_DIR=artifacts/fullrun_queko_mqt_routed_seed20260324
mkdir -p "$RUN_DIR"

# Fresh-run default checkpoint path:
CKPT="$RUN_DIR/latest.pt"

# If you already trained earlier and want to reuse that checkpoint directly,
# comment out the line above and set CKPT to the old saved checkpoint.
# Example:
# CKPT=artifacts/fullrun_queko_mqt_seed20260324/latest.pt

# Optional artifact directories for saved routed circuits
ROUTED_QASM_DIR="$RUN_DIR/routed_qasm"
ROUTED_QPY_DIR="$RUN_DIR/routed_qpy"

mkdir -p "$ROUTED_QASM_DIR"
mkdir -p "$ROUTED_QPY_DIR"


# -----------------------------------------------------------------------------
# 3) OPTIONAL: real training run
# -----------------------------------------------------------------------------
# Run this section only if you need a fresh checkpoint.
#
# If you already have a trained checkpoint from before the routing update,
# SKIP this section and go directly to:
#   4) confirm checkpoint exists
#
# Notes:
# - 50 epochs = current real-run setting
# - batch size stays at 1
# - latest.pt will be refreshed every epoch
# -----------------------------------------------------------------------------

python -m kmw.cli.main train \
  --manifest "$TRAIN_MANIFEST" \
  --device cuda \
  --epochs 50 \
  --seed 20260324 \
  --batch-size 1 \
  --num-workers 0 \
  --pin-memory \
  --mapper-lr 1e-4 \
  --reindexer-lr 1e-4 \
  --weight-decay 0.0 \
  --grad-clip-norm 1.0 \
  --log-every-steps 10 \
  --save-every-epochs 1 \
  --tau-r 1.0 \
  --tau-m 0.10 \
  --lambda-p 1.0 \
  --lambda-s 1.0 \
  --lambda-d 0.25 \
  --kappa-depth 1.0 \
  --alpha-loc 0.0 \
  --beta-cons 0.0 \
  --checkpoint-dir "$RUN_DIR" \
  --checkpoint-prefix fullrun_queko_mqt


# -----------------------------------------------------------------------------
# 4) confirm checkpoint exists
# -----------------------------------------------------------------------------
# This must pass whether the checkpoint was:
# - produced by section 3 above, or
# - produced earlier before routing was added
# -----------------------------------------------------------------------------

ls -lh "$RUN_DIR" 2>/dev/null || true
test -f "$CKPT" && echo "Checkpoint OK: $CKPT"


# -----------------------------------------------------------------------------
# 5) held-out validation: QASMBench (ROUTED FINAL EVAL, NO ROUTED ARTIFACT SAVE)
# -----------------------------------------------------------------------------
# Use this validation form first if earlier routed-eval runs returned:
#   - PermissionError
#   - success_count = 0
#   - null summary statistics
#
# This still performs real routed downstream evaluation, but it temporarily
# disables routed QASM/QPY export so the validation pass is not blocked by a
# filesystem write failure.
#
# If this succeeds cleanly and you later still want saved routed artifacts,
# re-enable the two save flags after confirming the target directories are
# writable under the current user.
# -----------------------------------------------------------------------------

python -m kmw.cli.main eval   --manifest "$QASMBENCH_VAL"   --checkpoint "$CKPT"   --device cuda   --seed 20260324   --batch-size 1   --num-workers 0   --pin-memory   --tau-m 0.10   --lambda-p 1.0   --lambda-s 1.0   --lambda-d 0.25   --kappa-depth 1.0   --alpha-loc 0.0   --beta-cons 0.0   --per-circuit-csv "$RUN_DIR/qasmbench_val_per_circuit_nosave.csv"   --summary-json "$RUN_DIR/qasmbench_val_summary_nosave.json"   --eval-split-name qasmbench_val   --route-final-eval   --routing-method sabre   --transpile-optimization-level 0   --seed-transpiler 20260324   --include-readout-in-pst


# -----------------------------------------------------------------------------
# 6) held-out test: QASMBench (ROUTED FINAL EVAL)
# -----------------------------------------------------------------------------

conda activate graphqmap_pascal
cd ~/KMWs_workspace/GraphQMap/KMW
export PYTHONPATH="$PWD/src"

QASMBENCH_TEST=data/manifests/full/recipes/heldout_qasmbench/test.jsonl
RUN_DIR=artifacts/fullrun_queko_mqt_routed_seed20260324
mkdir -p "$RUN_DIR"

CKPT=artifacts/fullrun_queko_mqt_seed20260324/latest.pt

test -f "$CKPT" && echo "Checkpoint OK: $CKPT" || echo "Missing checkpoint: $CKPT"

<-----

python -m kmw.cli.main eval \
  --manifest "$QASMBENCH_TEST" \
  --checkpoint "$CKPT" \
  --device cuda \
  --seed 20260324 \
  --batch-size 1 \
  --num-workers 0 \
  --pin-memory \
  --tau-m 0.10 \
  --lambda-p 1.0 \
  --lambda-s 1.0 \
  --lambda-d 0.25 \
  --kappa-depth 1.0 \
  --alpha-loc 0.0 \
  --beta-cons 0.0 \
  --per-circuit-csv "$RUN_DIR/qasmbench_test_per_circuit_nosave.csv" \
  --summary-json "$RUN_DIR/qasmbench_test_summary_nosave.json" \
  --eval-split-name qasmbench_test \
  --route-final-eval \
  --routing-method sabre \
  --transpile-optimization-level 0 \
  --seed-transpiler 20260324 \
  --include-readout-in-pst


# -----------------------------------------------------------------------------
# 7) held-out validation: RevLib (ROUTED FINAL EVAL)
# -----------------------------------------------------------------------------

python -m kmw.cli.main eval \
  --manifest "$REVLIB_VAL" \
  --checkpoint "$CKPT" \
  --device cuda \
  --seed 20260324 \
  --batch-size 1 \
  --num-workers 0 \
  --pin-memory \
  --tau-m 0.10 \
  --lambda-p 1.0 \
  --lambda-s 1.0 \
  --lambda-d 0.25 \
  --kappa-depth 1.0 \
  --alpha-loc 0.0 \
  --beta-cons 0.0 \
  --per-circuit-csv "$RUN_DIR/revlib_val_per_circuit.csv" \
  --summary-json "$RUN_DIR/revlib_val_summary.json" \
  --eval-split-name revlib_val \
  --route-final-eval \
  --routing-method sabre \
  --transpile-optimization-level 0 \
  --seed-transpiler 20260324 \
  --include-readout-in-pst \
  --save-routed-qasm-dir "$ROUTED_QASM_DIR/revlib_val" \
  --save-routed-qpy-dir "$ROUTED_QPY_DIR/revlib_val"


# -----------------------------------------------------------------------------
# 8) held-out test: RevLib (ROUTED FINAL EVAL)
# -----------------------------------------------------------------------------

python -m kmw.cli.main eval \
  --manifest "$REVLIB_TEST" \
  --checkpoint "$CKPT" \
  --device cuda \
  --seed 20260324 \
  --batch-size 1 \
  --num-workers 0 \
  --pin-memory \
  --tau-m 0.10 \
  --lambda-p 1.0 \
  --lambda-s 1.0 \
  --lambda-d 0.25 \
  --kappa-depth 1.0 \
  --alpha-loc 0.0 \
  --beta-cons 0.0 \
  --per-circuit-csv "$RUN_DIR/revlib_test_per_circuit.csv" \
  --summary-json "$RUN_DIR/revlib_test_summary.json" \
  --eval-split-name revlib_test \
  --route-final-eval \
  --routing-method sabre \
  --transpile-optimization-level 0 \
  --seed-transpiler 20260324 \
  --include-readout-in-pst \
  --save-routed-qasm-dir "$ROUTED_QASM_DIR/revlib_test" \
  --save-routed-qpy-dir "$ROUTED_QPY_DIR/revlib_test"


# -----------------------------------------------------------------------------
# 9) final output check
# -----------------------------------------------------------------------------

find "$RUN_DIR" -maxdepth 2 -type f | sort


# -----------------------------------------------------------------------------
# 10) quick result files to inspect
# -----------------------------------------------------------------------------
# Expected main outputs:
#
# - qasmbench_val_per_circuit.csv
# - qasmbench_val_summary.json
# - qasmbench_test_per_circuit.csv
# - qasmbench_test_summary.json
# - revlib_val_per_circuit.csv
# - revlib_val_summary.json
# - revlib_test_per_circuit.csv
# - revlib_test_summary.json
#
# In routed mode, successful rows should include real routed fields such as:
# - real_pst_gate_readout
# - swap_inserted_count
# - added_2q_ops
# - routing_compile_time_s
# - original_depth
# - routed_depth
# - depth_increase_abs
# - depth_increase_ratio
# -----------------------------------------------------------------------------




conda activate graphqmap_pascal
cd ~/KMWs_workspace/GraphQMap/KMW
export PYTHONPATH="$PWD/src"

RUN_DIR=artifacts/fullrun_queko_mqt_routed_seed20260324
mkdir -p "$RUN_DIR"

CKPT=artifacts/fullrun_queko_mqt_seed20260324/latest.pt
test -f "$CKPT" && echo "Checkpoint OK: $CKPT" || echo "Missing checkpoint: $CKPT"

# Then use this small manifest auto-detection block so you do not have to guess whether your repo uses recipes/train_* or source_manifests/* paths:

if [ -f data/manifests/full/recipes/train_queko/test.jsonl ]; then
  QUEKO_TEST=data/manifests/full/recipes/train_queko/test.jsonl
else
  QUEKO_TEST=data/manifests/full/source_manifests/queko/test.jsonl
fi

if [ -f data/manifests/full/recipes/train_mqt_bench/test.jsonl ]; then
  MQT_TEST=data/manifests/full/recipes/train_mqt_bench/test.jsonl
else
  MQT_TEST=data/manifests/full/source_manifests/mqt_bench/test.jsonl
fi

echo "QUEKO_TEST=$QUEKO_TEST"
echo "MQT_TEST=$MQT_TEST"
test -f "$QUEKO_TEST" && echo "QUEKO manifest OK"
test -f "$MQT_TEST" && echo "MQT manifest OK"

# Now the actual eval commands.

1) QUEKO test performance

python -m kmw.cli.main eval \
  --manifest "$QUEKO_TEST" \
  --checkpoint "$CKPT" \
  --device cuda \
  --seed 20260324 \
  --batch-size 1 \
  --num-workers 0 \
  --pin-memory \
  --tau-m 0.10 \
  --lambda-p 1.0 \
  --lambda-s 1.0 \
  --lambda-d 0.25 \
  --kappa-depth 1.0 \
  --alpha-loc 0.0 \
  --beta-cons 0.0 \
  --per-circuit-csv "$RUN_DIR/queko_test_per_circuit_nosave.csv" \
  --summary-json "$RUN_DIR/queko_test_summary_nosave.json" \
  --eval-split-name queko_test \
  --route-final-eval \
  --routing-method sabre \
  --transpile-optimization-level 0 \
  --seed-transpiler 20260324 \
  --include-readout-in-pst

2) MQT Bench test performance

python -m kmw.cli.main eval \
  --manifest "$MQT_TEST" \
  --checkpoint "$CKPT" \
  --device cuda \
  --seed 20260324 \
  --batch-size 1 \
  --num-workers 0 \
  --pin-memory \
  --tau-m 0.10 \
  --lambda-p 1.0 \
  --lambda-s 1.0 \
  --lambda-d 0.25 \
  --kappa-depth 1.0 \
  --alpha-loc 0.0 \
  --beta-cons 0.0 \
  --per-circuit-csv "$RUN_DIR/mqt_test_per_circuit_nosave.csv" \
  --summary-json "$RUN_DIR/mqt_test_summary_nosave.json" \
  --eval-split-name mqt_test \
  --route-final-eval \
  --routing-method sabre \
  --transpile-optimization-level 0 \
  --seed-transpiler 20260324 \
  --include-readout-in-pst

# And to inspect the summaries afterward:

cat "$RUN_DIR/queko_test_summary_nosave.json"
cat "$RUN_DIR/mqt_test_summary_nosave.json"