# =============================================================================
# REAL FULL RUN: train on QUEKO + MQT, evaluate on QASMBench + RevLib
# =============================================================================

conda activate graphqmap_pascal
cd ~/KMWs_workspace/GraphQMap/KMW
export PYTHONPATH="$PWD/src"

# -----------------------------------------------------------------------------
# 0) manifest paths
# -----------------------------------------------------------------------------

TRAIN_MANIFEST=data/manifests/full/recipes/train_queko_mqt_bench/train.jsonl

QASMBENCH_VAL=data/manifests/full/recipes/heldout_qasmbench/val.jsonl
QASMBENCH_TEST=data/manifests/full/recipes/heldout_qasmbench/test.jsonl

REVLIB_VAL=data/manifests/full/recipes/heldout_revlib/val.jsonl
REVLIB_TEST=data/manifests/full/recipes/heldout_revlib/test.jsonl

RUN_DIR=artifacts/fullrun_queko_mqt_seed20260324
CKPT="$RUN_DIR/latest.pt"

mkdir -p "$RUN_DIR"

# -----------------------------------------------------------------------------
# 1) real training run
# -----------------------------------------------------------------------------
# Notes:
# - 50 epochs = current trainer default, and appropriate for a real run here
# - batch size stays at 1 (current stable project default)
# - latest.pt will be refreshed every epoch
# - pin_memory is useful on CUDA
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
# 2) confirm checkpoint exists
# -----------------------------------------------------------------------------

ls -lh "$RUN_DIR"
test -f "$CKPT" && echo "Checkpoint OK: $CKPT"

# -----------------------------------------------------------------------------
# 3) held-out validation: QASMBench
# -----------------------------------------------------------------------------

python -m kmw.cli.main eval \
  --manifest "$QASMBENCH_VAL" \
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
  --per-circuit-csv "$RUN_DIR/qasmbench_val_per_circuit.csv" \
  --summary-json "$RUN_DIR/qasmbench_val_summary.json" \
  --eval-split-name qasmbench_val

# -----------------------------------------------------------------------------
# 4) held-out test: QASMBench
# -----------------------------------------------------------------------------

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
  --per-circuit-csv "$RUN_DIR/qasmbench_test_per_circuit.csv" \
  --summary-json "$RUN_DIR/qasmbench_test_summary.json" \
  --eval-split-name qasmbench_test

# -----------------------------------------------------------------------------
# 5) held-out validation: RevLib
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
  --eval-split-name revlib_val

# -----------------------------------------------------------------------------
# 6) held-out test: RevLib
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
  --eval-split-name revlib_test

# -----------------------------------------------------------------------------
# 7) final output check
# -----------------------------------------------------------------------------

find "$RUN_DIR" -maxdepth 1 -type f | sort


