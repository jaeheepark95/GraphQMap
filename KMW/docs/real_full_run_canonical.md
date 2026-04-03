# =============================================================================
# REAL FULL RUN (ROUTED FINAL EVAL): train on QUEKO + MQT, evaluate on QASMBench + RevLib
# CANONICAL BRANCH VERSION (src/kmw1 / clean canonical-only CLI)
# =============================================================================
#
# Purpose
# -------
# This run sheet is the canonical-hardware v1.4 replacement for the old learned-
# reindexer route sheet.
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
#
# Important canonical-branch changes
# ----------------------------------
# - package namespace is now: src/kmw1
# - CLI entrypoint is now: python -m kmw1.cli.main
# - no learned reindexer
# - no tau_r schedule
# - no Pass-A / Pass-B split
# - no alpha_loc / beta_cons / freeze_hardware_reindex / canonical pretrain
# - hardware is deterministically canonicalized before the mapper
# - mapper logits are decoded back to native hardware order before task loss
#
# Locked/default canonical settings used here
# ------------------------------------------
# - backend_name = fake_toronto_v2
# - batch size = 1
# - optimizer = AdamW
# - lr = 1e-4
# - weight_decay = 1e-4
# - grad_clip_norm = 1.0
# - sinkhorn_tau = 0.50
# - sinkhorn_iters = 30
# - eps = 1e-8
# - eps_surv = 1e-12
# - route_step_2q_mult = 3.0
# - route_step_1q_mult = 2.0
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
# B) evaluate an already-trained canonical checkpoint:
#    set RUN_DIR to the exact folder containing the kmw1 run outputs
#    and set CKPT to the specific checkpoint you want to evaluate
#
# Note:
# - current kmw1 trainer always saves epoch checkpoints
# - best.pt is only created if --val-manifest was used during training
# -----------------------------------------------------------------------------

RUN_DIR=artifacts/fullrun_queko_mqt_kmw1_canonical_seed20260331
mkdir -p "$RUN_DIR"

CKPT="$RUN_DIR/checkpoints/epoch_060.pt"

ROUTED_QASM_DIR="$RUN_DIR/routed_qasm"
ROUTED_QPY_DIR="$RUN_DIR/routed_qpy"

mkdir -p "$ROUTED_QASM_DIR"
mkdir -p "$ROUTED_QPY_DIR"


# -----------------------------------------------------------------------------
# 3) OPTIONAL: real training run with canonical-hardware branch
# -----------------------------------------------------------------------------
# Run this section only if you need a fresh checkpoint.
# If you already have a trained canonical checkpoint, skip to section 4.
# -----------------------------------------------------------------------------

cat > "$RUN_DIR/trainer_config_kmw1.json" <<'JSON'
{
  "lr": 1e-4,
  "weight_decay": 1e-4,
  "grad_clip_norm": 1.0,
  "log_every_steps": 10,
  "checkpoint_every_epochs": 1,
  "fail_on_nonfinite_grad": true,
  "save_best_on": "val_L_task"
}
JSON

cat > "$RUN_DIR/loss_config_kmw1.json" <<'JSON'
{
  "sinkhorn_tau": 0.50,
  "sinkhorn_iters": 30,

  "eps": 1e-8,
  "eps_surv": 1e-12,

  "route_step_2q_mult": 3.0,
  "route_step_1q_mult": 2.0
}
JSON

python -m kmw1.cli.main train \
  --project-root "$PWD" \
  --manifest "$TRAIN_MANIFEST" \
  --backend-name fake_toronto_v2 \
  --device cuda \
  --epochs 60 \
  --seed 20260331 \
  --batch-size 1 \
  --num-workers 0 \
  --sinkhorn-tau 0.50 \
  --sinkhorn-iters 30 \
  --trainer-config "$RUN_DIR/trainer_config_kmw1.json" \
  --loss-config "$RUN_DIR/loss_config_kmw1.json" \
  --run-dir "$RUN_DIR"

# If you want a best.pt checkpoint instead of using epoch_060.pt, use:
#   --val-manifest "$QASMBENCH_VAL"
# in the train command above, then set:
#   CKPT="$RUN_DIR/checkpoints/best.pt"


# -----------------------------------------------------------------------------
# 4) confirm checkpoint exists
# -----------------------------------------------------------------------------

echo "RUN_DIR=$RUN_DIR"
echo "CKPT=$CKPT"
find "$RUN_DIR/checkpoints" -maxdepth 1 -type f 2>/dev/null | sort || true
test -f "$CKPT" && echo "Checkpoint OK: $CKPT" || echo "Checkpoint missing: $CKPT"


# -----------------------------------------------------------------------------
# 4a) ensure eval config and loss config exist
# -----------------------------------------------------------------------------
# In kmw1, routed-eval knobs are passed through --eval-config JSON.
# -----------------------------------------------------------------------------

test -f "$RUN_DIR/loss_config_kmw1.json" || cat > "$RUN_DIR/loss_config_kmw1.json" <<'JSON'
{
  "sinkhorn_tau": 0.50,
  "sinkhorn_iters": 30,

  "eps": 1e-8,
  "eps_surv": 1e-12,

  "route_step_2q_mult": 3.0,
  "route_step_1q_mult": 2.0
}
JSON

cat > "$RUN_DIR/eval_config_base.json" <<'JSON'
{
  "print_console_summary": true,
  "fail_fast": false,

  "routing_method": "sabre",
  "transpile_optimization_level": 0,
  "seed_transpiler": 20260325
}
JSON

test -f "$RUN_DIR/loss_config_kmw1.json" && echo "Loss config OK: $RUN_DIR/loss_config_kmw1.json"
test -f "$RUN_DIR/eval_config_base.json" && echo "Eval config OK: $RUN_DIR/eval_config_base.json"


# -----------------------------------------------------------------------------
# 5) held-out validation: QASMBench (ROUTED FINAL EVAL, NO ROUTED ARTIFACT SAVE)
# -----------------------------------------------------------------------------

python -m kmw1.cli.main eval \
  --project-root "$PWD" \
  --manifest "$QASMBENCH_VAL" \
  --checkpoint "$CKPT" \
  --backend-name fake_toronto_v2 \
  --device cuda \
  --seed 20260331 \
  --num-workers 0 \
  --sinkhorn-tau 0.50 \
  --sinkhorn-iters 30 \
  --loss-config "$RUN_DIR/loss_config_kmw1.json" \
  --eval-config "$RUN_DIR/eval_config_base.json" \
  --per-circuit-csv "$RUN_DIR/qasmbench_val_per_circuit_nosave.csv" \
  --summary-json "$RUN_DIR/qasmbench_val_summary_nosave.json" \
  --eval-split qasmbench_val \
  --route-final-eval


# -----------------------------------------------------------------------------
# 6) held-out test: QASMBench (ROUTED FINAL EVAL, NO ROUTED ARTIFACT SAVE)
# -----------------------------------------------------------------------------

python -m kmw1.cli.main eval \
  --project-root "$PWD" \
  --manifest "$QASMBENCH_TEST" \
  --checkpoint "$CKPT" \
  --backend-name fake_toronto_v2 \
  --device cuda \
  --seed 20260331 \
  --num-workers 0 \
  --sinkhorn-tau 0.50 \
  --sinkhorn-iters 30 \
  --loss-config "$RUN_DIR/loss_config_kmw1.json" \
  --eval-config "$RUN_DIR/eval_config_base.json" \
  --per-circuit-csv "$RUN_DIR/qasmbench_test_per_circuit_nosave.csv" \
  --summary-json "$RUN_DIR/qasmbench_test_summary_nosave.json" \
  --eval-split qasmbench_test \
  --route-final-eval


# -----------------------------------------------------------------------------
# 7) held-out validation: RevLib (ROUTED FINAL EVAL, SAVE ROUTED ARTIFACTS)
# -----------------------------------------------------------------------------

cat > "$RUN_DIR/eval_config_revlib_val.json" <<JSON
{
  "print_console_summary": true,
  "fail_fast": false,

  "routing_method": "sabre",
  "transpile_optimization_level": 0,
  "seed_transpiler": 20260325,

  "save_routed_qasm_dir": "$ROUTED_QASM_DIR/revlib_val",
  "save_routed_qpy_dir": "$ROUTED_QPY_DIR/revlib_val"
}
JSON

mkdir -p "$ROUTED_QASM_DIR/revlib_val"
mkdir -p "$ROUTED_QPY_DIR/revlib_val"

python -m kmw1.cli.main eval \
  --project-root "$PWD" \
  --manifest "$REVLIB_VAL" \
  --checkpoint "$CKPT" \
  --backend-name fake_toronto_v2 \
  --device cuda \
  --seed 20260331 \
  --num-workers 0 \
  --sinkhorn-tau 0.50 \
  --sinkhorn-iters 30 \
  --loss-config "$RUN_DIR/loss_config_kmw1.json" \
  --eval-config "$RUN_DIR/eval_config_revlib_val.json" \
  --per-circuit-csv "$RUN_DIR/revlib_val_per_circuit.csv" \
  --summary-json "$RUN_DIR/revlib_val_summary.json" \
  --eval-split revlib_val \
  --route-final-eval


# -----------------------------------------------------------------------------
# 8) held-out test: RevLib (ROUTED FINAL EVAL, SAVE ROUTED ARTIFACTS)
# -----------------------------------------------------------------------------

cat > "$RUN_DIR/eval_config_revlib_test.json" <<JSON
{
  "print_console_summary": true,
  "fail_fast": false,

  "routing_method": "sabre",
  "transpile_optimization_level": 0,
  "seed_transpiler": 20260325,

  "save_routed_qasm_dir": "$ROUTED_QASM_DIR/revlib_test",
  "save_routed_qpy_dir": "$ROUTED_QPY_DIR/revlib_test"
}
JSON

mkdir -p "$ROUTED_QASM_DIR/revlib_test"
mkdir -p "$ROUTED_QPY_DIR/revlib_test"

python -m kmw1.cli.main eval \
  --project-root "$PWD" \
  --manifest "$REVLIB_TEST" \
  --checkpoint "$CKPT" \
  --backend-name fake_toronto_v2 \
  --device cuda \
  --seed 20260331 \
  --num-workers 0 \
  --sinkhorn-tau 0.50 \
  --sinkhorn-iters 30 \
  --loss-config "$RUN_DIR/loss_config_kmw1.json" \
  --eval-config "$RUN_DIR/eval_config_revlib_test.json" \
  --per-circuit-csv "$RUN_DIR/revlib_test_per_circuit.csv" \
  --summary-json "$RUN_DIR/revlib_test_summary.json" \
  --eval-split revlib_test \
  --route-final-eval


# -----------------------------------------------------------------------------
# 9) final output check
# -----------------------------------------------------------------------------

find "$RUN_DIR" -maxdepth 3 -type f | sort


# -----------------------------------------------------------------------------
# 10) quick result files to inspect
# -----------------------------------------------------------------------------
# Expected main outputs from current kmw1 implementation:
#
# - run_config.json
# - logs/train_metrics.jsonl
# - logs/epoch_metrics.jsonl
# - checkpoints/epoch_001.pt ... checkpoints/epoch_060.pt
# - checkpoints/best.pt                       # only if --val-manifest was used during train
# - qasmbench_val_per_circuit_nosave.csv
# - qasmbench_val_summary_nosave.json
# - qasmbench_test_per_circuit_nosave.csv
# - qasmbench_test_summary_nosave.json
# - revlib_val_per_circuit.csv
# - revlib_val_summary.json
# - revlib_test_per_circuit.csv
# - revlib_test_summary.json
#
# Current per-circuit eval rows include fields like:
# - L_task_hard
# - L_native_hard
# - L_route_hard
# - S_proxy_exec_hard
# - native_mapping
# - canonical_p
#
# In routed mode, current kmw1 eval additionally writes routed fields such as:
# - routed_success
# - swap_count
# - routed_2q_count
# - orig_depth
# - routed_depth
# - depth_increase
# -----------------------------------------------------------------------------

