<!--
===============================================================================
UPDATE LOG (2026-03-31)
- Updated the training command to match the stabilized v1.31 staged curriculum:
    warmup / stage1 / stage2 = 10 / 25 / 25
- Updated tau_r annealing to the gentler schedule:
    warmup : 1.0 (identity reindex, Pass A only)
    stage1 : 1.0  -> 0.70
    stage2 : 0.70 -> 0.25
- Turned on stage-2 consistency loss:
    stage2_beta_cons = 0.10
- Updated the run sheet to the loss-plan v1.4.1 replacement:
    L_task      = L_native + L_route
    L_native    = L_1Q + L_ro + L_2Q
    final score = S_proxy_exec (surrogate score, not literal PST)
- Removed old lambda_p / lambda_s / lambda_d / kappa_depth usage from this run sheet.
- Added trainer/loss JSON config files so the command lines stay aligned with the
  current staged trainer and v1.4.1 loss implementation.
===============================================================================
-->

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
# A checkpoint trained BEFORE the curriculum update is still valid for this file,
# as long as:
# - the model architecture in src/kmw/models/model.py is unchanged, and
# - the checkpoint contains the usual "model" state_dict.
#
# The staged curriculum was added on the training side only.
# So you can skip the training section below and start from the evaluation
# sections using your already-trained checkpoint.
#
# Locked routed-eval settings
# ---------------------------
# - --route-final-eval
# - --routing-method sabre
# - --transpile-optimization-level 0
# - --seed-transpiler 20260325
# - readout error included in real PST
#
# Recommended stabilized training defaults
# ----------------------------------------
# - batch size = 1
# - num_workers = 0
# - pin_memory on CUDA
# - staged curriculum enabled
# - warmup / stage1 / stage2 = 10 / 25 / 25 epochs
# - tau_r schedule:
#     warmup : 1.0 (identity reindex, Pass A only)
#     stage1 : 1.0  -> 0.70
#     stage2 : 0.70 -> 0.25
# - tau_m = 0.10
# - eps_surv = 1e-12
# - route_step_2q_mult = 3.0
# - route_step_1q_mult = 2.0
# - alpha_loc:
#     warmup : 0.0
#     stage1 : 0.02
#     stage2 : 0.05
# - beta_cons:
#     stage1 : 0.0
#     stage2 : 0.10
# - mapper_lr = 1e-4
# - reindexer_lr = 5e-5
# - weight_decay = 1e-4
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
# B) evaluate an already-trained checkpoint:
#    set CKPT to that checkpoint path and SKIP the training section
# -----------------------------------------------------------------------------

RUN_DIR=artifacts/fullrun_queko_mqt_curriculum_v141_routed_seed20260331
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
# 3) OPTIONAL: real training run with staged curriculum
# -----------------------------------------------------------------------------
# Run this section only if you need a fresh checkpoint.
#
# If you already have a trained checkpoint, SKIP this section and go directly to:
#   4) confirm checkpoint exists
#
# Notes:
# - 60 total epochs = recommended stabilized full run
# - batch size stays at 1
# - latest.pt will be refreshed every epoch
# -----------------------------------------------------------------------------

cat > "$RUN_DIR/trainer_config_v131_v141.json" <<'JSON'
{
  "use_staged_curriculum": true,

  "warmup_epochs": 10,
  "stage1_epochs": 25,
  "stage2_epochs": 25,

  "tau_r_start": 1.0,
  "tau_r_mid": 0.70,
  "tau_r_end": 0.25,
  "tau_r_schedule": "cosine",

  "stage1_alpha_loc": 0.02,
  "stage1_beta_cons": 0.0,
  "stage2_alpha_loc": 0.05,
  "stage2_beta_cons": 0.10,

  "warmup_reindex_mode": "identity",

  "mapper_lr": 1e-4,
  "reindexer_lr": 5e-5,
  "weight_decay": 1e-4,

  "grad_clip_norm": 1.0,
  "log_every_steps": 10,
  "save_every_epochs": 1,

  "freeze_hardware_reindex": false,
  "enable_canonical_pretrain": false,
  "pretrain_epochs": 0
}
JSON

cat > "$RUN_DIR/loss_config_v141.json" <<'JSON'
{
  "tau_m": 0.10,
  "sinkhorn_iters": 20,

  "eps": 1e-6,
  "eps_surv": 1e-12,

  "route_step_2q_mult": 3.0,
  "route_step_1q_mult": 2.0,

  "alpha_loc": 0.0,
  "beta_cons": 0.0
}
JSON

python -m kmw.cli.main train \
  --manifest "$TRAIN_MANIFEST" \
  --device cuda \
  --epochs 60 \
  --seed 20260331 \
  --batch-size 1 \
  --num-workers 0 \
  --pin-memory \
  --mapper-lr 1e-4 \
  --reindexer-lr 5e-5 \
  --weight-decay 1e-4 \
  --grad-clip-norm 1.0 \
  --log-every-steps 10 \
  --save-every-epochs 1 \
  --tau-r 1.0 \
  --tau-m 0.10 \
  --alpha-loc 0.0 \
  --beta-cons 0.0 \
  --trainer-config-json "$RUN_DIR/trainer_config_v131_v141.json" \
  --loss-config-json "$RUN_DIR/loss_config_v141.json" \
  --checkpoint-dir "$RUN_DIR" \
  --checkpoint-prefix fullrun_queko_mqt_curriculum_v141


# updated cmd line:
# A. turn on stage-2 consistency loss → command line only
# B. make stage 2 gentler → command line only

cat > "$RUN_DIR/trainer_config_reidxfix_v141.json" <<'JSON'
{
  "use_staged_curriculum": true,

  "warmup_epochs": 10,
  "stage1_epochs": 25,
  "stage2_epochs": 25,

  "tau_r_start": 1.0,
  "tau_r_mid": 0.70,
  "tau_r_end": 0.25,
  "tau_r_schedule": "cosine",

  "stage1_alpha_loc": 0.02,
  "stage1_beta_cons": 0.0,
  "stage2_alpha_loc": 0.05,
  "stage2_beta_cons": 0.10,

  "warmup_reindex_mode": "identity",

  "mapper_lr": 1e-4,
  "reindexer_lr": 5e-5,
  "weight_decay": 1e-4,

  "grad_clip_norm": 1.0,
  "log_every_steps": 10,
  "save_every_epochs": 1,

  "freeze_hardware_reindex": false,
  "enable_canonical_pretrain": false,
  "pretrain_epochs": 0
}
JSON

cat > "$RUN_DIR/loss_config_reidxfix_v141.json" <<'JSON'
{
  "tau_m": 0.10,
  "sinkhorn_iters": 20,

  "eps": 1e-6,
  "eps_surv": 1e-12,

  "route_step_2q_mult": 3.0,
  "route_step_1q_mult": 2.0,

  "alpha_loc": 0.0,
  "beta_cons": 0.0
}
JSON

python -m kmw.cli.main train \
  --manifest "$TRAIN_MANIFEST" \
  --device cuda \
  --epochs 60 \
  --seed 20260331 \
  --batch-size 1 \
  --num-workers 0 \
  --pin-memory \
  --mapper-lr 1e-4 \
  --reindexer-lr 5e-5 \
  --weight-decay 1e-4 \
  --grad-clip-norm 1.0 \
  --log-every-steps 10 \
  --save-every-epochs 1 \
  --tau-r 1.0 \
  --tau-m 0.10 \
  --alpha-loc 0.0 \
  --beta-cons 0.0 \
  --trainer-config-json "$RUN_DIR/trainer_config_reidxfix_v141.json" \
  --loss-config-json "$RUN_DIR/loss_config_reidxfix_v141.json" \
  --checkpoint-dir "$RUN_DIR" \
  --checkpoint-prefix fullrun_queko_mqt_curriculum_reidxfix1_v141


# -----------------------------------------------------------------------------
# 4) confirm checkpoint exists
# -----------------------------------------------------------------------------

ls -lh "$RUN_DIR" 2>/dev/null || true
test -f "$CKPT" && echo "Checkpoint OK: $CKPT"


# -----------------------------------------------------------------------------
# 5) held-out validation: QASMBench (ROUTED FINAL EVAL, NO ROUTED ARTIFACT SAVE)
# -----------------------------------------------------------------------------
# This still performs real routed downstream evaluation, but it temporarily
# disables routed QASM/QPY export so the validation pass is not blocked by a
# filesystem write failure.
# -----------------------------------------------------------------------------

python -m kmw.cli.main eval \
  --manifest "$QASMBENCH_VAL" \
  --checkpoint "$CKPT" \
  --device cuda \
  --seed 20260331 \
  --batch-size 1 \
  --num-workers 0 \
  --pin-memory \
  --tau-m 0.10 \
  --alpha-loc 0.0 \
  --beta-cons 0.0 \
  --loss-config-json "$RUN_DIR/loss_config_v141.json" \
  --per-circuit-csv "$RUN_DIR/qasmbench_val_per_circuit_nosave.csv" \
  --summary-json "$RUN_DIR/qasmbench_val_summary_nosave.json" \
  --eval-split-name qasmbench_val \
  --route-final-eval \
  --routing-method sabre \
  --transpile-optimization-level 0 \
  --seed-transpiler 20260325 \
  --include-readout-in-pst


# -----------------------------------------------------------------------------
# 6) held-out test: QASMBench (ROUTED FINAL EVAL)
# -----------------------------------------------------------------------------

python -m kmw.cli.main eval \
  --manifest "$QASMBENCH_TEST" \
  --checkpoint "$CKPT" \
  --device cuda \
  --seed 20260331 \
  --batch-size 1 \
  --num-workers 0 \
  --pin-memory \
  --tau-m 0.10 \
  --alpha-loc 0.0 \
  --beta-cons 0.0 \
  --loss-config-json "$RUN_DIR/loss_config_v141.json" \
  --per-circuit-csv "$RUN_DIR/qasmbench_test_per_circuit_nosave.csv" \
  --summary-json "$RUN_DIR/qasmbench_test_summary_nosave.json" \
  --eval-split-name qasmbench_test \
  --route-final-eval \
  --routing-method sabre \
  --transpile-optimization-level 0 \
  --seed-transpiler 20260325 \
  --include-readout-in-pst


# -----------------------------------------------------------------------------
# 7) held-out validation: RevLib (ROUTED FINAL EVAL)
# -----------------------------------------------------------------------------

python -m kmw.cli.main eval \
  --manifest "$REVLIB_VAL" \
  --checkpoint "$CKPT" \
  --device cuda \
  --seed 20260331 \
  --batch-size 1 \
  --num-workers 0 \
  --pin-memory \
  --tau-m 0.10 \
  --alpha-loc 0.0 \
  --beta-cons 0.0 \
  --loss-config-json "$RUN_DIR/loss_config_v141.json" \
  --per-circuit-csv "$RUN_DIR/revlib_val_per_circuit.csv" \
  --summary-json "$RUN_DIR/revlib_val_summary.json" \
  --eval-split-name revlib_val \
  --route-final-eval \
  --routing-method sabre \
  --transpile-optimization-level 0 \
  --seed-transpiler 20260325 \
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
  --seed 20260331 \
  --batch-size 1 \
  --num-workers 0 \
  --pin-memory \
  --tau-m 0.10 \
  --alpha-loc 0.0 \
  --beta-cons 0.0 \
  --loss-config-json "$RUN_DIR/loss_config_v141.json" \
  --per-circuit-csv "$RUN_DIR/revlib_test_per_circuit.csv" \
  --summary-json "$RUN_DIR/revlib_test_summary.json" \
  --eval-split-name revlib_test \
  --route-final-eval \
  --routing-method sabre \
  --transpile-optimization-level 0 \
  --seed-transpiler 20260325 \
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
# - train_metrics.jsonl
# - latest.pt
# - fullrun_queko_mqt_curriculum_v141_epoch_XXXX.pt
# - qasmbench_val_per_circuit_nosave.csv
# - qasmbench_val_summary_nosave.json
# - qasmbench_test_per_circuit_nosave.csv
# - qasmbench_test_summary_nosave.json
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