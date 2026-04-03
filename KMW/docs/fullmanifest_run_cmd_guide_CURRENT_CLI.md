# =============================================================================
# fullmanifest_run_cmd_guide_CURRENT_CLI.md
# =============================================================================
#
# This is the corrected command guide for the CURRENT kmw.cli.main interface.
#
# Important:
# - train/eval both use a single --manifest
# - there is NO --config
# - there is NO --run-name
# - eval uses --checkpoint plus output-path flags
# - checkpoints are written under --checkpoint-dir, with latest.pt saved each epoch
#
# =============================================================================
# 0) run everything from:
# =============================================================================

conda activate graphqmap_pascal
cd ~/KMWs_workspace/GraphQMap/KMW
export PYTHONPATH="$PWD/src"


# =============================================================================
# 1) Manifest generation
# =============================================================================

# 1A) Full build from circuit_v2
python scripts/build_manifest_full.py \
  --project-root "$PWD" \
  --circuits-root "$PWD/data/circuits_v2/qasm" \
  --output-dir "$PWD/data/manifests/full" \
  --sources queko mlqd mqt_bench qasmbench revlib benchmarks \
  --train-side-sources queko mlqd mqt_bench \
  --heldout-sources qasmbench revlib \
  --benchmark-sources benchmarks \
  --seed 20260323 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --min-qubits 2 \
  --max-qubits 27

# 1B) Primitive per-source manifests only
python scripts/build_manifest_full.py \
  --project-root "$PWD" \
  --circuits-root "$PWD/data/circuits_v2/qasm" \
  --output-dir "$PWD/data/manifests/full" \
  --sources queko mlqd mqt_bench qasmbench revlib benchmarks \
  --emit-source-manifests-only


# =============================================================================
# 2) Quick sanity checks after manifest build
# =============================================================================

python -m json.tool data/manifests/full/catalog.json | head -n 80

find data/manifests/full/source_manifests -maxdepth 2 -type f | sort
find data/manifests/full/recipes -maxdepth 2 -type f | sort

wc -l data/manifests/full/recipes/smoke_mqt/*.jsonl
wc -l data/manifests/full/recipes/train_queko_mqt_bench/*.jsonl
wc -l data/manifests/full/recipes/heldout_qasmbench/*.jsonl
wc -l data/manifests/full/recipes/heldout_revlib/*.jsonl


# =============================================================================
# 3) Define manifest variables for experiment switching
# =============================================================================

# Smoke
SMOKE_TRAIN=data/manifests/full/recipes/smoke_mqt/train.jsonl
SMOKE_VAL=data/manifests/full/recipes/smoke_mqt/val.jsonl
SMOKE_TEST=data/manifests/full/recipes/smoke_mqt/test.jsonl

# Main-train candidate (recommended practical start)
MAIN_TRAIN=data/manifests/full/recipes/train_queko_mqt_bench/train.jsonl
MAIN_VAL=data/manifests/full/recipes/train_queko_mqt_bench/val.jsonl
MAIN_TEST=data/manifests/full/recipes/train_queko_mqt_bench/test.jsonl

# Full train-side recipe
FULL_TRAIN=data/manifests/full/recipes/train_queko_mlqd_mqt_bench/train.jsonl
FULL_VAL=data/manifests/full/recipes/train_queko_mlqd_mqt_bench/val.jsonl
FULL_TEST=data/manifests/full/recipes/train_queko_mlqd_mqt_bench/test.jsonl

# Held-out evaluation
QASMBENCH_VAL=data/manifests/full/recipes/heldout_qasmbench/val.jsonl
QASMBENCH_TEST=data/manifests/full/recipes/heldout_qasmbench/test.jsonl

REVLIB_VAL=data/manifests/full/recipes/heldout_revlib/val.jsonl
REVLIB_TEST=data/manifests/full/recipes/heldout_revlib/test.jsonl

# Optional benchmark challenge set
BENCHMARK_TEST=data/manifests/full/recipes/benchmark_benchmarks/test.jsonl


# =============================================================================
# 4) Smoke run (first thing to do)
# =============================================================================
#
# Current CLI has no separate val-manifest argument for train.
# So the pattern is:
#   1) train on smoke_mqt/train.jsonl
#   2) eval on smoke_mqt/val.jsonl
#   3) eval on smoke_mqt/test.jsonl
#

# 4A) Smoke train
python -m kmw.cli.main train \
  --manifest "$SMOKE_TRAIN" \
  --epochs 1 \
  --seed 20260323 \
  --checkpoint-dir artifacts/smoke_mqt_seed20260323 \
  --checkpoint-prefix smoke_mqt

# 4B) Confirm checkpoint files exist
ls -R artifacts/smoke_mqt_seed20260323

# 4C) Smoke validation eval
python -m kmw.cli.main eval \
  --manifest "$SMOKE_VAL" \
  --checkpoint artifacts/smoke_mqt_seed20260323/latest.pt \
  --seed 20260323 \
  --per-circuit-csv artifacts/smoke_mqt_seed20260323/val_per_circuit_metrics.csv \
  --summary-json artifacts/smoke_mqt_seed20260323/val_summary.json \
  --eval-split-name smoke_mqt_val

# 4D) Smoke test eval
python -m kmw.cli.main eval \
  --manifest "$SMOKE_TEST" \
  --checkpoint artifacts/smoke_mqt_seed20260323/latest.pt \
  --seed 20260323 \
  --per-circuit-csv artifacts/smoke_mqt_seed20260323/test_per_circuit_metrics.csv \
  --summary-json artifacts/smoke_mqt_seed20260323/test_summary.json \
  --eval-split-name smoke_mqt_test


# =============================================================================
# 5) Main comparison training (recommended practical first version)
#    start with QUEKO + MQT
# =============================================================================

# 5A) Main train
python -m kmw.cli.main train \
  --manifest "$MAIN_TRAIN" \
  --epochs 1 \
  --seed 20260323 \
  --checkpoint-dir artifacts/train_queko_mqt_bench_seed20260323 \
  --checkpoint-prefix train_queko_mqt_bench

# 5B) Optional validation on the recipe val split
python -m kmw.cli.main eval \
  --manifest "$MAIN_VAL" \
  --checkpoint artifacts/train_queko_mqt_bench_seed20260323/latest.pt \
  --seed 20260323 \
  --per-circuit-csv artifacts/train_queko_mqt_bench_seed20260323/main_val_per_circuit.csv \
  --summary-json artifacts/train_queko_mqt_bench_seed20260323/main_val_summary.json \
  --eval-split-name train_queko_mqt_bench_val

# 5C) In-domain recipe test split
python -m kmw.cli.main eval \
  --manifest "$MAIN_TEST" \
  --checkpoint artifacts/train_queko_mqt_bench_seed20260323/latest.pt \
  --seed 20260323 \
  --per-circuit-csv artifacts/train_queko_mqt_bench_seed20260323/main_test_per_circuit.csv \
  --summary-json artifacts/train_queko_mqt_bench_seed20260323/main_test_summary.json \
  --eval-split-name train_queko_mqt_bench_test

# 5D) Held-out QASMBench eval
python -m kmw.cli.main eval \
  --manifest "$QASMBENCH_TEST" \
  --checkpoint artifacts/train_queko_mqt_bench_seed20260323/latest.pt \
  --seed 20260323 \
  --per-circuit-csv artifacts/train_queko_mqt_bench_seed20260323/qasmbench_per_circuit.csv \
  --summary-json artifacts/train_queko_mqt_bench_seed20260323/qasmbench_summary.json \
  --eval-split-name heldout_qasmbench

# 5E) Held-out RevLib eval
python -m kmw.cli.main eval \
  --manifest "$REVLIB_TEST" \
  --checkpoint artifacts/train_queko_mqt_bench_seed20260323/latest.pt \
  --seed 20260323 \
  --per-circuit-csv artifacts/train_queko_mqt_bench_seed20260323/revlib_per_circuit.csv \
  --summary-json artifacts/train_queko_mqt_bench_seed20260323/revlib_summary.json \
  --eval-split-name heldout_revlib


# =============================================================================
# 6) Full train-side comparison run
#    use only after you are satisfied with MLQD inclusion / filtering behavior
# =============================================================================

# 6A) Full train-side train
python -m kmw.cli.main train \
  --manifest "$FULL_TRAIN" \
  --epochs 1 \
  --seed 20260323 \
  --checkpoint-dir artifacts/train_queko_mlqd_mqt_bench_seed20260323 \
  --checkpoint-prefix train_queko_mlqd_mqt_bench

# 6B) Optional validation on full train-side val split
python -m kmw.cli.main eval \
  --manifest "$FULL_VAL" \
  --checkpoint artifacts/train_queko_mlqd_mqt_bench_seed20260323/latest.pt \
  --seed 20260323 \
  --per-circuit-csv artifacts/train_queko_mlqd_mqt_bench_seed20260323/full_val_per_circuit.csv \
  --summary-json artifacts/train_queko_mlqd_mqt_bench_seed20260323/full_val_summary.json \
  --eval-split-name train_queko_mlqd_mqt_bench_val

# 6C) Held-out QASMBench eval
python -m kmw.cli.main eval \
  --manifest "$QASMBENCH_TEST" \
  --checkpoint artifacts/train_queko_mlqd_mqt_bench_seed20260323/latest.pt \
  --seed 20260323 \
  --per-circuit-csv artifacts/train_queko_mlqd_mqt_bench_seed20260323/qasmbench_per_circuit.csv \
  --summary-json artifacts/train_queko_mlqd_mqt_bench_seed20260323/qasmbench_summary.json \
  --eval-split-name heldout_qasmbench

# 6D) Held-out RevLib eval
python -m kmw.cli.main eval \
  --manifest "$REVLIB_TEST" \
  --checkpoint artifacts/train_queko_mlqd_mqt_bench_seed20260323/latest.pt \
  --seed 20260323 \
  --per-circuit-csv artifacts/train_queko_mlqd_mqt_bench_seed20260323/revlib_per_circuit.csv \
  --summary-json artifacts/train_queko_mlqd_mqt_bench_seed20260323/revlib_summary.json \
  --eval-split-name heldout_revlib


# =============================================================================
# 7) Ablation examples: switch dataset by changing only the manifest
# =============================================================================

# 7A) QUEKO only
python -m kmw.cli.main train \
  --manifest data/manifests/full/recipes/train_queko/train.jsonl \
  --epochs 1 \
  --seed 20260323 \
  --checkpoint-dir artifacts/ablation_train_queko_only \
  --checkpoint-prefix train_queko

# 7B) MQT only
python -m kmw.cli.main train \
  --manifest data/manifests/full/recipes/train_mqt_bench/train.jsonl \
  --epochs 1 \
  --seed 20260323 \
  --checkpoint-dir artifacts/ablation_train_mqt_only \
  --checkpoint-prefix train_mqt_bench

# 7C) QUEKO + MLQD
python -m kmw.cli.main train \
  --manifest data/manifests/full/recipes/train_queko_mlqd/train.jsonl \
  --epochs 1 \
  --seed 20260323 \
  --checkpoint-dir artifacts/ablation_train_queko_mlqd \
  --checkpoint-prefix train_queko_mlqd

# 7D) MLQD + MQT
python -m kmw.cli.main train \
  --manifest data/manifests/full/recipes/train_mlqd_mqt_bench/train.jsonl \
  --epochs 1 \
  --seed 20260323 \
  --checkpoint-dir artifacts/ablation_train_mlqd_mqt \
  --checkpoint-prefix train_mlqd_mqt_bench


# =============================================================================
# 8) Direct primitive-source experiments
#    use these if you want one source only without recipe manifests
# =============================================================================

# 8A) Source-only QUEKO training
python -m kmw.cli.main train \
  --manifest data/manifests/full/source_manifests/queko/train.jsonl \
  --epochs 1 \
  --seed 20260323 \
  --checkpoint-dir artifacts/source_only_queko \
  --checkpoint-prefix source_only_queko

# 8B) Evaluate that source-only model on primitive RevLib test split
python -m kmw.cli.main eval \
  --manifest data/manifests/full/source_manifests/revlib/test.jsonl \
  --checkpoint artifacts/source_only_queko/latest.pt \
  --seed 20260323 \
  --per-circuit-csv artifacts/source_only_queko/revlib_per_circuit.csv \
  --summary-json artifacts/source_only_queko/revlib_summary.json \
  --eval-split-name source_only_queko_to_revlib


# =============================================================================
# 9) Optional benchmark-set evaluation
# =============================================================================

python -m kmw.cli.main eval \
  --manifest "$BENCHMARK_TEST" \
  --checkpoint artifacts/train_queko_mqt_bench_seed20260323/latest.pt \
  --seed 20260323 \
  --per-circuit-csv artifacts/train_queko_mqt_bench_seed20260323/benchmarks_per_circuit.csv \
  --summary-json artifacts/train_queko_mqt_bench_seed20260323/benchmarks_summary.json \
  --eval-split-name benchmark_benchmarks


# =============================================================================
# 10) Minimal recommended sequence
# =============================================================================
#
# 1. build manifests
# 2. run section 4A -> 4D
# 3. if smoke looks stable, run section 5A -> 5E
# 4. only then consider section 6
#
# Practical recommendation:
# - use train_queko_mqt_bench first
# - do NOT rely on the full MLQD-inclusive recipe until you review why MLQD
#   inclusion is currently so low in your manifest build
#
# Optional:
# - use the ablation commands in section 7 to compare source composition effects


