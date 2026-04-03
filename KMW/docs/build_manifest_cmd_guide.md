# =============================================================================
# <smoke run cmd guide>
# =============================================================================

// note that all cmds run through cli(Command Line Interface)/main

# =============================================================================
# 0) run everything from:
# =============================================================================

conda activate graphqmap_pascal -> older PyTorch setting (compatible with pascal(titanXP) architecture)
conda activate graphqmap_1 -> latest PyTorch setting (notcompatible with pascal(titanXP) architecture)

cd ~/KMWs_workspace/GraphQMap/KMW
export PYTHONPATH="$PWD/src"


# =============================================================================
# 1) Manifest generation
# =============================================================================

# Test command block

# 1) Full build from circuit_v2 -> train, run
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

# 2) Primitive per-source manifests only
python scripts/build_manifest_full.py \
  --project-root "$PWD" \
  --circuits-root "$PWD/data/circuits_v2/qasm" \
  --output-dir "$PWD/data/manifests/full" \
  --sources queko mlqd mqt_bench qasmbench revlib benchmarks \
  --train-side-sources queko mlqd mqt_bench \
  --heldout-sources qasmbench revlib \
  --benchmark-sources benchmarks \
  --emit-source-manifests-only

# 3) Quick sanity checks
find data/manifests/full -maxdepth 3 -type f | sort
python -m json.tool data/manifests/full/catalog.json | head -n 80
wc -l data/manifests/full/source_manifests/mqt_bench/*.jsonl
wc -l data/manifests/full/recipes/smoke_mqt/*.jsonl






# =============================================================================
# 2) quick sanity checks after manifest build (exact, runnable now)
# =============================================================================

python -m json.tool data/manifests/full/catalog.json | head -n 80

find data/manifests/full/source_manifests -maxdepth 2 -type f | sort
find data/manifests/full/recipes -maxdepth 2 -type f | sort

wc -l data/manifests/full/recipes/smoke_mqt/*.jsonl
wc -l data/manifests/full/recipes/train_queko_mqt_bench/*.jsonl
wc -l data/manifests/full/recipes/heldout_qasmbench/*.jsonl
wc -l data/manifests/full/recipes/heldout_revlib/*.jsonl


# =============================================================================
# 3) define manifest variables for experiment switching
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
# 4) smoke experiment (template: use after manifest-aware train CLI is wired)
#    goal: verify preprocessing, numerics, and no exploding loss
# =============================================================================

python -m kmw.cli.main train \
  --config configs/smoke_mqt.yaml \
  --train-manifest "$SMOKE_TRAIN" \
  --val-manifest "$SMOKE_VAL" \
  --run-name smoke_mqt_seed20260323

# smoke eval on smoke split
python -m kmw.cli.main eval \
  --config configs/eval.yaml \
  --checkpoint artifacts/smoke_mqt_seed20260323/best.pt \
  --test-manifest "$SMOKE_TEST" \
  --run-name smoke_mqt_eval_seed20260323


# =============================================================================
# 5) main comparison training (recommended practical first version)
#    use QUEKO + MQT first
# =============================================================================

python -m kmw.cli.main train \
  --config configs/train_main.yaml \
  --train-manifest "$MAIN_TRAIN" \
  --val-manifest "$MAIN_VAL" \
  --run-name train_queko_mqt_bench_seed20260323

# evaluate that checkpoint on held-out QASMBench
python -m kmw.cli.main eval \
  --config configs/eval.yaml \
  --checkpoint artifacts/train_queko_mqt_bench_seed20260323/best.pt \
  --test-manifest "$QASMBENCH_TEST" \
  --run-name eval_qasmbench_from_queko_mqt

# evaluate that checkpoint on held-out RevLib
python -m kmw.cli.main eval \
  --config configs/eval.yaml \
  --checkpoint artifacts/train_queko_mqt_bench_seed20260323/best.pt \
  --test-manifest "$REVLIB_TEST" \
  --run-name eval_revlib_from_queko_mqt


# =============================================================================
# 6) full train-side comparison run
#    use only after you are satisfied with MLQD inclusion / filtering behavior
# =============================================================================

python -m kmw.cli.main train \
  --config configs/train_main.yaml \
  --train-manifest "$FULL_TRAIN" \
  --val-manifest "$FULL_VAL" \
  --run-name train_queko_mlqd_mqt_bench_seed20260323

python -m kmw.cli.main eval \
  --config configs/eval.yaml \
  --checkpoint artifacts/train_queko_mlqd_mqt_bench_seed20260323/best.pt \
  --test-manifest "$QASMBENCH_TEST" \
  --run-name eval_qasmbench_from_fulltrain

python -m kmw.cli.main eval \
  --config configs/eval.yaml \
  --checkpoint artifacts/train_queko_mlqd_mqt_bench_seed20260323/best.pt \
  --test-manifest "$REVLIB_TEST" \
  --run-name eval_revlib_from_fulltrain


# =============================================================================
# 7) ablation examples: switch dataset by changing only the manifest
# =============================================================================

# QUEKO only
python -m kmw.cli.main train \
  --config configs/train_main.yaml \
  --train-manifest data/manifests/full/recipes/train_queko/train.jsonl \
  --val-manifest data/manifests/full/recipes/train_queko/val.jsonl \
  --run-name ablation_train_queko_only

# MQT only (larger than smoke, but same dataset family)
python -m kmw.cli.main train \
  --config configs/train_main.yaml \
  --train-manifest data/manifests/full/recipes/train_mqt_bench/train.jsonl \
  --val-manifest data/manifests/full/recipes/train_mqt_bench/val.jsonl \
  --run-name ablation_train_mqt_only

# QUEKO + MLQD
python -m kmw.cli.main train \
  --config configs/train_main.yaml \
  --train-manifest data/manifests/full/recipes/train_queko_mlqd/train.jsonl \
  --val-manifest data/manifests/full/recipes/train_queko_mlqd/val.jsonl \
  --run-name ablation_train_queko_mlqd

# MLQD + MQT
python -m kmw.cli.main train \
  --config configs/train_main.yaml \
  --train-manifest data/manifests/full/recipes/train_mlqd_mqt_bench/train.jsonl \
  --val-manifest data/manifests/full/recipes/train_mlqd_mqt_bench/val.jsonl \
  --run-name ablation_train_mlqd_mqt


# =============================================================================
# 8) direct primitive-source experiments
#    use these if you want maximum manual control instead of recipe manifests
# =============================================================================

python -m kmw.cli.main train \
  --config configs/train_main.yaml \
  --train-manifest data/manifests/full/source_manifests/queko/train.jsonl \
  --val-manifest data/manifests/full/source_manifests/queko/val.jsonl \
  --run-name source_only_queko

python -m kmw.cli.main eval \
  --config configs/eval.yaml \
  --checkpoint artifacts/source_only_queko/best.pt \
  --test-manifest data/manifests/full/source_manifests/revlib/test.jsonl \
  --run-name source_only_queko_to_revlib


# =============================================================================
# 9) optional benchmark-set evaluation
# =============================================================================

python -m kmw.cli.main eval \
  --config configs/eval.yaml \
  --checkpoint artifacts/train_queko_mqt_bench_seed20260323/best.pt \
  --test-manifest "$BENCHMARK_TEST" \
  --run-name eval_optional_benchmarks


# =============================================================================
# 10) if your current main.py uses different flag names
# =============================================================================
# keep the manifest paths and experiment ordering exactly the same;
# only substitute the CLI flag names that your actual implementation expects.

