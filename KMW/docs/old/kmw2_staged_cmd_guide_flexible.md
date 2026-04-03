# kmw2 staged cmd guide (flexible CLI, existing-splits version)

## What this version assumes
- Use the **existing** source manifests already in the repo.
- Do **not** rebuild train/val/test splits for the real run.
- `QUEKO(train)` means `data/manifests/full/source_manifests/queko/train.jsonl`.
- `QASMBench(train)` means `data/manifests/full/source_manifests/qasmbench/train.jsonl`.
- Held-out eval stays aligned with the old `real_full_run_canonical v1.41.md` protocol:
  - `qasmbench/val.jsonl`
  - `qasmbench/test.jsonl`
  - `revlib/val.jsonl`
  - `revlib/test.jsonl`

## Staged training semantics

### Stage 1 — Synthetic Warmup
- dataset: `queko/train.jsonl`
- sampler: `source_balanced`
- intent: warm start on synthetic circuits only

### Stage 2 — Mixed Synthetic + Real Training
- dataset union:
  - `queko/train.jsonl`
  - `qasmbench/train.jsonl`
- sampler: `group_balanced`
- group weights:
  - synthetic = `0.7`
  - real = `0.3`

### Stage 3 — Real-Focused Calibration
- start from **Stage-2 best** checkpoint
- same dataset union as Stage 2
- sampler: `group_balanced`
- group weights:
  - synthetic = `0.5`
  - real = `0.5`
- lower LR than Stage 2

---

## Environment
```bash
conda activate graphqmap_pascal
cd ~/KMWs_workspace/GraphQMap/KMW
export PYTHONPATH="$PWD/src"
python -m pip install -e .
python -m pip install qiskit-aer
```

---

## Minimal staged training run
```bash
python -m kmw2.cli.main train-staged \
  --config configs/base.yaml \
  --config configs/train_staged.yaml
```

This writes by default to:
```text
runs/kmw2/staged_main/
```

Important outputs:
```text
runs/kmw2/staged_main/run_config.json
runs/kmw2/staged_main/logs/train_metrics.json
runs/kmw2/staged_main/logs/train_metrics.jsonl
runs/kmw2/staged_main/logs/epoch_metrics.jsonl
runs/kmw2/staged_main/checkpoints/best.pt
runs/kmw2/staged_main/checkpoints/last.pt
runs/kmw2/staged_main/model_final.pth
runs/kmw2/staged_main/stages/<stage_name>/metrics.json
```

---

## Flexible staged training run (recommended)
This restores the old run-sheet style flexibility: custom run dir, trainer/loss/stage configs, epoch overrides, batch size, LR, backend, device, and routed output dirs.

```bash
python -m kmw2.cli.main train-staged \
  --config configs/base.yaml \
  --config configs/train_staged.yaml \
  --trainer-config configs/overrides/trainer_default.yaml \
  --loss-config configs/overrides/loss_default.yaml \
  --stage-config configs/overrides/stages_existing_splits.yaml \
  --run-dir runs/kmw2/real_full_run_staged \
  --checkpoint-name model_final.pth \
  --epochs 60 \
  --batch-size 16 \
  --num-workers 0 \
  --backend-name FakeTorontoV2 \
  --device cuda
```

### Notes on overrides
- `--run-dir` changes the whole training output root.
- `--checkpoint-name` changes the final model filename under the run dir.
- `--epochs`, `--batch-size`, `--num-workers`, `--lr`, `--weight-decay` override the top-level training config.
- For **staged training**, `--epochs`, `--batch-size`, `--num-workers`, `--lr`, `--weight-decay` also override each stage entry.
- `--trainer-config`, `--loss-config`, `--stage-config`, `--eval-config` are merged on top of the base configs.

---

## Evaluate held-out QASMBench val
```bash
python -m kmw2.cli.main eval \
  --config configs/base.yaml \
  --config configs/eval_qasmbench_val.yaml \
  --run-dir runs/kmw2/real_full_run_staged \
  --checkpoint-name model_final.pth \
  --per-circuit-csv runs/kmw2/eval/qasmbench_val/per_circuit.csv \
  --per-circuit-json runs/kmw2/eval/qasmbench_val/per_circuit.json \
  --summary-json runs/kmw2/eval/qasmbench_val/summary.json \
  --save-routed-qasm-dir runs/kmw2/eval/qasmbench_val/routed_qasm \
  --save-routed-qpy-dir runs/kmw2/eval/qasmbench_val/routed_qpy \
  --backend-name FakeTorontoV2 \
  --device cuda
```

## Evaluate held-out QASMBench test
```bash
python -m kmw2.cli.main eval \
  --config configs/base.yaml \
  --config configs/eval_qasmbench_test.yaml \
  --run-dir runs/kmw2/real_full_run_staged \
  --checkpoint-name model_final.pth \
  --per-circuit-csv runs/kmw2/eval/qasmbench_test/per_circuit.csv \
  --per-circuit-json runs/kmw2/eval/qasmbench_test/per_circuit.json \
  --summary-json runs/kmw2/eval/qasmbench_test/summary.json \
  --save-routed-qasm-dir runs/kmw2/eval/qasmbench_test/routed_qasm \
  --save-routed-qpy-dir runs/kmw2/eval/qasmbench_test/routed_qpy \
  --backend-name FakeTorontoV2 \
  --device cuda
```

## Evaluate held-out RevLib val
```bash
python -m kmw2.cli.main eval \
  --config configs/base.yaml \
  --config configs/eval_revlib_val.yaml \
  --run-dir runs/kmw2/real_full_run_staged \
  --checkpoint-name model_final.pth \
  --per-circuit-csv runs/kmw2/eval/revlib_val/per_circuit.csv \
  --per-circuit-json runs/kmw2/eval/revlib_val/per_circuit.json \
  --summary-json runs/kmw2/eval/revlib_val/summary.json \
  --save-routed-qasm-dir runs/kmw2/eval/revlib_val/routed_qasm \
  --save-routed-qpy-dir runs/kmw2/eval/revlib_val/routed_qpy \
  --backend-name FakeTorontoV2 \
  --device cuda
```

## Evaluate held-out RevLib test
```bash
python -m kmw2.cli.main eval \
  --config configs/base.yaml \
  --config configs/eval_revlib_test.yaml \
  --run-dir runs/kmw2/real_full_run_staged \
  --checkpoint-name model_final.pth \
  --per-circuit-csv runs/kmw2/eval/revlib_test/per_circuit.csv \
  --per-circuit-json runs/kmw2/eval/revlib_test/per_circuit.json \
  --summary-json runs/kmw2/eval/revlib_test/summary.json \
  --save-routed-qasm-dir runs/kmw2/eval/revlib_test/routed_qasm \
  --save-routed-qpy-dir runs/kmw2/eval/revlib_test/routed_qpy \
  --backend-name FakeTorontoV2 \
  --device cuda
```

---

## Single-circuit eval
```bash
python -m kmw2.cli.main eval-one \
  --config configs/base.yaml \
  --config configs/eval_qasmbench_test.yaml \
  --run-dir runs/kmw2/real_full_run_staged \
  --checkpoint-name model_final.pth \
  --circuit path/to/your_circuit.qasm \
  --save-routed-qasm-dir runs/kmw2/eval/single/routed_qasm \
  --save-routed-qpy-dir runs/kmw2/eval/single/routed_qpy \
  --backend-name FakeTorontoV2 \
  --device cuda
```

---

## Supported flexible CLI flags
### Shared train / train-staged / eval / eval-one
- `--run-dir`
- `--checkpoint`
- `--checkpoint-name`
- `--trainer-config`
- `--loss-config`
- `--stage-config`
- `--eval-config`
- `--epochs`
- `--batch-size`
- `--num-workers`
- `--lr`
- `--weight-decay`
- `--backend-name`
- `--device`
- `--save-routed-qasm-dir`
- `--save-routed-qpy-dir`

### Eval / eval-one only
- `--per-circuit-csv`
- `--per-circuit-json`
- `--summary-json`

---

## Output metrics
Per-circuit eval currently reports:
- `pst` *(null unless a PST callable is wired in)*
- `compile_seconds`
- `swap_overhead` = number of added `swap` gates
- `depth_increase` = routed depth - original depth
- `original_depth`
- `routed_depth`
- `original_gate_count`
- `routed_gate_count`

---

## Important warning
Do **not** run `scripts/build_manifests.py` for the real staged experiment if your existing `source_manifests/.../*.jsonl` are already the authoritative split. This staged workflow is intentionally built to consume those **existing** manifests directly.
