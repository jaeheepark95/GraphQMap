# kmw2 staged cmd guide (existing-split version)

## What this version assumes
- Do **not** rebuild train/val/test splits.
- `QUEKO(train)` means `data/manifests/full/source_manifests/queko/train.jsonl`.
- `QASMBench(train)` means `data/manifests/full/source_manifests/qasmbench/train.jsonl`.
- Held-out validation/test stay aligned with the old `real_full_run_canonical v1.41.md` protocol:
  - `data/manifests/full/source_manifests/qasmbench/val.jsonl`
  - `data/manifests/full/source_manifests/qasmbench/test.jsonl`
  - `data/manifests/full/source_manifests/revlib/val.jsonl`
  - `data/manifests/full/source_manifests/revlib/test.jsonl`

## Environment
```bash
conda activate graphqmap_pascal
cd ~/KMWs_workspace/GraphQMap/KMW
export PYTHONPATH="$PWD/src"
python -m pip install -e .
python -m pip install qiskit-aer
```

## What to copy into your repo
Copy these from the zip into your repo root:
- `src/kmw2`
- `configs/`
- `scripts/`

Do **not** run `scripts/build_manifests.py` for the real staged experiment unless you intentionally want to create a new split family.

## Staged training semantics
### Stage 1 — Synthetic Warmup
- dataset: existing `queko/train.jsonl`
- sampling: source-balanced (choose source bucket first, then uniformly inside it)
- purpose: warm up on synthetic only

### Stage 2 — Mixed Synthetic + Real
- datasets:
  - existing `queko/train.jsonl`
  - existing `qasmbench/train.jsonl`
- sampling: two-level group-balanced
- group weights:
  - synthetic = 0.7
  - real = 0.3

### Stage 3 — Real-Focused Calibration
- start from Stage-2 best checkpoint
- datasets: same two existing train manifests
- sampling:
  - synthetic = 0.5
  - real = 0.5
- lower LR than Stage 2

## Run staged training
```bash
python -m kmw2.cli.main train-staged --config configs/train_staged.yaml
```

Outputs:
- `runs/kmw2/staged_main/model_final.pth`
- `runs/kmw2/staged_main/train_metrics.json`
- `runs/kmw2/staged_main/checkpoints/best.pt`
- `runs/kmw2/staged_main/checkpoints/last.pt`
- `runs/kmw2/staged_main/checkpoints/stage1_synthetic_warmup_best.pt`
- `runs/kmw2/staged_main/checkpoints/stage2_mixed_synthetic_real_best.pt`
- `runs/kmw2/staged_main/checkpoints/stage3_real_calibration_best.pt`

## Run held-out evals
QASMBench val:
```bash
python -m kmw2.cli.main eval --config configs/eval_qasmbench_val.yaml
```

QASMBench test:
```bash
python -m kmw2.cli.main eval --config configs/eval_qasmbench_test.yaml
```

RevLib val:
```bash
python -m kmw2.cli.main eval --config configs/eval_revlib_val.yaml
```

RevLib test:
```bash
python -m kmw2.cli.main eval --config configs/eval_revlib_test.yaml
```

## One-circuit diagnostic
```bash
python -m kmw2.cli.main eval-one --config configs/eval_qasmbench_test.yaml --circuit path/to/circuit.qasm
```

## Metrics reported
- `pst` (only if you wire a PST callable)
- `compile_seconds`
- `swap_overhead` = added `swap` gate count
- `depth_increase` = `routed_depth - original_depth`

## Important note on PST
`pst` stays `null` unless you set `evaluation.pst_callable` to your actual PST helper. The other three metrics work without that hook.
