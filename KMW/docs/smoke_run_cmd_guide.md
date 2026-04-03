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

PYTHONPATH="$PWD/src" python scripts/build_manifests.py \
  --project-root "$PWD" \
  --circuits-root "$PWD/data/circuits/qasm" \
  --sources mqt_smoke \
  --output-dir "$PWD/data/manifests" \
  --seed 20260321 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --min-qubits 2 \
  --max-qubits 27

# =============================================================================
# 2) Smoke training command
# =============================================================================

cd ~/KMWs_workspace/GraphQMap/KMW
PYTHONPATH="$PWD/src" python -m kmw.cli.main train \
  --manifest "$PWD/data/manifests/train.jsonl" \
  --batch-size 1 \
  --num-workers 0 \
  --device cpu \
  --epochs 1 \
  --tau-r 1.0 \
  --tau-m 0.10 \
  --lambda-p 1.0 \
  --lambda-s 1.0 \
  --lambda-d 0.25 \
  --kappa-depth 1.0 \
  --alpha-loc 0.0 \
  --beta-cons 0.0 \
  --checkpoint-dir "$PWD/checkpoints" \
  --checkpoint-prefix kmw_smoke

# =============================================================================
# 3) Smoke evaluation command
# =============================================================================

cd ~/KMWs_workspace/GraphQMap/KMW
export PYTHONPATH="$PWD/src"
python -m kmw.cli.main eval \
  --manifest "$PWD/data/manifests/val.jsonl" \
  --batch-size 1 \
  --num-workers 0 \
  --device cuda \
  --checkpoint "$PWD/checkpoints/kmw_smoke_gpu_epoch_0001.pt" \
  --eval-split-name smoke_val \
  --per-circuit-csv "$PWD/artifacts/eval/smoke_val_metrics.csv" \
  --summary-json "$PWD/artifacts/eval/smoke_val_summary.json"


# =============================================================================
# 4) Smoke Run Summary
# =============================================================================



# What we accomplished with the smoke run

We got the entire KMW smoke pipeline working end-to-end.

Concretely, we:

>fixed the src/kmw package wiring and CLI↔dataset mismatches,
>fixed manifest generation and confirmed it writes authoritative train/val/test.jsonl,
>fixed backend resolution so FakeTorontoV2 uses the new qiskit_ibm_runtime.fake_provider path,
>fixed the dataset metadata bug (slots=True dataclass + __dict__),
>built a separate Pascal-compatible GPU env for TITAN Xp,
>ran smoke training successfully on both CPU and CUDA,
>saved checkpoints,
>ran smoke evaluation successfully and wrote the CSV and summary JSON.

That means the exact staged goal of the smoke phase was achieved: the project design says the smoke test is there to verify preprocessing, finite-value handling, no exploding loss, and that the two-pass training loop works before moving on to the main run.

# What we found out

The smoke run told us several useful things.

First, the pipeline is stable:

>no NaN/Inf blowups,
>both training passes ran,
>checkpoints were saved,
>eval completed with Success=1, InvalidMapping=0, Exceptions=0.

Second, the current architecture and loss wiring are operational:

>the native-frame preprocessing,
>soft reindexer,
>mapper,
>decode-back-to-native loss path,
>and Hungarian-based eval path all worked together, which is exactly the intended design.

Third, the environment path is now known-good("graphqmap_pascal"):

>graphqmap_pascal
>torch 2.0.1+cu118
>numpy 1.26.4
>qiskit 2.3.1
>qiskit_ibm_runtime 0.45.1
>TITAN Xp usable on CUDA.

So the smoke run did its job: it validated infrastructure and numerical stability, not research performance. The plan explicitly treats smoke as a stabilization phase, not the final comparison run.