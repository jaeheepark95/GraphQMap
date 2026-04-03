# Design Plan v1.1-derived - Current `kmw2` Implementation Specification

## 0. Document status

This document updates the earlier **v1.1 explicit implementation specification** to match the **current `kmw2` project plan and package structure** now being used in your repository.

It is **not** a pure historical v1.1 reproduction document anymore.

Instead, it records the current state as:

- **core modeling semantics inherited from v1.1**
- **repackaged under `src/kmw2`**
- **staged training added**
- **existing source-manifest splits preserved**
- **flexible CLI/config workflow restored**
- **real PST evaluation added through noisy-vs-ideal distribution overlap**

### 0.1 What remains faithful to v1.1

The current plan still preserves the v1.1 core where intended:

- fixed `n = 27`
- `BackendV2`-based hardware extraction
- canonical hardware reindexing
- logical interaction matrix `W`
- active-mask `m`
- 5-channel `(27 x 27)` U-Net grid input
- logical / physical token conditioning
- Sinkhorn during training
- Hungarian during inference
- proxy loss structure based on PST-like cost, routing pressure, and depth proxy

### 0.2 What is intentionally newer than v1.1

The current `kmw2` plan adds packaging and workflow layers that were **not** part of the original standalone v1.1 scripts:

- `src/kmw2` package layout
- unified CLI under `kmw2.cli.main`
- YAML-driven configuration
- staged training
- direct consumption of **existing** `source_manifests/.../*.jsonl`
- flexible run/output path control
- per-split held-out evaluation configs
- in-package PST metric implementation under `src/kmw2/metrics/pst.py`

---

## 1. Project objective

### 1.1 Core task

Given:

- a quantum circuit with `K <= 27` logical qubits
- a fixed IBM-style `BackendV2` hardware target with `n = 27` physical qubits

predict an **initial one-to-one mapping** from logical qubits to physical qubits.

The model remains **label-free**: it is not trained from ground-truth mappings, but from proxy objectives intended to correlate with:

- execution quality / PST-like behavior
- routing burden
- depth growth after transpilation

### 1.2 Output contract

For each circuit, the current system still produces:

1. a score matrix `S in R^(27 x 27)`
2. a soft assignment `P` during training via Sinkhorn
3. a hard assignment `M` during inference via Hungarian matching
4. a final mapping from logical qubits to **native backend qubit IDs** via canonical-to-native remapping

---

## 2. Current repository structure

The active code is now organized under `KMW/src/kmw2` rather than the original mixed research-script layout.

```text
KMW/
|- scripts/
|- src/
|  |- kmw/
|  |- kmw1/
|  |- kmw1_backup/
|  |- kmw2/
|  |  |- __init__.py
|  |  |- utils.py
|  |  |
|  |  |- cli/
|  |  |  |- __init__.py
|  |  |  |- main.py
|  |  |
|  |  |- data/
|  |  |  |- __init__.py
|  |  |  |- dataset.py
|  |  |
|  |  |- evaluation/
|  |  |  |- __init__.py
|  |  |  |- evaluate.py
|  |  |
|  |  |- losses/
|  |  |  |- __init__.py
|  |  |  |- loss.py
|  |  |
|  |  |- metrics/
|  |  |  |- __init__.py
|  |  |  |- pst.py
|  |  |
|  |  |- models/
|  |  |  |- __init__.py
|  |  |  |- layers.py
|  |  |  |- model.py
|  |  |
|  |  |- preprocessing/
|  |  |  |- __init__.py
|  |  |  |- extractor.py
|  |  |  |- canonical_indexer.py
|  |  |  |- featurizer.py
|  |  |  |- pipeline.py
|  |  |
|  |  |- training/
|  |  |  |- __init__.py
|  |  |  |- samplers.py
|  |  |  |- trainer.py
|  |
|  |- kmw2_old/
|
|- data/
|  |- manifests/
|  |  |- full/
|  |  |  |- source_manifests/
|  |  |  |  |- qasmbench/
|  |  |  |  |  |- all.jsonl
|  |  |  |  |  |- train.jsonl
|  |  |  |  |  |- val.jsonl
|  |  |  |  |  |- test.jsonl
|  |  |  |  |
|  |  |  |  |- queko/
|  |  |  |  |  |- all.jsonl
|  |  |  |  |  |- train.jsonl
|  |  |  |  |  |- val.jsonl
|  |  |  |  |  |- test.jsonl
|  |  |  |  |
|  |  |  |  |- revlib/
|  |  |  |  |- mqt_bench/
|  |  |  |  |- mlqd/
|  |  |  |  |- catalog.json
|  |  |  |  |- train.jsonl
|  |  |  |  |- val.jsonl
|  |  |  |  |- test.jsonl
|  |
|  |- circuits/
|  |- circuits_v2/
```

### 2.1 Key structural migration from the old v1.1-style layout

| Original v1.1-style area | Current `kmw2` area |
|---|---|
| backend extraction / canonicalization / featurization | `src/kmw2/preprocessing/` |
| model layers and mapper | `src/kmw2/models/` |
| dataset logic | `src/kmw2/data/dataset.py` |
| loss | `src/kmw2/losses/loss.py` |
| training loop | `src/kmw2/training/trainer.py` |
| staged samplers | `src/kmw2/training/samplers.py` |
| evaluation loop | `src/kmw2/evaluation/evaluate.py` |
| PST metric helper | `src/kmw2/metrics/pst.py` |
| CLI entrypoint | `src/kmw2/cli/main.py` |

---

## 3. Current data-split contract

### 3.1 Authoritative split rule

The current project must use the **pre-existing** source-specific split files under:

```text
data/manifests/full/source_manifests/<source>/{train,val,test}.jsonl
```

These are the authoritative definitions of `QUEKO(train)`, `QASMBench(train)`, `QASMBench(val)`, etc.

### 3.2 Important non-negotiable rule

The current `kmw2` workflow must **not** redefine or rebuild the train/val/test split for the real run.

So:

- `QUEKO(train)` means `data/manifests/full/source_manifests/queko/train.jsonl`
- `QASMBench(train)` means `data/manifests/full/source_manifests/qasmbench/train.jsonl`
- `QASMBench(val)` means `data/manifests/full/source_manifests/qasmbench/val.jsonl`
- `QASMBench(test)` means `data/manifests/full/source_manifests/qasmbench/test.jsonl`
- `RevLib(val)` means `data/manifests/full/source_manifests/revlib/val.jsonl`
- `RevLib(test)` means `data/manifests/full/source_manifests/revlib/test.jsonl`

This is a deliberate departure from any temporary scaffold behavior that rebuilt manifests from raw QASM folders.

---

## 4. End-to-end pipeline

The current `kmw2` pipeline is:

```text
existing manifest(s)
  -> dataset loader
  -> load QASM circuit(s)
  -> hardware extraction from BackendV2
  -> canonical hardware reindexing
  -> circuit featurization (W, m)
  -> build 5-channel grid + logical/physical tokens
  -> UNetMapping forward pass
  -> Sinkhorn (training) or Hungarian (inference)
  -> canonical-to-native mapping
  -> transpile with initial_layout
  -> compute eval metrics
```

### 4.1 Packaging-level difference from original v1.1

The original v1.1 scripts were direct script-entry flows.

The current implementation is routed through:

```text
python -m kmw2.cli.main <subcommand> ...
```

with config merging and CLI overrides.

---

## 5. Current modeling semantics retained from v1.1

The following remain unchanged in intent:

### 5.1 Hardware preprocessing

`src/kmw2/preprocessing/extractor.py`
- extracts backend-native hardware tensors
- derives adjacency / 1Q badness / 2Q badness from the backend target

`src/kmw2/preprocessing/canonical_indexer.py`
- computes a deterministic canonical qubit order
- all model-facing tensors are in canonical hardware order

### 5.2 Circuit featurization

`src/kmw2/preprocessing/featurizer.py`
- computes symmetric logical interaction matrix `W`
- computes active-mask `m`
- rejects circuits with `K > 27`

### 5.3 Model input family

The current model still uses:

- `X in R^(B x 5 x 27 x 27)`
- logical raw tokens
- physical raw tokens

and still produces:

- `S in R^(B x 27 x 27)`

### 5.4 Assignment behavior

- training: Sinkhorn
- inference/evaluation: Hungarian matching

### 5.5 Loss family

The current `kmw2` loss module still follows the same broad label-free logic:

- PST-like proxy term
- routing / SWAP-pressure proxy
- depth proxy

The exact packaging location changed to:

```text
src/kmw2/losses/loss.py
```

---

## 6. Current staged-training plan

The current `kmw2` plan adds a three-stage curriculum while keeping the old held-out evaluation meaning.

## 6.1 Stage 1 - Synthetic Warmup

Train only on:

- `QUEKO(train)`

Manifest used:

```text
data/manifests/full/source_manifests/queko/train.jsonl
```

Sampling:

- source-balanced sampling
- in practice, because Stage 1 uses only QUEKO, this reduces to sampling from that single source bucket

Purpose:

- learn stable placement behavior from synthetic circuits
- avoid early overfitting to small real datasets
- stabilize model behavior before mixed training

Recommended duration:

- about 20-30% of total training steps

## 6.2 Stage 2 - Mixed Synthetic + Real Training

Train on:

- `QUEKO(train)`
- `QASMBench(train)`

Manifests used:

```text
data/manifests/full/source_manifests/queko/train.jsonl
data/manifests/full/source_manifests/qasmbench/train.jsonl
```

Sampling:

- two-level group-balanced sampling

Group weights:

- synthetic group = `0.7`
- real group = `0.3`

Group composition:

- synthetic -> `queko`
- real -> `qasmbench`

Purpose:

- inject real-circuit signal
- prevent synthetic domination
- avoid overly aggressive oversampling of small real sets

Recommended duration:

- about 60-70% of total training steps

## 6.3 Stage 3 - Real-Focused Calibration

Start from the **best Stage-2 checkpoint**.

Train on the same Stage-2 train manifests:

```text
data/manifests/full/source_manifests/queko/train.jsonl
data/manifests/full/source_manifests/qasmbench/train.jsonl
```

Sampling:

- group-balanced sampling
- synthetic = `0.5`
- real = `0.5`

Learning rate:

- lower than Stage 2

Purpose:

- calibrate more strongly toward real-circuit behavior
- retain enough synthetic exposure to reduce collapse / overfitting

Recommended duration:

- about 10-15% of total training steps

### 6.4 Current concrete stage-config realization

The current stage config is represented conceptually as:

```yaml
final_checkpoint: last
stages:
  - name: stage1_synthetic_warmup
    dataset.manifest_path: data/manifests/full/source_manifests/queko/train.jsonl
    sampler.kind: source_balanced

  - name: stage2_mixed_synthetic_real
    load_from: previous_best
    dataset.manifest_paths:
      - data/manifests/full/source_manifests/queko/train.jsonl
      - data/manifests/full/source_manifests/qasmbench/train.jsonl
    sampler.kind: group_balanced
    sampler.group_weights:
      synthetic: 0.7
      real: 0.3

  - name: stage3_real_calibration
    load_from: stage2_mixed_synthetic_real:best
    dataset.manifest_paths:
      - data/manifests/full/source_manifests/queko/train.jsonl
      - data/manifests/full/source_manifests/qasmbench/train.jsonl
    sampler.kind: group_balanced
    sampler.group_weights:
      synthetic: 0.5
      real: 0.5
```

---

## 7. Current dataset behavior

The dataset layer now lives in:

```text
src/kmw2/data/dataset.py
```

### 7.1 Current capabilities

The dataset code must support:

- loading from a single manifest path
- loading from multiple manifest paths directly
- loading from `.jsonl` manifests already present in `source_manifests/...`
- filtering by allowed source names where needed

### 7.2 Current intended usage

For the real staged run, the dataset is expected to consume the pre-existing manifests directly rather than rebuilding split recipes.

---

## 8. Current sampler behavior

The staged workflow introduces explicit sampler logic under:

```text
src/kmw2/training/samplers.py
```

### 8.1 Supported balancing modes

The current plan expects support for:

- **source-balanced** sampling
- **group-balanced** sampling

### 8.2 Source-balanced sampling

Procedure:

1. choose source bucket
2. sample uniformly within that bucket

### 8.3 Group-balanced sampling

Procedure:

1. choose a high-level group according to configured group weights
2. choose a source within that group
3. sample uniformly from that source bucket

In the current real run:

- synthetic group contains `queko`
- real group contains `qasmbench`

---

## 9. Current trainer behavior

Training logic now lives in:

```text
src/kmw2/training/trainer.py
```

### 9.1 Trainer responsibilities

The trainer now handles:

- model construction
- loss construction
- optimizer / scheduler setup
- run-directory creation
- checkpointing
- logging
- stage-to-stage checkpoint carry-over
- staged sampler selection

### 9.2 Current run-artifact structure

A typical run directory now contains artifacts such as:

```text
runs/kmw2/<run_name>/
|- run_config.json
|- logs/
|  |- train_metrics.json
|  |- train_metrics.jsonl
|  |- epoch_metrics.jsonl
|- checkpoints/
|  |- epoch_XXX.pt
|  |- best.pt
|  |- last.pt
|- stages/
```

### 9.3 Checkpoint carry-over rule

The current staged plan requires:

- Stage 2 may load from the previous best checkpoint
- Stage 3 must load from the best Stage-2 checkpoint

---

## 10. Current evaluation protocol

Evaluation logic now lives in:

```text
src/kmw2/evaluation/evaluate.py
```

### 10.1 Held-out evaluation policy

The validation/test protocol remains aligned with the old `real_full_run_canonical v1.41` meaning:

Held-out evaluation remains on:

- `QASMBench(val)`
- `QASMBench(test)`
- `RevLib(val)`
- `RevLib(test)`

using the pre-existing manifests:

```text
data/manifests/full/source_manifests/qasmbench/val.jsonl
data/manifests/full/source_manifests/qasmbench/test.jsonl
data/manifests/full/source_manifests/revlib/val.jsonl
data/manifests/full/source_manifests/revlib/test.jsonl
```

### 10.2 Current evaluation output metrics

The current eval loop is expected to report:

- `pst`
- `compile_seconds`
- `swap_overhead`
- `depth_increase`

along with supporting fields such as:

- `original_depth`
- `routed_depth`
- `original_gate_count`
- `routed_gate_count`
- operation counts
- mapping

### 10.3 Routing contract

The evaluation loop still:

1. obtains a hard mapping from the model
2. converts canonical indices back to native backend IDs
3. passes the mapping into Qiskit transpilation as `initial_layout`
4. computes post-routing metrics from the transpiled circuit

---

## 11. Current real-PST implementation

The PST metric is now expected to live in:

```text
src/kmw2/metrics/pst.py
```

### 11.1 Current PST definition

The current intended PST function is:

```text
PST = 100 * sum_x min(p_noisy(x), p_ideal(x))
```

where:

- `p_noisy(x)` = noisy simulated output probability for bitstring `x`
- `p_ideal(x)` = ideal simulated output probability for bitstring `x`

This is the distribution overlap between noisy and ideal output distributions.

### 11.2 Practical meaning

- `100.0` -> noisy and ideal distributions match exactly
- `0.0` -> no overlap

### 11.3 Measurement rule

To compute a valid PST from simulator counts, evaluation should use a **measured copy** of the routed circuit.

That means:

- if the transpiled circuit already contains measurement operations, use it as-is
- otherwise, create a copied circuit and append measurements before simulator execution

### 11.4 Noisy / ideal simulation rule

PST should be computed by:

1. simulating the measured routed circuit on a noisy simulator derived from the backend
2. simulating the same measured routed circuit on an ideal simulator with no noise model
3. comparing the two resulting count dictionaries using the overlap metric above

---

## 12. Current flexible CLI structure

The current package restores flexible run-sheet style control through:

```text
python -m kmw2.cli.main <command> ...
```

### 12.1 Main subcommands

- `train`
- `train-staged`
- `eval`
- `eval-one`

### 12.2 Important CLI override knobs

The current CLI is expected to support options such as:

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
- `--per-circuit-csv`
- `--per-circuit-json`
- `--summary-json`
- `--save-routed-qasm-dir`
- `--save-routed-qpy-dir`

### 12.3 Important staged-training caution

For staged runs, global CLI overrides like `--epochs` or `--lr` must be used carefully, because they can overwrite the stage-specific settings.

The preferred workflow is:

- keep stage-specific epoch/LR values inside the stage config
- use CLI flags only when deliberately overriding the full staged schedule

---

## 13. Current config layout

The packaged `kmw2` workflow assumes a config family like:

```text
configs/
|- base.yaml
|- train_staged.yaml
|- eval_qasmbench_val.yaml
|- eval_qasmbench_test.yaml
|- eval_revlib_val.yaml
|- eval_revlib_test.yaml
|- overrides/
|  |- trainer_default.yaml
|  |- loss_default.yaml
|  |- stages_existing_splits.yaml
|  |- eval_qasmbench_val.yaml
|  |- eval_qasmbench_test.yaml
|  |- eval_revlib_val.yaml
|  |- eval_revlib_test.yaml
```

If only `src/kmw2` is copied into the repo, these YAML files must be created manually.

---

## 14. Current run protocol

### 14.1 Training

The intended real run is now:

1. load existing source manifests
2. run `train-staged`
3. save staged checkpoints and logs under a chosen run directory

### 14.2 Validation / test

After training, run evaluation separately on:

- `QASMBench(val)`
- `QASMBench(test)`
- `RevLib(val)`
- `RevLib(test)`

using the final or chosen checkpoint from the staged run.

### 14.3 Optional routed artifact saving

Evaluation may additionally save:

- routed QASM files
- routed QPY files

for post-hoc inspection.

---

## 15. Updated interpretation rules

To avoid ambiguity, the current project should be interpreted using the following rules:

1. **The pre-existing source manifests define the official splits.**
   They are not to be regenerated for the real run.

2. **Rows of mapping tensors correspond to logical qubits / logical slots.**
   Columns correspond to canonical physical qubits.

3. **Model-side mapping tensors remain in canonical hardware order.**
   Native backend IDs appear only after canonical-to-native remapping.

4. **Staged training is now part of the official workflow.**
   It is not an optional documentation-only addition.

5. **Held-out QASMBench and RevLib evaluation remains semantically aligned with the old v1.41 experimental loop.**

6. **Real PST in the current plan is defined by noisy-vs-ideal measured distribution overlap unless replaced later by a recovered legacy helper.**

7. **If only `src/kmw2` is copied, external configs are still required.**

---

## 16. Practical delta summary from the old explicit v1.1 spec

Compared with the earlier v1.1 explicit specification, the current project changes are:

- repository moved to `src/kmw2`
- preprocessing, model, loss, trainer, eval, and PST metric are separated by package area
- staged training added
- sampler logic added
- existing source-manifest splits are used directly
- held-out evaluation remains on QASMBench / RevLib
- flexible CLI and config overrides restored
- PST now has an in-package real-valued evaluation path

What did **not** conceptually change:

- 27-qubit fixed-size framing
- canonical hardware view
- logical interaction featurization
- UNet-style mapping model
- Sinkhorn training / Hungarian inference
- routing-aware proxy training philosophy

---

## 17. Final summary

The current plan is best understood as:

> a **v1.1-derived initial-mapping model** preserved at the modeling level, but **repackaged into `kmw2`** with staged training, direct existing-manifest consumption, flexible CLI control, and explicit held-out evaluation for QASMBench and RevLib.

So this updated design is **not pure historical v1.1**, but rather the **current operational implementation plan** for your repository.
