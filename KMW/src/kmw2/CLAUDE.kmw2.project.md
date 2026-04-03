# CLAUDE.md

## Project Overview

This repository contains multiple branches/packages of quantum initial-mapping work.  
The **active target for this project is `src/kmw2`**.

`kmw2` should be understood as:

- a **v1.1-derived initial qubit mapper**
- repackaged into a more modern **CLI + YAML + manifest** workflow
- extended with **staged training**
- using **existing source-manifest splits**
- evaluated with **held-out QASMBench + RevLib**
- and currently using an explicit in-package **real PST** metric.

This is **not** the same thing as:

- `src/kmw1` — canonical-hardware v1.41-style branch
- `src/kmw` — older/other package branch
- the example uploaded `CLAUDE.md`, which is only a formatting/detail reference, not the project definition.

The core modeling intention inherited from v1.1 is still:

- fixed **27-qubit** framing
- `BackendV2`-based hardware extraction
- canonical hardware reindexing
- logical interaction matrix `W`
- active-mask `m`
- **5-channel `(27 x 27)`** tensor input
- logical / physical token conditioning
- Sinkhorn during training
- Hungarian during inference
- label-free proxy training rather than supervised ground-truth mappings.


## Project Status

This is **not** a pure historical reproduction of old v1.1 anymore.

Current `kmw2` adds:

- `src/kmw2` package layout
- unified CLI at `kmw2.cli.main`
- YAML-driven configs
- staged training
- direct consumption of **existing** `source_manifests/.../*.jsonl`
- flexible run/output path control
- held-out eval configs
- explicit PST metric under `src/kmw2/metrics/pst.py`.

Use this mental model:

> Preserve the **v1.1 mapper core**, but run it inside a **modern `kmw2` experiment shell**.


## Tech Stack

- Python 3.10+
- PyTorch
- Qiskit
- `qiskit-ibm-runtime`
- `qiskit-aer`
- PyYAML
- NumPy
- Pandas
- SciPy if Hungarian / assignment utilities require it in your local code path

Primary environment name:

- `graphqmap_pascal`

Current backend style:

- IBM `BackendV2` / `FakeBackendV2`
- current default run backend in this project context: `FakeTorontoV2`

The active PST path uses Aer simulation, so `qiskit-aer` is required for real PST evaluation.


## High-Priority File Read Strategy

Claude should **not** read the whole repo by default.  
Use the following **minimal core read set** first.

### Core read set for most tasks

1. `Design_Plan_v1_1_Current_kmw2_Implementation_Spec.md`  
   Primary research/implementation authority for current `kmw2`. Read this first.

2. `src/kmw2/cli/main.py`  
   Read for actual CLI behavior, override precedence, staged-train subcommand behavior, output-path handling.

3. `src/kmw2/training/trainer.py`  
   Read for trainer lifecycle, checkpointing, run-dir layout, optimizer/scheduler behavior.

4. `src/kmw2/training/samplers.py`  
   Read for staged source-balanced and group-balanced sampler semantics.

5. `src/kmw2/data/dataset.py`  
   Read for manifest loading, multi-manifest support, `.jsonl` handling, source filtering.

6. `src/kmw2/preprocessing/pipeline.py`  
   Read for top-level tensor-building flow.

7. `src/kmw2/preprocessing/extractor.py`  
   Read for hardware extraction from BackendV2.

8. `src/kmw2/preprocessing/canonical_indexer.py`  
   Read for canonical hardware ordering / `p` mapping semantics.

9. `src/kmw2/preprocessing/featurizer.py`  
   Read for circuit feature construction (`W`, `m`, input tensor channels).

10. `src/kmw2/models/model.py`  
    Read for top-level UNet+token mapper structure.

11. `src/kmw2/models/layers.py`  
    Read for attention, token-conditioning, and lower-level blocks.

12. `src/kmw2/losses/loss.py`  
    Read for actual proxy-loss implementation.

13. `src/kmw2/evaluation/evaluate.py`  
    Read for routing/eval flow, PST hook, routed artifact saving, per-circuit metrics.

14. `src/kmw2/metrics/pst.py`  
    Read for current real PST definition.

### Config files Claude should read when the task concerns running experiments

15. `configs/base.yaml`
16. `configs/train_staged.yaml`
17. `configs/overrides/stages_existing_splits.yaml`
18. `configs/overrides/trainer_default.yaml`
19. `configs/overrides/loss_default.yaml`
20. the relevant eval YAML(s):
    - `configs/overrides/eval_qasmbench_val.yaml`
    - `configs/overrides/eval_qasmbench_test.yaml`
    - `configs/overrides/eval_revlib_val.yaml`
    - `configs/overrides/eval_revlib_test.yaml`

### Files to read only when relevant

- `real_full_run_canonical v1.41.md`  
  Read only for **evaluation protocol semantics / run-sheet style reference**, not as the architectural source of truth.
- `kmw2` run-guide markdown files  
  Read only when the task is about command usage or reproducing the current experiment shell.
- `src/kmw1/...` or `src/kmw/...`  
  Read only if the user explicitly asks to compare branches or port logic.

### Important token-efficiency rule

Default behavior should be:

- read the **spec**
- then read only the **minimum subset of code files relevant to the requested task**
- do **not** scan unrelated packages or old branches automatically


## Project Structure

Current active structure:

```text
KMW/
├─ scripts/
├─ src/
│  ├─ kmw/
│  ├─ kmw1/
│  ├─ kmw1_backup/
│  ├─ kmw2/
│  │  ├─ cli/
│  │  │  └─ main.py
│  │  ├─ data/
│  │  │  └─ dataset.py
│  │  ├─ evaluation/
│  │  │  └─ evaluate.py
│  │  ├─ losses/
│  │  │  └─ loss.py
│  │  ├─ metrics/
│  │  │  └─ pst.py
│  │  ├─ models/
│  │  │  ├─ layers.py
│  │  │  └─ model.py
│  │  ├─ preprocessing/
│  │  │  ├─ extractor.py
│  │  │  ├─ canonical_indexer.py
│  │  │  ├─ featurizer.py
│  │  │  └─ pipeline.py
│  │  ├─ training/
│  │  │  ├─ samplers.py
│  │  │  └─ trainer.py
│  │  ├─ __init__.py
│  │  └─ utils.py
│  └─ kmw2_old/
├─ data/
│  ├─ manifests/
│  │  └─ full/
│  │     └─ source_manifests/
│  │        ├─ qasmbench/
│  │        ├─ queko/
│  │        ├─ revlib/
│  │        ├─ mqt_bench/
│  │        ├─ mlqd/
│  │        ├─ catalog.json
│  │        ├─ train.jsonl
│  │        ├─ val.jsonl
│  │        └─ test.jsonl
│  ├─ circuits/
│  └─ circuits_v2/
└─ configs/