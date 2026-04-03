# Combined Design Plan v1.3

## 0. Document Status

This document is the combined implementation specification for the KMW project.

It merges and consolidates the following inputs into one place:

- `project overview.pdf`
- `readme.md`
- `file_explanation_v1.pdf`
- `Design Plan v1.1 - reformatted.md`
- `Design Plan v1.1.1 — Clarification Patch.md`
- `Attn_Unet_27_SPEC - reformatted.md`
- `build_manifest_full.py`
- `dataset.py` (manifest_full-compatible revision)
- `full_run_routing_real_eval_design_plan.md`

This file is intended to be the working implementation authority for the current KMW build.

### 0.1 Precedence rule for conflicts

If two source documents disagree, use this order of authority:

1. explicit implementation decisions locked during the design review for v1.2,
2. `full_run_routing_real_eval_design_plan.md` for full-run manifest/evaluation workflow changes,
3. `Attn_Unet_27_SPEC.md` for the mapper backbone,
4. `Design Plan v1.1.1 — Clarification Patch.md`,
5. `Design Plan v1.1 - reformatted.md`,
6. `project overview.pdf`, `readme.md`, and `file_explanation_v1.pdf` for structure, intent, and non-conflicting context.

### 0.2 Locked v1.2 implementation resolutions

The following design-review decisions are locked for v1.2 and are part of this combined plan:

#### 0.2.1 Mapper backbone authority

Treat `Attn_Unet_27_SPEC.md` as the final implementation authority for the mapper backbone.

Implement the mapper as:

- `Interpolate -> Conv3x3 -> attention` on the resize stages,
- final output head `Conv3x3(128 -> 1)`,
- no stride-2 conv / transpose-conv backbone substitution unless a later revision explicitly overrides the spec.

#### 0.2.2 Spatial mapper normalization

Use no normalization layers inside the spatial mapper backbone.

Lock the spatial blocks as:

- `down1`: conv + ReLU
- `down2`: interpolate + conv + ReLU + attention
- `bottleneck`: interpolate + conv + ReLU + attention
- `up1`: interpolate + conv + ReLU + additive skip + attention
- `head`: final projection only

Do **not** use BatchNorm, LayerNorm, or GroupNorm in the spatial conv blocks.

Normalization remains only in token / reindexer MLPs where explicitly specified.

#### 0.2.3 Backend target and resolver

Use a generic backend extractor over a `BackendV2`-like object.

For the current project phase:

- fixed hardware size: `n = 27`,
- default local smoke backend: `FakeTorontoV2`,
- current implementation target: Toronto-style 27-qubit backend,
- backend choice should be supplied by config / resolver,
- swapping to another 27-qubit backend later should require changes only in:
  - config,
  - backend resolver,
  - cached backend tensors / manifests if needed,
- mapper / reindexer code should not need to change when the backend is swapped, as long as `n = 27`.

#### 0.2.4 Project/package layout

Use pure `src/kmw` implementation only.

Do **not** preserve legacy `mqm/...` or `training/...` paths as final implementation paths.
Legacy paths may still be mentioned only as historical references.

#### 0.2.5 Manifest policy

Lock the manifest policy as:

- format: `JSONL`,
- split names remain `train`, `val`, `test`,
- use relative paths only,
- split manifests are authoritative and should be reused once created,
- the full-run manifest root is `data/manifests/full/`,
- primitive manifests are stored under `data/manifests/full/source_manifests/<source>/`,
- combined reusable experiment manifests are stored under `data/manifests/full/recipes/<recipe_name>/`,
- source splits are assigned deterministically **per source**, not by one global shuffle,
- smoke behavior is determined by which manifest/recipe is selected, not by special split names,
- keep enough metadata for filtering, reproducibility, and debugging, but not unnecessary bulk.

Recommended per-row schema for the full manifest family:

```json
{
  "id": "mqt_bench_example_000123",
  "source": "mqt_bench",
  "split": "train",
  "qasm_relpath": "data/circuits_v2/qasm/mqt_bench/.../example_000123.qasm",
  "k_logical": 12,
  "num_1q": 85,
  "num_2q": 47,
  "is_disconnected_logical_graph": false,
  "passed_parse": true,
  "passed_filter": true,
  "filter_tags": [],
  "include": true,
  "cache_key": "mqt_bench_example_000123",
  "dataset_version": "circuit_v2",
  "source_role": "train_side"
}
```

#### 0.2.6 Cache granularity

Use `data/cache/` for:

- backend tensors cached once per backend,
- per-circuit parsed / native features cached once per circuit file,
- optional validated preprocessing-side metadata that improves reproducibility and speed.

#### 0.2.7 Environment lock

Current implementation environment:

- Python `3.10.20`
- Torch `2.10.0+cu128`
- Qiskit `2.3.1`

#### 0.2.8 Dataset loader compatibility

The dataset loader must remain manifest-driven and native-frame only.

Specifically:

- `dataset.py` reads manifest rows,
- `__getitem__` resolves `qasm_relpath` against `project_root`,
- the returned sample contains native-frame tensors only,
- the returned metadata preserves the manifest content via `asdict(row)`,
- the manifest row now includes `dataset_version` and `source_role` in addition to the original row fields.

#### 0.2.9 Full-run routed final evaluation

The project remains an **initial mapper**, not a full router.

For the full run, final evaluation is extended as follows:

- routed evaluation is implemented in `evaluate.py`, not in training,
- the model still trains only on proxy objectives,
- the official routed-eval transpilation setting is `optimization_level = 0`,
- the official real PST metric includes both gate error and readout error,
- the official SWAP-overhead report includes both exact inserted SWAP count and added 2Q op count,
- routed evaluation is an extension of the current eval flow, not a rewrite of the model or training loop.

---

## 1. High-Level Project Overview

The following section preserves the high-level conceptual summary for the project and serves as the intent layer above the implementation sections.

1. What the project is trying to do
The project is a learned initial qubit mapper for a fixed 27-qubit IBM BackendV2
device.
Its job is not to perform full routing. Its job is to choose a strong initial logical-tophysical placement so that later routing incurs less performance loss, especially in terms
of PST degradation, SWAP overhead, and depth increase. The output is still a one-toone mapping between logical qubits and physical qubits.
2. The main conceptual change
The old design used a fixed canonical indexing module to reorder hardware qubits
before the mapper saw them.
That is now gone.
The new design uses a SoftPermutationReindexer instead. The model now works like
this:
native circuit/backend tensors → learned soft reindexing → mapper → decode back
to native frame → assignment/loss
So the model no longer depends on a handcrafted canonical qubit order. Instead, it
learns an internal ordering that is easier for the mapper to use, and then converts the
result back to the original native hardware IDs before producing the final mapping.
3. How to think about the model architecture
The design is now closer to an Imagen-style conditioning idea:
•

the circuit is treated like the main spatial object, similar to an “image”

•

the hardware is treated like conditioning information, similar to “text”

Concretely:
•

the U-Net input is the circuit-side matrix A

•

the queries in attention come from the circuit-side feature maps

•

the keys/values come from hardware-only tokens

So the hardware is no longer mixed into the U-Net input as a big 5-channel image in
the old way. Instead, the hardware conditions the circuit representation through
attention. This is one of the biggest simplifications in the revised plan.

4. What the main inputs mean now
You standardized the notation around four main objects:
•

A = circuit representation

•

B = hardware adjacency / connectivity

•

c1 = per-physical-qubit cost

•

c2 = per-physical-edge cost

There is also:
•

D = shortest-path distance matrix on the hardware graph

But D is not part of the hardware tokens. It is kept for the routing-related loss
terms, especially the SWAP/depth proxies.
5. How the hardware is represented
Each physical qubit becomes one hardware token.
For physical qubit j, the token is built from:
xj(hw)=[ B[j,:] ∥ c2[j,:] ∥ c1[j] ]x_j^{(\mathrm{hw})} = [\,B[j,:] \;\Vert\; c2[j,:] \;\Vert\;
c1[j]\,]xj(hw)=[B[j,:]∥c2[j,:]∥c1[j]]
So each token contains:
•

which other physical qubits j is connected to

•

how costly those physical couplings are

•

how costly qubit j itself is

Then these raw token vectors are embedded with an MLP and used as the keys and
values in cross-attention. This means the mapper sees the circuit as the main structure,
while consulting hardware quality and topology through attention.
6. How the circuit is represented
The circuit is not represented as a simple binary logical adjacency matrix.
Instead, A is a weighted logical interaction matrix:
•

off-diagonal entries describe how strongly pairs of logical qubits interact, mainly
through 2-qubit gate activity

•

diagonal entries summarize per-logical-qubit burden

For now, the diagonal is kept simple and stable:
A[u,u]=log⁡(1+αN1Q(u)+βN2Q-part(u))A[u,u] = \log(1 + \alpha N_{1Q}(u) + \beta
N_{2Q\text{-part}}(u))A[u,u]=log(1+αN1Q(u)+βN2Q-part(u))
So the diagonal says, roughly, “how busy or important is logical qubit uuu?”
You explicitly decided to keep this relatively simple for the first stable version, with the
option to revisit richer gate-type weighting later. This is a stabilization-first choice, not a
claim that richer circuit features are never useful.
7. How the model learns
The model still learns through a proxy objective, not through direct target layouts.
The training assignment head remains:
•

Sinkhorn during training

•

Hungarian during inference

And the task objective remains conceptually the same:
•

PST-related term

•

SWAP proxy

•

depth proxy

The crucial change is where this loss is computed: it is now computed in the native
frame after decoding, not in a canonicalized frame. That makes the learning objective
align better with the actual backend indices and removes the old dependence on a fixed
canonical permutation.
8. How the reindexer is trained
The reindexer does not solve the mapping directly.
Its only purpose is to learn a better internal ordering of:
•

logical rows

•

hardware rows/columns

Training is split into two passes:
•

one pass updates the mapper

•

one pass updates the reindexer

The mapper is optimized only for the mapping task itself.
The reindexer can additionally use locality/consistency auxiliary losses, but for the first
stable run you decided to start with those auxiliary losses off and only turn them on
later in stages. That keeps the first experiment focused on answering one question: does
the new pipeline train stably at all?
9. The stability philosophy of the whole plan
A major theme of the revised plan is numerical stability first.
That is why you locked these choices:
•

batch size = 1

•

no infinities anywhere

•

explicit validation of each major loss term

•

fail loudly on NaN/Inf

•

keep D finite

•

keep non-edge handling explicit instead of encoding impossibility with ∞

This is not just an optimization detail. It is part of the design philosophy: the model
should learn from cleanly separated signals rather than from brittle sentinel values.
10. How invalid edges are handled
A subtle but important part is how non-adjacent physical pairs are represented.
You decided that:
•

B[i,j]B[i,j]B[i,j] explicitly says whether a physical edge exists

•

c2[i,j]=0c2[i,j]=0c2[i,j]=0 when B[i,j]=0B[i,j]=0B[i,j]=0

At first glance that can feel dangerous, because zero can look like “good.”
But the model does not see c2 alone. It sees B and c2 together, so it can
learn the difference between:
•

“no edge exists”

•

“edge exists and has low error”

•

“edge exists and has high error”

On top of that, the loss still punishes far or poor placements through the distance-based
routing proxy. So the model is not expected to infer impossibility from one scalar alone;
it learns from the combination of representation and training objective.
11. How tensors are normalized
You also settled a clean normalization policy:
•

for A and D: max normalization with a small ε\varepsilonε

•

for c1 and c2: z-score style normalization with a small ε\varepsilonε

•

for c2 specifically: normalize over valid edges only, so the many zero nonedges do not corrupt the scale

So the representation is not only finite, but also numerically comparable across samples.
This matters because the model is learning from several feature families with very
different raw scales.
12. How you will evaluate it
The dataset policy is now also staged.
First: smoke test
Run a short stabilization phase on MQT Bench only, mainly to verify that the new
architecture and loss do not explode.
Then: main training
Train on the three “training-side” sources from the GraphQMap environment:
•

QUEKO

•

MLQD

•

MQT Bench

Final test
Evaluate on the more realistic held-out sources:
•

QASMBench

•

RevLib

That aligns with the GraphQMap dataset structure, where QUEKO/MLQD/MQT are the
main training-side corpora and QASMBench/RevLib are separate real-circuit sets. The
repo README lists 900 QUEKO circuits, 4,443 MLQD circuits, 1,448 MQT Bench circuits,
111 QASMBench circuits, and 263 RevLib circuits.
13. How to think about fairness of comparison
You are using the same circuit environment as the comparison repo, but not necessarily
the same supervision target.
That is important because GraphQMap’s labels are backend-specific: the README says
QUEKO and MLQD labels are tied to particular target backends such as Aspen-4, Tokyo,
Rochester, Sycamore, Grid 5x5, and Melbourne. Those are not the same as your fixed
IBM 27-qubit BackendV2 setup.
So the fair interpretation is:
•

same dataset ecosystem

•

different mapping method

•

label-free / proxy-loss training on your side

•

backend-specific label supervision on their side

That distinction should be stated clearly when you later discuss results.
14. One-sentence summary
At a high level, your project is now:
a stability-first, label-free initial qubit mapper that learns in native hardware space,
uses a soft reindexer instead of fixed canonical IDs, treats the circuit as the main UNet object and the hardware as attention conditioning, and is trained first for
stability and then for fair comparison on the GraphQMap dataset environment


---

## 2. Final Workspace and File Structure

### 2.1 Project tree

# KMW Project Structure

This folder is the isolated workspace for the KMW method inside the shared `GraphQMap` repository.

## Design rule

- Keep the structure **small and readable**.
- Separate by function, not by excessive file splitting.
- Put **all KMW-specific code** under `KMW/`.
- Keep `KMW/data/circuits_v2/qasm/` as the raw-circuit area for the current full-run dataset family.
- Keep `KMW/data/manifests/full/` as the authoritative manifest root for the full run.

---

## Final recommended tree

```text
KMW/
├── readme.md
├── Combined Design Plan v1.2.md
│
├── configs/
│   ├── base.yaml
│   ├── smoke_mqt.yaml
│   ├── train_main.yaml
│   └── eval.yaml
│
├── docs/
│   ├── project_overview.md
│   ├── data_protocol.md
│   └── training_protocol.md
│
├── data/
│   ├── readme.md
│   ├── circuits_v2/
│   │   └── qasm/
│   │       ├── queko/
│   │       ├── mlqd/
│   │       ├── mqt_bench/
│   │       ├── qasmbench/
│   │       ├── revlib/
│   │       └── benchmarks/
│   ├── manifests/
│   │   └── full/
│   │       ├── source_manifests/
│   │       ├── recipes/
│   │       └── catalog.json
│   └── cache/
│
├── src/
│   └── kmw/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── dataset.py
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   └── pipeline.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── model.py
│       ├── losses/
│       │   ├── __init__.py
│       │   └── loss.py
│       ├── training/
│       │   ├── __init__.py
│       │   └── trainer.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── evaluate.py
│       ├── cli/
│       │   ├── __init__.py
│       │   └── main.py
│       └── utils.py
│
├── scripts/
│   ├── build_manifests.py
│   ├── build_manifest_full.py
│   ├── run_smoke.sh
│   ├── run_main.sh
│   └── run_eval.sh
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── smoke/
│
├── runs/
└── artifacts/
```

### 2.2 One-line purpose for each file/folder

One-line purpose for each file/folder

Root

- `readme.md` — quick guide to how the KMW project is organized.
- `Combined Design Plan v1.2.md` — working implementation authority for the KMW method.

`configs/`

- `configs/` — experiment configuration files.
- `configs/base.yaml` — shared defaults for paths, model size, loss weights, and training settings.
- `configs/smoke_mqt.yaml` — config for the MQT-only stability smoke run.
- `configs/train_main.yaml` — config for the main training run on the full manifest family.
- `configs/eval.yaml` — config for evaluation, including routed final evaluation options.

`docs/`

- `docs/` — short project documents other than the main design plan.
- `docs/project_overview.md` — high-level explanation of the method.
- `docs/data_protocol.md` — rules for dataset layout, filtering, and manifests.
- `docs/training_protocol.md` — rules for training stages, logging, and fail-fast checks.

`data/`

- `data/` — local data workspace for KMW.
- `data/readme.md` — explains what each data subfolder stores.
- `data/circuits_v2/qasm/` — raw circuit datasets used by the current full-run pipeline.
- `data/manifests/full/` — authoritative full-run manifest root.
- `data/manifests/full/source_manifests/` — primitive per-source manifests with deterministic per-source train/val/test splits.
- `data/manifests/full/recipes/` — reusable combined manifests assembled from primitive source manifests.
- `data/manifests/full/catalog.json` — reproducibility/inspection catalog for one manifest-build run.
- `data/cache/` — cached backend tensors, circuit features, and preprocessing results.

`src/`

- `src/` — container for importable source code only.

`src/kmw/`

- `src/kmw/` — Python package root for the KMW project.
- `src/kmw/__init__.py` — package initializer.
- `src/kmw/utils.py` — shared helper functions used across the project.

`src/kmw/data/`

- `src/kmw/data/` — dataset loading and manifest logic.
- `src/kmw/data/__init__.py` — package initializer.
- `src/kmw/data/dataset.py` — main dataset code, manifest loading, full-manifest row parsing, caching, and dataloader-facing sample return logic.

`src/kmw/preprocessing/`

- `src/kmw/preprocessing/` — tensor construction and preprocessing logic.
- `src/kmw/preprocessing/__init__.py` — package initializer.
- `src/kmw/preprocessing/pipeline.py` — backend extraction, circuit featurization, normalization, token construction, distance computation, and validation checks.

`src/kmw/models/`

- `src/kmw/models/` — all learnable model components.
- `src/kmw/models/__init__.py` — package initializer.
- `src/kmw/models/model.py` — reindexer, hardware token encoder, U-Net mapper, cross-attention, and assignment helpers.

`src/kmw/losses/`

- `src/kmw/losses/` — training objectives and proxy metrics.
- `src/kmw/losses/__init__.py` — package initializer.
- `src/kmw/losses/loss.py` — task loss, PST-related proxy terms, swap/depth proxies, and reindexer auxiliary losses.

`src/kmw/training/`

- `src/kmw/training/` — training loop and optimization logic.
- `src/kmw/training/__init__.py` — package initializer.
- `src/kmw/training/trainer.py` — two-pass training loop, optimizer setup, checkpointing, logging, and fail-fast checks.

`src/kmw/evaluation/`

- `src/kmw/evaluation/` — inference and evaluation code.
- `src/kmw/evaluation/__init__.py` — package initializer.
- `src/kmw/evaluation/evaluate.py` — inference pipeline, hard reindexing, proxy metrics, routed final evaluation, and summary/report generation.

`src/kmw/cli/`

- `src/kmw/cli/` — command-line entrypoints.
- `src/kmw/cli/__init__.py` — package initializer.
- `src/kmw/cli/main.py` — unified CLI entrypoint for training, evaluation, and routed-eval configuration wiring.

`scripts/`

- `scripts/` — thin helpers for common project runs.
- `scripts/build_manifests.py` — older/simple manifest builder retained for compatibility and smoke-era workflows.
- `scripts/build_manifest_full.py` — authoritative full-run manifest builder for `circuit_v2`.
- `scripts/run_smoke.sh` — runs the MQT smoke test.
- `scripts/run_main.sh` — runs the main training job.
- `scripts/run_eval.sh` — runs final evaluation on the configured held-out recipe.

`tests/`

- `tests/` — test suite for the KMW project.
- `tests/unit/` — small isolated tests for preprocessing, model pieces, losses, and routed-eval helpers.
- `tests/integration/` — multi-module tests such as dataset-to-forward-pass and one-sample routed-eval checks.
- `tests/smoke/` — short end-to-end sanity tests like one-batch training.

`runs/`

- `runs/` — temporary experiment outputs such as run folders and intermediate results.

`artifacts/`

- `artifacts/` — saved checkpoints, CSV summaries, routed-eval reports, figures, and final outputs.

Why this version is the right one

This structure is intentionally minimal:

- one main file for dataset logic
- one main file for preprocessing logic
- one main file for model logic
- one main file for loss logic
- one main file for training logic
- one main file for evaluation logic
- one CLI entrypoint
- one dedicated full-manifest builder for the full run

So the implementation stays modular, but the file count stays low enough to read easily.

Recommended internal responsibility split

To keep the number of files low, each major file should absorb related functions:

- `dataset.py` should handle manifests, filtering hooks, and sample return.
- `pipeline.py` should handle backend extraction, `A/B/c1/c2/D` construction, normalization, and validators.
- `model.py` should contain the reindexer, token encoder, mapper, and assignment helpers.
- `loss.py` should contain both task loss and reindexer auxiliary losses.
- `trainer.py` should contain both passes of training, logging, fail-fast checks, and checkpoint saving.
- `evaluate.py` should contain inference, proxy evaluation, routed final evaluation, and report summaries.

Import style

Use imports from the `kmw` package root, for example:

```python
from kmw.data.dataset import ...
from kmw.preprocessing.pipeline import ...
from kmw.models.model import ...
from kmw.losses.loss import ...
from kmw.training.trainer import ...
```

This is the reason for the `src/kmw/` structure:

- `KMW/` identifies the workspace/project,
- `src/` identifies importable source code,
- `kmw/` identifies the Python package namespace.

Immediate next-step note

The first milestone remains the MQT smoke pipeline. The full-run path now assumes the `circuit_v2` dataset family and the full manifest builder are available, after which the same `src/kmw` codepath scales to train-side, held-out, and benchmark recipes.

Short update summary:

- kept the minimal clean structure
- kept the standard `src/kmw/` package layout
- kept one main `.py` file per subsystem
- updated the data and manifest tree for `circuit_v2`
- added `build_manifest_full.py`
- updated evaluation ownership to include routed final evaluation

---

## 3. Full Base Implementation Specification (v1.1)


The following is the full base implementation specification that defines the main project logic.

# Design Plan v1.1 — Implementation Specification

## Table of Contents

- [0. Document Status](#0-document-status)
- [1. Objective](#1-objective)
- [2. Non-Goals](#2-non-goals)
- [3. Locked Design Decisions](#3-locked-design-decisions)
  - [3.1 Canonical indexing is removed](#31-canonical-indexing-is-removed)
  - [3.2 Mapper conditioning style](#32-mapper-conditioning-style)
  - [3.3 U-Net input](#33-u-net-input)
  - [3.4 Hardware token definition](#34-hardware-token-definition)
  - [3.5 Distance matrix role](#35-distance-matrix-role)
  - [3.6 No infinities anywhere](#36-no-infinities-anywhere)
  - [3.7 Batch size](#37-batch-size)
  - [3.8 Auxiliary losses](#38-auxiliary-losses)
- [4. Notation](#4-notation)
  - [4.1 Dimensions](#41-dimensions)
  - [4.2 Circuit tensors](#42-circuit-tensors)
  - [4.3 Hardware tensors](#43-hardware-tensors)
  - [4.4 Reindexer outputs](#44-reindexer-outputs)
  - [4.5 Reordered tensors](#45-reordered-tensors)
  - [4.6 Mapper outputs](#46-mapper-outputs)
  - [4.7 Decode rule](#47-decode-rule)
- [5. Dataset Protocol](#5-dataset-protocol)
  - [5.1 Phase A — stability smoke test](#51-phase-a-stability-smoke-test)
  - [5.2 Phase B — main comparison training](#52-phase-b-main-comparison-training)
  - [5.3 Final evaluation](#53-final-evaluation)
  - [5.4 Hard inclusion rules](#54-hard-inclusion-rules)
  - [5.5 Keep disconnected logical graphs](#55-keep-disconnected-logical-graphs)
  - [5.6 Labels](#56-labels)
  - [5.7 Train/val splitting](#57-trainval-splitting)
  - [5.8 Source balancing](#58-source-balancing)
- [6. Backend Extraction](#6-backend-extraction)
  - [6.1 Input](#61-input)
  - [6.2 Output](#62-output)
  - [6.3 Hardware adjacency B](#63-hardware-adjacency-b)
  - [6.4 Per-qubit cost c1](#64-per-qubit-cost-c1)
  - [6.5 Per-edge cost c2](#65-per-edge-cost-c2)
  - [6.6 Distance matrix D](#66-distance-matrix-d)
  - [6.7 Backend extractor invariants](#67-backend-extractor-invariants)
- [7. Circuit Featurization](#7-circuit-featurization)
  - [7.1 Logical qubit count](#71-logical-qubit-count)
  - [7.2 Logical mask m](#72-logical-mask-m)
  - [7.3 Off-diagonal entries of A](#73-off-diagonal-entries-of-a)
  - [7.4 Default 2Q gate-type weights](#74-default-2q-gate-type-weights)
  - [7.5 Diagonal entries of A](#75-diagonal-entries-of-a)
  - [7.6 Padding](#76-padding)
  - [7.7 Symmetry](#77-symmetry)
  - [7.8 Degenerate circuit rejection](#78-degenerate-circuit-rejection)
- [8. Normalization and Finite-Value Policy](#8-normalization-and-finite-value-policy)
  - [8.1 Global epsilon](#81-global-epsilon)
  - [8.2 Circuit matrix A](#82-circuit-matrix-a)
  - [8.3 Hardware adjacency B](#83-hardware-adjacency-b)
  - [8.4 Per-qubit cost c1](#84-per-qubit-cost-c1)
  - [8.5 Per-edge cost c2](#85-per-edge-cost-c2)
  - [8.6 Distance matrix D](#86-distance-matrix-d)
  - [8.7 Summary of finite rules](#87-summary-of-finite-rules)
- [9. Reindexer](#9-reindexer)
  - [9.1 File](#91-file)
  - [9.2 Components](#92-components)
  - [9.3 Reindexer hidden dimension](#93-reindexer-hidden-dimension)
  - [9.4 Logical branch input features](#94-logical-branch-input-features)
  - [9.5 Logical branch MLP](#95-logical-branch-mlp)
  - [9.6 Logical slot prototypes](#96-logical-slot-prototypes)
  - [9.7 Logical score matrix](#97-logical-score-matrix)
  - [9.8 Hardware branch input features](#98-hardware-branch-input-features)
  - [9.9 Hardware branch MLP](#99-hardware-branch-mlp)
  - [9.10 Hardware slot prototypes](#910-hardware-slot-prototypes)
  - [9.11 Hardware score matrix](#911-hardware-score-matrix)
  - [9.12 Reindexer Sinkhorn](#912-reindexer-sinkhorn)
  - [9.13 Reordered tensors](#913-reordered-tensors)
- [10. Mapper](#10-mapper)
  - [10.1 File](#101-file)
  - [10.2 Input contract](#102-input-contract)
  - [10.3 Hardware token encoder](#103-hardware-token-encoder)
  - [10.4 No logical tokens](#104-no-logical-tokens)
  - [10.5 U-Net backbone](#105-u-net-backbone)
  - [10.6 Cross-attention placement](#106-cross-attention-placement)
  - [10.7 Cross-attention rule](#107-cross-attention-rule)
  - [10.8 No timestep embedding](#108-no-timestep-embedding)
- [11. Decode and Assignment](#11-decode-and-assignment)
  - [11.1 Decode to native frame](#111-decode-to-native-frame)
  - [11.2 Training assignment](#112-training-assignment)
  - [11.3 Inference assignment](#113-inference-assignment)
  - [11.4 Mapping output](#114-mapping-output)
  - [11.5 Mapper Sinkhorn default](#115-mapper-sinkhorn-default)
- [12. Task Loss](#12-task-loss)
  - [12.1 Active logical set](#121-active-logical-set)
  - [12.2 Off-diagonal circuit mass](#122-off-diagonal-circuit-mass)
  - [12.3 PST 1Q term](#123-pst-1q-term)
  - [12.4 PST 2Q term](#124-pst-2q-term)
  - [12.5 PST total](#125-pst-total)
  - [12.6 SWAP proxy](#126-swap-proxy)
  - [12.7 Depth proxy](#127-depth-proxy)
  - [12.8 Total task loss](#128-total-task-loss)
  - [12.9 Dummy logical rows](#129-dummy-logical-rows)
- [13. Reindexer Auxiliary Losses](#13-reindexer-auxiliary-losses)
  - [13.1 Locality loss](#131-locality-loss)
  - [13.2 Consistency loss](#132-consistency-loss)
  - [13.3 Reindexer objective](#133-reindexer-objective)
  - [13.4 Default staged coefficients](#134-default-staged-coefficients)
- [14. Training Loop](#14-training-loop)
  - [14.1 Files](#141-files)
  - [14.2 Optimizers](#142-optimizers)
  - [14.3 Gradient clipping](#143-gradient-clipping)
  - [14.4 Pass A — mapper update](#144-pass-a-mapper-update)
  - [14.5 Pass B — reindexer update](#145-pass-b-reindexer-update)
  - [14.6 Temperature scheduling](#146-temperature-scheduling)
  - [14.7 Batch size](#147-batch-size)
- [15. Inference Protocol](#15-inference-protocol)
  - [15.1 Steps](#151-steps)
  - [15.2 Output format](#152-output-format)
- [16. Validation and Fail-Safes](#16-validation-and-fail-safes)
  - [16.1 Tensor checks](#161-tensor-checks)
  - [16.2 Loss checks](#162-loss-checks)
  - [16.3 Sinkhorn sanity checks](#163-sinkhorn-sanity-checks)
  - [16.4 Gradient checks](#164-gradient-checks)
  - [16.5 Hard-fail conditions](#165-hard-fail-conditions)
  - [16.6 Warning conditions](#166-warning-conditions)
  - [16.7 Failure report](#167-failure-report)
- [17. File and Module Layout](#17-file-and-module-layout)
  - [17.1 backend_extractor.py](#171-backendextractorpy)
  - [17.2 circuit_featurizer.py](#172-circuitfeaturizerpy)
  - [17.3 dataset_graphqmap.py](#173-datasetgraphqmappy)
  - [17.4 reindexer.py](#174-reindexerpy)
  - [17.5 hardware_token_encoder.py](#175-hardwaretokenencoderpy)
  - [17.6 unet_mapper.py](#176-unetmapperpy)
  - [17.7 assignment.py](#177-assignmentpy)
  - [17.8 loss_task.py](#178-losstaskpy)
  - [17.9 loss_reindexer.py](#179-lossreindexerpy)
  - [17.10 samplers.py](#1710-samplerspy)
  - [17.11 checks.py](#1711-checkspy)
- [18. Default Config](#18-default-config)
- [19. Implementation Order](#19-implementation-order)
- [20. Acceptance Criteria](#20-acceptance-criteria)
  - [20.1 Data](#201-data)
  - [20.2 Numerics](#202-numerics)
  - [20.3 Shape contract](#203-shape-contract)
  - [20.4 Assignment](#204-assignment)
  - [20.5 Logging](#205-logging)
  - [20.6 Reindexer](#206-reindexer)
- [21. Explicitly Deferred to Later Revisions](#21-explicitly-deferred-to-later-revisions)
- [22. One-Paragraph Execution Summary](#22-one-paragraph-execution-summary)

---

## 0. Document Status

This document is the implementation authority for the revised project.

If this document conflicts with:

the older Design Plan v1_1.pdf, or

the older canonical-indexing version of the model,

this document wins.

This document is written for implementation, not for high-level presentation.

---

## 1. Objective

Build a label-free learned initial qubit mapper for a fixed 27-qubit IBM BackendV2 target.

The system must:

accept a quantum circuit with K <= 27 logical qubits,

extract a fixed-size circuit representation,

extract fixed native hardware tensors from the backend,

learn a soft internal reordering of logical and hardware indices,

run a U-Net-style mapper in the reordered latent frame,

decode logits back to the native hardware frame,

produce a one-to-one initial mapping using Sinkhorn during training and Hungarian during inference,

train against a proxy objective correlated with:

PST,

SWAP overhead,

depth increase.

The one-to-one assignment requirement and the Sinkhorn/Hungarian contract come from the prior mapper design; the native-frame decode and two-pass optimization come from the reindexer design.

---

## 2. Non-Goals

This revision does not do the following:

full routing / explicit SWAP insertion,

diffusion or timestep conditioning,

backend-specific supervised imitation of GraphQMap labels,

multi-backend training in a single run,

direction-aware hardware tokenization beyond the symmetrized cost extraction defined below.

No diffusion/timestep embedding is used in the prior mapper design, and that remains true here.

---

## 3. Locked Design Decisions

### 3.1 Canonical indexing is removed

Do not implement or use a canonical indexer.
Do not compute any permutation p.
Do not convert canonical indices back to native IDs.

Use only:

native logical tensors,

native hardware tensors,

learned soft reindexing,

native-frame decode.

This follows the reindexer replacement design.

### 3.2 Mapper conditioning style

Use Imagen-style circuit-to-hardware conditioning:

the circuit tensor is the main U-Net spatial input,

hardware is injected as cross-attention conditioning,

queries come from U-Net feature maps,

keys and values come from hardware-only tokens.

Do not use logical tokens in the mapper.

### 3.3 U-Net input

The U-Net input is only the circuit matrix A.

Do not use the old 5-channel X = [W, A, c2, C1_col, Mmask] input contract from the previous design as the mapper input.
That older grid remains historically relevant, but this revision replaces it with a single-channel circuit image + hardware attention tokens. The old plan explicitly used a fixed 5-channel grid and logical+physical token concatenation; this spec does not.

### 3.4 Hardware token definition

For each physical qubit j, define the raw hardware token as:

x_hw[j] = concat(B[j, :], c2[j, :], c1[j])

with shape:

B[j, :]   -> (27,)
c2[j, :]  -> (27,)
c1[j]     -> (1,)
x_hw[j]   -> (55,)

This is the exact mapper hardware-token contract for this revision.

### 3.5 Distance matrix role

Keep D in:

loss computation,

routing proxies,

reindexer hardware branch features,

but do not include D[j, :] in the mapper hardware token.

### 3.6 No infinities anywhere

Never use inf, -inf, or large artificial sentinels in:

B,

c1,

c2,

D,

attention inputs,

loss computation,

routing proxy computation.

Finite-only distance handling is explicitly required in the reindexer plan.

### 3.7 Batch size

Use:

batch_size = 1

as the default and recommended setting. The reindexer plan lists batch size 1 as the recommended default.

### 3.8 Auxiliary losses

Implement reindexer auxiliary losses in code, but for the initial stable run set:

alpha_loc = 0.0
beta_cons = 0.0

Then turn them on only in later runs.

The reindexer plan defines L_reindex = L_task + alpha * L_loc + beta * L_cons, but also warns not to start with large coefficients.

---

## 4. Notation

Use the following notation consistently in code and docs.

### 4.1 Dimensions

n = 27     # fixed physical qubit count
K <= 27    # logical qubit count for a given circuit
Bsz = 1    # batch size

### 4.2 Circuit tensors

A      : (27, 27)  # circuit interaction matrix
m      : (27,)     # logical-valid mask

### 4.3 Hardware tensors

B      : (27, 27)  # binary hardware adjacency / connectivity
c1     : (27,)     # per-physical-qubit cost
c2     : (27, 27)  # per-physical-edge cost
D      : (27, 27)  # shortest-path distance on B

### 4.4 Reindexer outputs

R_L    : (27, 27)  # logical soft permutation
R_H    : (27, 27)  # hardware soft permutation

Orientation convention:

rows    = latent slots
columns = original IDs

So:

R_L[t, u] = probability that original logical node u goes to latent slot t
R_H[t, j] = probability that original hardware node j goes to latent slot t

This orientation is explicitly defined in the reindexer plan.

### 4.5 Reordered tensors

A*   = R_L A R_L^T        # circuit matrix in latent logical frame
m*   = R_L m
B*   = R_H B R_H^T
c1*  = R_H c1
c2*  = R_H c2 R_H^T
D*   = R_H D R_H^T

### 4.6 Mapper outputs

S*      : (27, 27)  # latent-frame mapping logits
S_nat   : (27, 27)  # decoded native-frame logits
P_map   : (27, 27)  # soft assignment in training
M_map   : (27, 27)  # hard assignment in inference

### 4.7 Decode rule

S_nat = R_L^T S* R_H

and in inference, replace R_L, R_H with hard permutations R_L_hat, R_H_hat. This is the exact reconstruction rule in the reindexer plan.

---

## 5. Dataset Protocol

The MQT generator script you uploaded creates circuits across many algorithm types and qubit counts from 2 to 127, with multiple variants for VQE and QAOA. That is why hard filtering to K <= 27 is mandatory for this project.

### 5.1 Phase A — stability smoke test

Train briefly on the smoke recipe built from the full manifest family:

- `data/manifests/full/recipes/smoke_mqt/`

Purpose:

- verify preprocessing,
- verify finite-value handling,
- verify no exploding loss,
- verify the two-pass training loop works.

Conceptually this is still “MQT only”; operationally it is now selected through the reusable recipe manifest.

### 5.2 Phase B — main comparison training

Train on the `train_side` sources in `circuit_v2`:

- `queko`
- `mlqd`
- `mqt_bench`

Use source-balanced sampling across the three train-side sources.

Operationally, the full manifest builder may emit multiple train-side recipes so experiments can reuse the same primitive source splits while changing source combinations.

### 5.3 Final evaluation

Do not train on the held-out or benchmark-only sources:

- `qasmbench`
- `revlib`
- `benchmarks`

Use them only for final transfer/generalization evaluation.

For the official full run, final evaluation includes both:

- the existing proxy metrics in the native frame, and
- the routed downstream “real” metrics computed after transpilation/routing.

### 5.4 Hard inclusion rules

A circuit is included iff all are true:

- `2 <= K <= 27`
- contains at least one 2Q gate, unless a manifest builder explicitly enables `--allow-no-2q`
- parses successfully
- hardware preprocessing succeeds
- circuit interaction matrix is not all-zero off-diagonal

### 5.5 Keep disconnected logical graphs

Do not remove circuits just because the logical interaction graph is disconnected.
Only remove degenerate cases with no useful 2Q structure.

### 5.6 Labels

Do not use GraphQMap labels as direct training targets.
This method is label-free and uses proxy-loss training only.

### 5.7 Train/val splitting

Use deterministic per-source splitting inside the full manifest builder.

The authoritative concrete split definition is the manifest family stored under:

- `data/manifests/full/source_manifests/<source>/`
- `data/manifests/full/recipes/<recipe_name>/`

Rules:

- primitive source manifests are split independently per source using a deterministic source-specific seed,
- combined recipe manifests are constructed by concatenating already-split primitive source manifests,
- once manifests are generated for a run family, they must be reused for reproducibility,
- no identical circuit file may appear across `train`/`val`/`test`.

### 5.8 Source balancing

In the main run, source balancing applies to the three `train_side` sources:

- `p(source = queko) = 1/3`
- `p(source = mlqd) = 1/3`
- `p(source = mqt_bench) = 1/3`

Then sample a circuit uniformly from the chosen source.

Do not just concatenate the train-side datasets and shuffle.


---

## 6. Backend Extraction

This module converts an IBM BackendV2 target into the native hardware tensors.

### 6.1 Input

backend: IBM BackendV2

### 6.2 Output

B_raw   : (27, 27)
c1_raw  : (27,)
c2_raw  : (27, 27)
D_raw   : (27, 27)

### 6.3 Hardware adjacency B

Construct a symmetric binary adjacency:

B[i, j] = 1 if at least one usable 2Q direction exists between i and j
B[i, j] = 0 otherwise
B[i, i] = 0

Then enforce symmetry:

B = max(B, B^T)

Use B only as topology/validity.
Do not encode hardware quality into B.

### 6.4 Per-qubit cost c1

Define:

e_ro[j] = readout error for qubit j if available, else 0.0

e_1q[j] = mean error over supported 1Q basis gates on qubit j; if unavailable, 0.0

Then:

c1_raw[j] = 0.5 * e_ro[j] + 0.5 * e_1q[j]

This yields a scalar per physical qubit.

### 6.5 Per-edge cost c2

For each unordered pair (i, j):

collect valid directional 2Q gate errors:

e_ij if (i -> j) exists,

e_ji if (j -> i) exists.

Then define:

if no valid direction exists:
    c2_raw[i, j] = 0
elif only one valid direction exists:
    c2_raw[i, j] = that direction's error
else:
    c2_raw[i, j] = min(e_ij, e_ji)

Then enforce symmetry:

c2_raw[j, i] = c2_raw[i, j]
c2_raw[i, i] = 0

This intentionally treats c2 as a symmetric edge badness for initial mapping.

### 6.6 Distance matrix D

Build the shortest-path distance matrix on the graph defined by B.

Use:

Floyd-Warshall,

or BFS from every node.

Define:

D_raw[i, i] = 0
D_raw[i, j] = shortest hop count if reachable
D_raw[i, j] = 28 if unreachable

because n + 1 = 28.

### 6.7 Backend extractor invariants

After extraction, assert:

B.shape   == (27, 27)
c1.shape  == (27,)
c2.shape  == (27, 27)
D.shape   == (27, 27)

B is symmetric binary
c2 is symmetric finite
D is symmetric finite
diag(B) == 0
diag(c2) == 0
diag(D) == 0

---

## 7. Circuit Featurization

This module converts a circuit into (A, m).

### 7.1 Logical qubit count

Let:

K = number of logical qubits in the circuit

Reject the circuit if:

K > 27

### 7.2 Logical mask m

Define:

m[u] = 1 for u < K
m[u] = 0 for u >= K

### 7.3 Off-diagonal entries of A

For u != v, define:

A_raw[u, v] = log(1 + sum_{g in G_2Q(u,v)} w_2q(type(g)))

where:

G_2Q(u,v) = all 2Q gates involving logical qubits u and v

w_2q(type) = gate-type weight

### 7.4 Default 2Q gate-type weights

For the first stable revision:

w_2q(type) = 1.0 for all 2Q gate types

So off-diagonals are effectively log-scaled 2Q interaction counts.

Do not differentiate 2Q gate types in v1.1 unless you intentionally run an ablation.

### 7.5 Diagonal entries of A

For logical qubit u, define:

N1Q(u)       = number of 1Q gates on u
N2Q_part(u)  = number of 2Q gates in which u participates

Then:

A_raw[u, u] = log(1 + alpha_diag * N1Q(u) + beta_diag * N2Q_part(u))

Use:

alpha_diag = 0.25
beta_diag  = 1.00

These defaults intentionally give more weight to 2Q participation than to 1Q load.

### 7.6 Padding

Initialize:

A_raw = zeros(27, 27)

Fill only the active K x K logical block.
All padded rows/cols remain zero.

### 7.7 Symmetry

After constructing off-diagonal entries, enforce:

A_raw = 0.5 * (A_raw + A_raw.T)

### 7.8 Degenerate circuit rejection

Reject the circuit if:

sum_{u != v} A_raw[u, v] == 0

This removes no-2Q or effectively zero-interaction circuits.

---

## 8. Normalization and Finite-Value Policy

### 8.1 Global epsilon

Use:

eps = 1e-8

for normalization denominators.

### 8.2 Circuit matrix A

Normalize by max value:

A = A_raw / (max(A_raw) + eps)

If max(A_raw) == 0, leave A as zeros.

### 8.3 Hardware adjacency B

Do not normalize.

Keep as:

B in {0, 1}

### 8.4 Per-qubit cost c1

Normalize with z-score across the 27 qubits:

mu_c1    = mean(c1_raw)
sigma_c1 = std(c1_raw)
c1       = (c1_raw - mu_c1) / (sigma_c1 + eps)

### 8.5 Per-edge cost c2

Important: compute statistics over valid edges only.

Define:

E_valid = {(i, j) | B[i, j] == 1}

Compute:

mu_c2    = mean(c2_raw[i, j] for (i, j) in E_valid)
sigma_c2 = std(c2_raw[i, j] for (i, j) in E_valid)

Then:

if B[i, j] == 1:
    c2[i, j] = (c2_raw[i, j] - mu_c2) / (sigma_c2 + eps)
else:
    c2[i, j] = 0

This preserves the non-edge rule:

c2[i, j] = 0 when B[i, j] = 0

and avoids having non-edge zeros corrupt the normalization statistics.

### 8.6 Distance matrix D

Normalize by max finite entry:

D = D_raw / (max(D_raw) + eps)

### 8.7 Summary of finite rules

Use exactly:

B[i, j] in {0, 1}
c2[i, j] = 0 when B[i, j] = 0
D[i, j]  = 28 when unreachable before normalization
no inf anywhere

---

## 9. Reindexer

The reindexer is a separate module that learns soft latent reorderings of:

logical indices,

hardware indices.

It does not solve the mapping itself. This role is explicitly stated in the reindexer design.

### 9.1 File

Create:

mqm/networks/reindexer.py

### 9.2 Components

Implement:

LogSinkhorn
LogicalReindexBranch
HardwareReindexBranch
SoftPermutationReindexer

This file structure follows the reindexer plan.

### 9.3 Reindexer hidden dimension

Use:

d_r = 128

The reindexer plan recommends d_r = 128.

### 9.4 Logical branch input features

For each logical slot u, define:

feat_L[u] = [
    log(1 + sum_v A[u, v]),
    top1_offdiag(A[u, :]),
    top2_offdiag(A[u, :]),
    m[u]
]

Rules:

top1_offdiag and top2_offdiag ignore the diagonal.

if fewer than 2 nonzero off-diagonal values exist, use 0.

### 9.5 Logical branch MLP

Use:

LayerNorm(4)
Linear(4, d_r)
ReLU
Dropout(0.1)
Linear(d_r, d_r)
LayerNorm(d_r)

### 9.6 Logical slot prototypes

Learn:

E_L : (27, d_r)

### 9.7 Logical score matrix

Compute:

H_L = LogicalMLP(feat_L)                 # (27, d_r)
G_L = H_L @ E_L.T                        # (27, 27)
G_L_tilde[t, u] = G_L[u, t]              # transpose orientation
R_L = LogSinkhorn(G_L_tilde / tau_r)

### 9.8 Hardware branch input features

For each hardware qubit j, define:

deg(j)        = sum_k B[j, k]
mean_c2(j)    = mean(c2[k] over valid neighbors), else 0
min_c2(j)     = min(c2[k] over valid neighbors), else 0
mean_D(j)     = mean(D[j, :])

Then:

feat_H[j] = [
    c1[j],
    deg(j),
    mean_c2(j),
    min_c2(j),
    mean_D(j)
]

### 9.9 Hardware branch MLP

Use:

LayerNorm(5)
Linear(5, d_r)
ReLU
Dropout(0.1)
Linear(d_r, d_r)
LayerNorm(d_r)

### 9.10 Hardware slot prototypes

Learn:

E_H : (27, d_r)

### 9.11 Hardware score matrix

Compute:

H_H = HardwareMLP(feat_H)               # (27, d_r)
G_H = H_H @ E_H.T                       # (27, 27)
G_H_tilde[t, j] = G_H[j, t]
R_H = LogSinkhorn(G_H_tilde / tau_r)

### 9.12 Reindexer Sinkhorn

Use log-domain Sinkhorn, not naive exp-Sinkhorn.

Defaults from the reindexer plan:

tau_r schedule: 1.0 -> 0.15
T_r = 20 iterations

### 9.13 Reordered tensors

Given (A, m, B, c1, c2, D) and (R_L, R_H):

A*   = R_L @ A  @ R_L.T
m*   = R_L @ m
B*   = R_H @ B  @ R_H.T
c1*  = R_H @ c1
c2*  = R_H @ c2 @ R_H.T
D*   = R_H @ D  @ R_H.T

Note:

during training, these reordered matrices become soft/dense,

this is expected and acceptable in the differentiable relaxation. The reindexer plan explicitly notes that reordered tensors become soft/dense during training.

---

## 10. Mapper

### 10.1 File

Use the existing mapper file, e.g.:

mqm/networks/unet_mapper.py

Modify it rather than redesigning the entire backbone.

### 10.2 Input contract

The mapper must accept:

A*        : (Bsz, 1, 27, 27)   # U-Net spatial input
T_hw*     : (Bsz, 27, d_tok)   # hardware tokens

### 10.3 Hardware token encoder

For reordered hardware tensors, build hardware tokens in latent hardware order:

x_hw*[j] = concat(B*[j, :], c2*[j, :], c1*[j])

with shape (55,).

Then embed with:

LayerNorm(55)
Linear(55, d_tok)
ReLU
Dropout(0.1)
Linear(d_tok, d_tok)
LayerNorm(d_tok)

Use:

d_tok = 128

### 10.4 No logical tokens

Do not build logical tokens for the mapper.

### 10.5 U-Net backbone

Reuse the current U-Net macro-architecture in the codebase.
Do not redesign the down/up channel schedule in this revision.

Required guarantees:

spatial input size: 27 x 27

spatial output size: 27 x 27

final logits shape: (Bsz, 1, 27, 27)

Interpretation:

S*[u, j] = score for mapping latent logical slot u to latent hardware slot j

### 10.6 Cross-attention placement

Inject cross-attention at exactly these 3 locations:

last down block,

bottleneck,

first up block.

This preserves the old Tier-2 cross-attention placement rule.

### 10.7 Cross-attention rule

At each injection point, with feature map F of shape (Bsz, C, H, W):

F_flat = reshape(F) to (Bsz, H*W, C)

Q = F_flat @ W_Q
K = T_hw* @ W_K
V = T_hw* @ W_V

Attn(Q, K, V) = softmax(Q K^T / sqrt(d_a)) V
DeltaF = reshape(AttnOut @ W_O) back to (Bsz, C, H, W)

F <- F + alpha_attn * DeltaF

Use:

multi-head attention,

alpha_attn as a learnable scalar per block,

initialize alpha_attn = 0.

The prior mapper design explicitly used residual cross-attention with a learned gate initialized to zero.

### 10.8 No timestep embedding

Do not add timestep embedding.

---

## 11. Decode and Assignment

### 11.1 Decode to native frame

After the mapper produces S*, decode:

S_nat = R_L.T @ S* @ R_H

### 11.2 Training assignment

Use Sinkhorn on decoded logits:

P_map = Sinkhorn(S_nat / tau_m)

### 11.3 Inference assignment

Use Hungarian on decoded logits:

M_map = Hungarian(S_nat)

This is exactly the native-frame assignment pattern specified in the reindexer plan.

### 11.4 Mapping output

Return final mapping as:

logical_u -> native_physical_j

for u = 0 .. K-1.

There is no canonical-to-native conversion table anymore.

### 11.5 Mapper Sinkhorn default

If your current code already has a stabilized mapper Sinkhorn setting, keep it.

Otherwise use:

tau_m = 0.10
T_m   = 20

as the default fallback.

---

## 12. Task Loss

The older mapper design used:

L = lambda_P * L_PST + lambda_S * L_swap + lambda_D * L_depth

and the reindexer plan keeps the task loss semantically unchanged, but computes it in the native frame after decode.

### 12.1 Active logical set

Define:

U_active = {u | m[u] = 1}

### 12.2 Off-diagonal circuit mass

Define:

A_off[u, v] = A[u, v] for u != v
A_off[u, u] = 0
mass_2q = max(sum_{u,v} A_off[u,v], 1e-6)
mass_1q = max(sum_u m[u], 1e-6)

### 12.3 PST 1Q term

Define:

L_PST_1Q_num =
    sum_u m[u] * sum_j P_map[u, j] * c1[j]

L_PST_1Q =
    L_PST_1Q_num / mass_1q

### 12.4 PST 2Q term

Define the usable edge-cost matrix:

C2_use = B * c2

Then:

L_PST_2Q_num =
    sum_{u,v} A_off[u,v] *
    sum_{i,j} P_map[u,i] * P_map[v,j] * C2_use[i,j]

L_PST_2Q =
    L_PST_2Q_num / mass_2q

### 12.5 PST total

Define:

L_PST_total = L_PST_1Q + L_PST_2Q

### 12.6 SWAP proxy

Define the expected physical distance for a logical pair:

E_D(u,v) =
    sum_{i,j} P_map[u,i] * P_map[v,j] * D[i,j]

Then:

L_swap_num =
    sum_{u,v} A_off[u,v] * E_D(u,v)

L_swap =
    L_swap_num / mass_2q

### 12.7 Depth proxy

Define:

L_depth = kappa_depth * L_swap

Use:

kappa_depth = 1.0

for v1.1.

### 12.8 Total task loss

Use:

lambda_P = 1.0
lambda_S = 1.0
lambda_D = 0.25

Then:

L_task = lambda_P * L_PST_total
       + lambda_S * L_swap
       + lambda_D * L_depth

### 12.9 Dummy logical rows

Dummy rows automatically contribute zero because:

m[u] = 0,

padded circuit rows/cols are zero.

This matches the prior mapper design.

---

## 13. Reindexer Auxiliary Losses

### 13.1 Locality loss

Compute on reordered tensors.

Logical locality
W_mass = max(sum_{u,v} A*[u,v], 1e-6)

L_loc_log =
    (1 / W_mass) *
    sum_{u,v} A*[u,v] * ((u - v) / 26)^2
Hardware locality
B_mass = max(sum_{i,j} B*[i,j], 1e-6)

L_loc_hw =
    (1 / B_mass) *
    sum_{i,j} B*[i,j] * ((i - j) / 26)^2
Total locality
L_loc = 0.5 * (L_loc_log + L_loc_hw)

This follows the locality structure defined in the reindexer plan.

### 13.2 Consistency loss

Sample random relabelings:

Pi_L : logical permutation matrix
Pi_H : hardware permutation matrix

Construct relabeled inputs:

A'   = Pi_L A Pi_L^T
m'   = Pi_L m
B'   = Pi_H B Pi_H^T
c1'  = Pi_H c1
c2'  = Pi_H c2 Pi_H^T
D'   = Pi_H D Pi_H^T

Run the reindexer again to obtain reordered tensors:

A*', m*', B*', c1*', c2*', D*'

Then compare reordered forms, not the permutations themselves:

L_cons =
    MSE(A*,  A*')
  + MSE(m*,  m*')
  + MSE(B*,  B*')
  + MSE(c1*, c1*')
  + MSE(c2*, c2*')
  + MSE(D*,  D*')

This is the consistency principle defined in the reindexer plan.

### 13.3 Reindexer objective

Implement:

L_reindex = L_task + alpha_loc * L_loc + beta_cons * L_cons

### 13.4 Default staged coefficients

For the initial stable implementation config:

alpha_loc = 0.0
beta_cons = 0.0

Later configs:

stage_1: alpha_loc = 0.02, beta_cons = 0.00
stage_2: alpha_loc = 0.05, beta_cons = 0.10

---

## 14. Training Loop

The reindexer plan explicitly requires two optimizers and two forward passes per batch.

### 14.1 Files

Update or create:

training/train_label_free_v11.py
training/train_utils.py

### 14.2 Optimizers

Use:

optimizer_mapper    = AdamW(mapper_params,    lr=1e-4, weight_decay=1e-4)
optimizer_reindexer = AdamW(reindexer_params, lr=5e-5, weight_decay=1e-4)

The learning-rate split follows the reindexer defaults.

### 14.3 Gradient clipping

Use:

clip_grad_norm_(mapper_params,    1.0)
clip_grad_norm_(reindexer_params, 1.0)

The reindexer defaults clip both at 1.0.

### 14.4 Pass A — mapper update

For one batch:

read native tensors (A, m, B, c1, c2, D)

compute R_L, R_H

detach them:

R_L_det = R_L.detach()
R_H_det = R_H.detach()

build reordered tensors using detached permutations

build reordered hardware tokens

run mapper to get S*

decode S_nat

compute P_map

compute L_task

backward only through mapper

clip mapper gradients

optimizer step for mapper only

### 14.5 Pass B — reindexer update

recompute R_L, R_H without detach

rebuild reordered tensors

rebuild reordered hardware tokens

run mapper with mapper parameters frozen

decode S_nat

compute P_map

compute L_task

compute L_loc, L_cons

compute L_reindex

backward only through reindexer

clip reindexer gradients

optimizer step for reindexer only

This follows the exact optimization separation prescribed in the reindexer plan.

### 14.6 Temperature scheduling

Use:

tau_r:
    start = 1.0
    end   = 0.15
    schedule = cosine or linear anneal over total training steps

Mapper Sinkhorn temperature:

tau_m = fixed

Do not anneal tau_m in v1.1 unless you already have a stable implementation doing so.

### 14.7 Batch size

Use:

batch_size = 1

Always.

---

## 15. Inference Protocol

The reindexer plan specifies hard reindexing at inference.

### 15.1 Steps

For one circuit:

- extract native `(A, m, B, c1, c2, D)`
- compute reindexer logits `G_L_tilde, G_H_tilde`
- compute hard permutations:
  - `R_L_hat = Hungarian(G_L_tilde)`
  - `R_H_hat = Hungarian(G_H_tilde)`
- reorder tensors with hard permutations
- build hardware tokens from hard-reordered hardware tensors
- run mapper to get `S*`
- decode:
  - `S_nat = R_L_hat.T @ S* @ R_H_hat`
- run Hungarian on `S_nat`:
  - `M_map = Hungarian(S_nat)`
- return mapping for rows `u < K`

### 15.2 Output format

Return:

```python
{
  "mapping": {logical_u: native_physical_j, ...},
  "M_map": M_map,
  "S_nat": S_nat,
  "R_L_hat": R_L_hat,
  "R_H_hat": R_H_hat
}
```

### 15.3 Official full-run routed final evaluation

The system remains an initial mapper. Routing is added only as a **final evaluation extension**.

When routed evaluation is enabled for the full run, the official per-sample sequence is:

1. run the current hard-reindexing inference path exactly as above,
2. convert the final hard mapping into a Qiskit `initial_layout`,
3. reload the original circuit from `qasm_relpath`,
4. transpile / route against the BackendV2 target using the model-derived initial layout,
5. capture routing-stage SWAP information before final translation destroys literal `swap` gates,
6. compute routed metrics from the final executable circuit,
7. write those real routed metrics beside the existing proxy metrics.

This routed-eval path must **not** change the training objective, mapper backbone, or reindexer semantics.

### 15.4 Official routed metrics for the full run

When routed evaluation is enabled, the required routed metrics are:

- `real_pst_gate_readout`
- `routing_compile_time_s`
- `routing_total_eval_time_s`
- `swap_inserted_count`
- `original_2q_count`
- `routed_2q_count`
- `added_2q_ops`
- `original_depth`
- `routed_depth`
- `depth_increase_abs`
- `depth_increase_ratio`

The locked official choices are:

- transpilation setting: `optimization_level = 0`
- real PST includes both gate error and readout error
- SWAP overhead is reported both as exact inserted SWAP count and added 2Q op count

### 15.5 Ownership of routed evaluation logic

Routed evaluation belongs in:

- `src/kmw/evaluation/evaluate.py` for the real transpilation/metric logic,
- `src/kmw/cli/main.py` for config/flag wiring only.

The dataset remains native-only and manifest-driven.
It must **not** perform routing itself.

---

## 16. Validation and Fail-Safes


This is mandatory, not optional.

### 16.1 Tensor checks

After each major tensor is created, assert:

correct shape,

finite values only,

no NaN,

no Inf.

Apply to:

A, m, B, c1, c2, D
R_L, R_H
A*, m*, B*, c1*, c2*, D*
S*, S_nat
P_map

### 16.2 Loss checks

Log and validate each of:

L_PST_1Q
L_PST_2Q
L_PST_total
L_swap
L_depth
L_task
L_loc_log
L_loc_hw
L_loc
L_cons
L_reindex

### 16.3 Sinkhorn sanity checks

For each Sinkhorn output:

row sums,

column sums,

min,

max,

entropy/sharpness.

Apply to:

R_L
R_H
P_map

### 16.4 Gradient checks

After backward, log:

mapper grad norm,

reindexer grad norm,

whether grad is finite.

### 16.5 Hard-fail conditions

Abort the step immediately if any of the following occurs:

any major tensor contains NaN/Inf,

any loss term is non-finite,

any gradient norm is non-finite,

any Sinkhorn output is non-finite.

### 16.6 Warning conditions

Warn but continue if:

max abs of a tensor exceeds 1e6,

Sinkhorn row/column sums deviate by more than 1e-2 from expected.

### 16.7 Failure report

On failure, log:

circuit_id
dataset_source
K
global_step
epoch
pass_type
offending_tensor_or_loss
min/max/mean
tau_r
tau_m
mapper_lr
reindexer_lr

---

## 17. File and Module Layout

The legacy-style module names below are retained as behavioral reference names from the earlier design documents.

For the current KMW implementation, use the `src/kmw/` package layout defined in Section 2 as the authoritative final location. Treat the section below as a responsibility mapping, not as a requirement to recreate the old `mqm/` or `training/` directory layout.

Recommended file structure:

mqm/
  networks/
    unet_mapper.py
    reindexer.py
    hardware_token_encoder.py
    assignment.py

training/
  backend_extractor.py
  circuit_featurizer.py
  dataset_graphqmap.py
  loss_task.py
  loss_reindexer.py
  train_label_free_v11.py
  evaluate.py
  samplers.py
  checks.py
  config_v11.py

### 17.1 backend_extractor.py

Responsibilities:

extract B_raw, c1_raw, c2_raw

compute D_raw

normalize to B, c1, c2, D

### 17.2 circuit_featurizer.py

Responsibilities:

parse circuit

compute A_raw

compute m

normalize to A

### 17.3 dataset_graphqmap.py

Responsibilities:

load QASM

parse circuit

filter by K <= 27

reject degenerate circuits

return native tensors only:

A, m, B, c1, c2, D, metadata

This native-only dataset output is required by the reindexer design.

### 17.4 reindexer.py

Responsibilities:

implement reindexer branches

implement log-Sinkhorn

produce R_L, R_H

optionally expose hard permutation inference helper

### 17.5 hardware_token_encoder.py

Responsibilities:

build x_hw[j]

embed to T_hw

### 17.6 unet_mapper.py

Responsibilities:

accept A* and T_hw*

run cross-attention at the 3 locked points

output S*

### 17.7 assignment.py

Responsibilities:

Sinkhorn for mapper training

Hungarian wrapper for inference

### 17.8 loss_task.py

Responsibilities:

L_PST_1Q

L_PST_2Q

L_PST_total

L_swap

L_depth

L_task

### 17.9 loss_reindexer.py

Responsibilities:

L_loc_log

L_loc_hw

L_loc

L_cons

L_reindex

### 17.10 samplers.py

Responsibilities:

MQT-only smoke sampler

balanced 3-source sampler for main training

### 17.11 checks.py

Responsibilities:

tensor sanity checks

Sinkhorn checks

gradient checks

structured crash reports

---

## 18. Default Config

Use this as the initial `config_v11.py` or YAML equivalent.

```yaml
hardware:
  n_qubits: 27
  unreachable_distance_fill: 28

normalization:
  eps: 1.0e-8

dataset:
  dataset_version: "circuit_v2"
  manifest_root: "data/manifests/full"
  smoke_recipe: "smoke_mqt"
  phase_b_train_side_sources: ["queko", "mlqd", "mqt_bench"]
  heldout_sources: ["qasmbench", "revlib"]
  benchmark_sources: ["benchmarks"]
  min_qubits: 2
  max_qubits: 27
  require_two_qubit_gate: true
  drop_degenerate_zero_interaction: true
  balance_phase_b_sources: true

reindexer:
  hidden_dim: 128
  sinkhorn_iters: 20
  tau_start: 1.0
  tau_end: 0.15
  dropout: 0.1

mapper:
  token_dim: 128
  attn_heads: 4
  attn_gate_init: 0.0
  tau_map: 0.10
  sinkhorn_iters: 20

loss:
  alpha_diag: 0.25
  beta_diag: 1.0
  lambda_p: 1.0
  lambda_s: 1.0
  lambda_d: 0.25
  kappa_depth: 1.0

auxiliary_losses:
  alpha_loc: 0.0
  beta_cons: 0.0

optim:
  batch_size: 1
  mapper_lr: 1.0e-4
  reindexer_lr: 5.0e-5
  weight_decay: 1.0e-4
  grad_clip: 1.0

evaluation:
  route_final_eval: false
  routing_method: "sabre"
  transpile_optimization_level: 0
  seed_transpiler: null
  include_readout_in_pst: true
  save_routed_qasm_dir: null
  save_routed_qpy_dir: null

training:
  hard_fail_on_nonfinite: true
  warn_on_large_abs_tensor: 1.0e6
  warn_on_sinkhorn_sum_error: 1.0e-2
```

---

## 19. Implementation Order

Implement in this exact order.

Phase 1 — data and preprocessing

- backend extractor
- circuit featurizer
- dataset filtering and metadata
- normalization
- sanity-check scripts
- full manifest generation for `circuit_v2`
- manifest/recipe selection for smoke vs main vs held-out eval

Phase 2 — core reindexer

- log-Sinkhorn
- logical branch
- hardware branch
- reordered tensor construction
- hard-permutation inference helper

Phase 3 — mapper integration

- hardware token encoder
- mapper input contract change
- remove logical tokens from mapper
- cross-attention K/V from hardware tokens only
- output `S*`

Phase 4 — assignment and losses

- decode to native frame
- Sinkhorn/Hungarian mapping head
- task loss terms
- logging
- failure checks

Phase 5 — training loop

- two optimizers
- mapper pass
- reindexer pass
- smoke-run config
- checkpointing

Phase 6 — evaluation

- inference with hard reindexing
- dataset-wise proxy evaluation
- full-run routed final evaluation
- per-circuit CSV and summary JSON extension for routed metrics

---

## 20. Acceptance Criteria

The implementation is acceptable only if all are true.

### 20.1 Data

- all dataset samples returned by the loader satisfy `2 <= K <= 27`
- no sample returned to training has zero 2Q interaction mass
- full-manifest rows preserve the required compatibility fields, including `qasm_relpath`, `dataset_version`, and `source_role`

### 20.2 Numerics

- no `inf` or `nan` in any major tensor
- smoke run completes without exploding loss
- both training passes run end-to-end

### 20.3 Shape contract

- `A` input to mapper is `(Bsz, 1, 27, 27)`
- mapper token input is `(Bsz, 27, 128)`
- mapper output is `(Bsz, 1, 27, 27)`

### 20.4 Assignment

- training uses Sinkhorn on `S_nat`
- inference uses Hungarian on `S_nat`
- returned mapping uses native hardware IDs only

### 20.5 Logging

- all required loss components and diagnostics are recorded per step
- proxy-only eval still works when routed evaluation is disabled
- routed final eval writes the required routed fields when enabled

### 20.6 Reindexer

- `R_L` and `R_H` are approximately doubly stochastic during training
- hard reindexing works in inference

### 20.7 Full-run routed evaluation

For successful routed rows:

- `0 <= real_pst_gate_readout <= 1`
- `swap_inserted_count >= 0`
- `routing_compile_time_s >= 0`
- `original_depth >= 0`
- `routed_depth >= 0`

And architecturally:

- training remains proxy-based and unchanged,
- routed evaluation uses the model-produced hard mapping as the transpiler initial layout,
- official full-run routed evaluation uses `optimization_level = 0`,
- exact inserted SWAP count is captured from the routing stage, not inferred only from the final translated circuit.

---

## 21. Explicitly Deferred to Later Revisions


These are intentionally not part of v1.1 implementation:

richer gate-type weighting for A,

extra circuit channels beyond A,

including D in mapper hardware tokens,

full activation of reindexer auxiliary losses in the first stable run,

direction-aware c2 modeling beyond the symmetric reduction,

using GraphQMap labels as supervision,

multi-backend or variable-hardware training.

---

## 22. One-Paragraph Execution Summary

Implement a native-frame label-free mapper for a fixed 27-qubit IBM backend. Extract native hardware tensors B, c1, c2, D, featurize each circuit into a fixed 27 x 27 circuit matrix A plus mask m, learn soft logical and hardware reorderings with a two-branch reindexer, run the mapper on reordered A* with hardware-only cross-attention tokens built from (B*, c2*, c1*), decode logits back to native frame, solve the training assignment with Sinkhorn and inference assignment with Hungarian, optimize PST/SWAP/depth proxy loss in the native frame, and train in two passes with strict finite-value checks and batch size 1. This preserves the original mapping-head/task-loss structure while replacing the old canonical-indexing pipeline with soft learned reindexing.


---

## 4. Clarification Patch (v1.1.1)

The following clarification patch takes precedence over conflicting wording in v1.1, except where the v1.2 locked decisions above resolve the remaining ambiguity.

# Design Plan v1.1.1 — Clarification Patch

## 0. Document Status

This document is a clarification revision of **Design Plan v1.1**.

**v1.1.1 introduces no semantic design changes.**  
It only resolves wording ambiguities and implementation-loose sections in v1.1 so that different implementers produce the same system.

If any section below conflicts with the wording of v1.1, the **v1.1.1 wording takes precedence**, but the intended model/design remains the same as v1.1.

---

## 1. Clarification: Reference U-Net Backbone

### Replace Section 10.5 with:

### 10.5 U-Net backbone

The mapper must preserve the **same macro-architecture and channel schedule** as the current reference U-Net used for this project.

This means:

- same number of down blocks,
- same number of up blocks,
- same bottleneck structure,
- same per-stage channel widths,
- same normalization type,
- same residual/non-residual block structure,
- same final logits head shape.

The phrase “reuse the current U-Net macro-architecture in the codebase” means **copy the exact architecture behavior**, not “implement a similar U-Net.”

Required guarantees remain:

- spatial input size: `27 x 27`
- spatial output size: `27 x 27`
- final logits shape: `(Bsz, 1, 27, 27)`

Interpretation:

`S*[u, j] = score for mapping latent logical slot u to latent hardware slot j`

### Implementation rule

When the KMW implementation is created under its own folder structure, the implementer must do one of the following:

1. directly reuse the reference U-Net implementation, or
2. reimplement it exactly and record the source file + commit/hash used as the reference.

This is to prevent silent divergence caused by “roughly similar” U-Net implementations.

---

## 2. Clarification: Raw vs Normalized Tensor Naming

### Replace Section 6.7 with:

### 6.7 Backend extractor invariants

Two sets of invariants must be checked.

#### 6.7.1 Raw tensor invariants

Immediately after backend extraction, assert:

- `B_raw.shape   == (27, 27)`
- `c1_raw.shape  == (27,)`
- `c2_raw.shape  == (27, 27)`
- `D_raw.shape   == (27, 27)`

and:

- `B_raw` is symmetric binary
- `c2_raw` is symmetric finite
- `D_raw` is symmetric finite
- `diag(B_raw) == 0`
- `diag(c2_raw) == 0`
- `diag(D_raw) == 0`

#### 6.7.2 Normalized tensor invariants

After normalization, assert:

- `B.shape   == (27, 27)`
- `c1.shape  == (27,)`
- `c2.shape  == (27, 27)`
- `D.shape   == (27, 27)`

and:

- `B` is symmetric binary
- `c1` is finite
- `c2` is symmetric finite
- `D` is symmetric finite
- `diag(B) == 0`
- `diag(c2) == 0`
- `diag(D) == 0`

This split is mandatory so that raw extraction bugs and normalization bugs are not conflated.

---

## 3. Clarification: Hardware Branch Feature Definitions

### Replace Section 9.8 with:

### 9.8 Hardware branch input features

For each hardware qubit `j`, define:

- `deg(j) = sum_k B[j, k]`
- `Nbr(j) = { k | B[j, k] = 1 }`

Then define:

- `mean_c2(j) = mean_k c2[j, k] over k in Nbr(j)`, else `0`
- `min_c2(j)  = min_k  c2[j, k] over k in Nbr(j)`, else `0`
- `mean_D(j)  = mean_k D[j, k] over all k in {0, ..., 26}`

Then:

`feat_H[j] = [ c1[j], deg(j), mean_c2(j), min_c2(j), mean_D(j) ]`

This wording makes it explicit that `c2` is indexed as a row-wise pairwise matrix, not a one-dimensional vector.

---

## 4. Clarification: Cross-Attention Variable Naming

### Replace Section 10.7 with:

### 10.7 Cross-attention rule

At each injection point, with feature map `F` of shape `(Bsz, C, H, W)`:

- `F_flat = reshape(F)` to `(Bsz, H*W, C)`
- `Q = F_flat @ W_Q`
- `K = T_hw* @ W_K`
- `V = T_hw* @ W_V`

Define:

`AttnOut = softmax(Q K^T / sqrt(d_a)) V`

Then:

- `DeltaF_flat = AttnOut @ W_O`
- `DeltaF = reshape(DeltaF_flat)` back to `(Bsz, C, H, W)`

Residual update:

`F <- F + alpha_attn * DeltaF`

Use:

- multi-head attention,
- `alpha_attn` as a learnable scalar per block,
- initialize `alpha_attn = 0`.

This is only a notation clarification; it does not change the attention mechanism already specified in v1.1. The v1.1 attention rule already defined Q/K/V and the residual gated update. 

---

## 5. Clarification: Mapper Sinkhorn Implementation

### Replace Section 11.2 / 11.5 wording with:

### 11.2 Training assignment

Use **log-domain Sinkhorn** on decoded logits:

`P_map = LogSinkhorn(S_nat / tau_m)`

### 11.5 Mapper Sinkhorn default

The mapper assignment head must use the **same stabilized log-domain Sinkhorn implementation family** as the reindexer unless a pre-existing stabilized implementation is already in the codebase and is documented as numerically equivalent.

Default fallback values:

- `tau_m = 0.10`
- `T_m = 20`

### Implementation rule

Do **not** use naive exp-domain Sinkhorn for the mapper if log-domain Sinkhorn is already used for the reindexer.

This is a clarification, not a design change: v1.1 already explicitly required log-domain Sinkhorn for the reindexer and used Sinkhorn for the mapper assignment head, but did not state clearly enough that the mapper should also use the stabilized variant.  

---

## 6. Clarification: Meaning of “PST” Loss Terms

### Replace the opening text of Section 12 with:

### 12. Task Loss

The loss terms named `L_PST_1Q`, `L_PST_2Q`, and `L_PST_total` are **normalized proxy costs correlated with PST degradation**.

They are **not literal physical PST probabilities**, because they are built from normalized cost tensors `c1` and `c2`, including z-score normalized quantities.

Therefore:

- these terms may be negative,
- these terms are optimization proxies,
- these terms should be interpreted as “PST-related cost proxies,” not direct PST values.

### Implementation note

For code readability, it is recommended to use names such as:

- `L_pst_proxy_1q`
- `L_pst_proxy_2q`
- `L_pst_proxy_total`

while preserving the same formulas and the same role in the total objective.

This clarification is necessary because v1.1 defines these terms using normalized `c1` and `c2` tensors. 

---

## 7. Clarification: Pass-B Mapper Freeze Semantics

### Replace the relevant part of Section 14.5 with:

### 14.5 Pass B — reindexer update

1. recompute `R_L, R_H` without detach
2. rebuild reordered tensors
3. rebuild reordered hardware tokens
4. run the mapper as a **frozen differentiable operator**
5. decode `S_nat`
6. compute `P_map`
7. compute `L_task`
8. compute `L_loc`, `L_cons`
9. compute `L_reindex`
10. backward only into reindexer parameters
11. clip reindexer gradients
12. optimizer step for reindexer only

### Frozen differentiable operator rule

“Run mapper with mapper parameters frozen” means:

- mapper parameters must **not** be updated in Pass B,
- gradients must still be allowed to flow **through mapper outputs back into reindexer outputs**,
- therefore **do not** wrap the mapper forward in `torch.no_grad()`.

Operationally:

- set mapper parameters to `requires_grad = False` for Pass B, or otherwise exclude them from gradient accumulation/update,
- keep the mapper forward differentiable with respect to its inputs,
- after Pass B, restore normal mapper parameter state for the next Pass A.

### Determinism rule

To avoid stochastic mismatch during reindexer optimization:

- the mapper should run in **evaluation mode** during Pass B,
- so dropout and running-stat updates do not introduce noise into the reindexer objective,
- then return the mapper to training mode afterward.

This does not change the two-pass design; it only makes the intended behavior explicit.

The underlying two-pass structure is already part of v1.1. 

---

## 8. Clarification: Consistency-Loss Sampling

### Replace Section 13.2 opening with:

### 13.2 Consistency loss

Per Pass-B update, sample **one** random logical relabeling `Pi_L` and **one** random hardware relabeling `Pi_H`.

Use those same sampled relabelings consistently across all tensors in that Pass-B computation.

So for one Pass-B step:

- sample one `Pi_L`,
- sample one `Pi_H`,
- construct `A', m', B', c1', c2', D'`,
- rerun the reindexer,
- compute `L_cons` from the reordered outputs.

### Default rule

- one relabeling pair per Pass-B step,
- resampled every Pass-B step,
- not cached across epochs by default.

This removes ambiguity without changing the consistency-loss definition already present in v1.1. 

---

## 9. Clarification: Dataset Split Authority

### Append to Section 5.7:

### 5.7.1 Split authority

For any concrete experiment run, the authoritative split definition is the manifest files stored under the KMW project data-manifest directory.

Once manifests are generated for a run family, they must be committed or archived and reused for reproducibility.

### Default rule

- use a fixed random seed for manifest generation,
- do not regenerate manifests silently between runs,
- validation must be held out at the circuit-file level,
- no identical circuit file may appear across train/val/test.

This does not change the dataset policy in v1.1; it only makes reproducibility operational.

---

## 10. Clarification: Path Wording for KMW Implementation

### Replace path language like “Use the existing mapper file, e.g. ...” with:

If the KMW project is implemented under its own directory tree, the **behavioral contract** is authoritative, not the original file path.

So references to older paths such as:

- `mqm/networks/unet_mapper.py`
- `mqm/networks/reindexer.py`
- `training/train_label_free_v11.py`

should be interpreted as **reference implementation locations**, not mandatory final file paths.

For the KMW project, equivalent modules may live under the KMW package structure, as long as behavior matches the specification exactly.

This clarification is necessary because v1.1 contains both design-authority wording and legacy/example path wording.  

---

## 11. Summary of v1.1.1

v1.1.1 does **not** change:

- architecture,
- token definition,
- reindexer math,
- native-frame decode,
- dataset policy,
- two-pass optimization,
- or loss formulas.

It only clarifies:

- exact U-Net reuse meaning,
- raw vs normalized tensor checks,
- row-wise `c2` feature definitions,
- attention variable naming,
- mapper Sinkhorn implementation choice,
- PST-term interpretation,
- Pass-B mapper-freeze semantics,
- consistency-loss sampling,
- and split/path authority.


---

## 5. Authoritative U-Net Mapper Backbone Specification

The following mapper specification is the final authority for the mapper backbone in v1.2.

# U-Net Mapper Structure (Locked for v1.1.1)

This section defines the **authoritative mapper backbone** for the KMW method.

### Design intent

This U-Net is a **shallow conditional mapper** that operates in the **latent reordered frame**.

It must:

- take the reordered circuit matrix `A*` as the **only spatial input**
- take reordered hardware tokens `T_hw*` as the **only conditioning source**
- output latent-frame mapping logits `S*`
- leave native-frame decode and final assignment **outside** the backbone

This is the converted version of the old 5-channel U-Net design.

## Table of Contents

- [1. Locked architectural choices](#1-locked-architectural-choices)
- [2. Input / output contract](#2-input--output-contract)
  - [Spatial input](#spatial-input)
  - [Conditioning input](#conditioning-input)
  - [Backbone output](#backbone-output)
  - [Interpretation](#interpretation)
- [3. Channel schedule](#3-channel-schedule)
- [4. Spatial resolution schedule](#4-spatial-resolution-schedule)
- [5. High-level block sequence](#5-high-level-block-sequence)
- [6. Exact forward structure](#6-exact-forward-structure)
  - [Stage 1 — down1](#stage-1--down1)
  - [Stage 2 — down2](#stage-2--down2)
  - [Stage 3 — bottleneck](#stage-3--bottleneck)
  - [Stage 4 — up1](#stage-4--up1)
  - [Stage 5 — up2 / head](#stage-5--up2--head)
- [7. ASCII diagram](#7-ascii-diagram)
- [8. Cross-attention block](#8-cross-attention-block)
  - [Attention dimensions](#attention-dimensions)
  - [Why 4 heads are locked](#why-4-heads-are-locked)
  - [Cross-attention rule](#cross-attention-rule)
- [9. Skip-connection policy](#9-skip-connection-policy)
- [10. What is intentionally NOT included](#10-what-is-intentionally-not-included)
- [11. Decode and assignment are outside the backbone](#11-decode-and-assignment-are-outside-the-backbone)
- [12. Reimplementation notes](#12-reimplementation-notes)
  - [Must preserve](#must-preserve)
  - [Allowed implementation freedom](#allowed-implementation-freedom)
  - [Not allowed without a later ablation revision](#not-allowed-without-a-later-ablation-revision)
- [Short update summary](#short-update-summary)

---

## 1. Locked architectural choices

The following choices are fixed:

- keep the **same shallow backbone**
- keep the **same three attention injection points**
- keep the **same additive skip**
- change the spatial input from **5-channel `X`** to **1-channel `A*`**
- remove logical tokens completely
- use **hardware-only tokens** from `B*`, `c2*`, `c1*`
- keep decode/native-frame assignment outside the backbone
- implement **actual multi-head attention**
- use **4 attention heads**
- make interpolation mode explicit
- do **not** deepen the network yet
- do **not** add full U-Net copy/crop skips yet

---

## 2. Input / output contract

### Spatial input

```text
A* : (B, 1, 27, 27)

B = batch size

channel count is always 1

A* is the reordered circuit interaction matrix
```

### Conditioning input

```text
T_hw* : (B, 27, 128)

where each hardware token comes from:

x_hw*[j] = concat(B*[j, :], c2*[j, :], c1*[j])    # shape (55,)
t_hw*[j] = MLP_hw(x_hw*[j])                        # shape (128,)

So:

T_hw* = stack(t_hw*[0], ..., t_hw*[26])           # shape (27, 128)
```

### Backbone output

```text
S* : (B, 1, 27, 27)
```

### Interpretation

```text
S*[u, j] = latent-frame logit for mapping latent logical slot u
           to latent hardware slot j
```

The backbone must return (B, 1, 27, 27) directly.
Do not squeeze the channel dimension inside the mapper.

---

## 3. Channel schedule

Use this fixed channel schedule:

```text
input     : 1
down1     : 64
down2     : 128
bottleneck: 256
up1       : 128
head      : 1
```

This is intentionally shallow.

---

## 4. Spatial resolution schedule

Use this fixed spatial schedule:

```text
27 x 27  ->  14 x 14  ->  7 x 7  ->  14 x 14  ->  27 x 27
```

All resizing must use explicit interpolation settings:

```text
mode = bilinear
align_corners = False
```

Use this same interpolation rule for every spatial resize inside the backbone.

---

## 5. High-level block sequence

The mapper has 5 main spatial stages:

- down1
- down2 + attention
- bottleneck + attention
- up1 + additive skip + attention
- up2 + output head

There is only one active skip connection:

```text
up1_pre_attn = up1_pre_attn + down2_out
```

There is no skip from down1.

There are no copy-and-crop U-Net skips.

---

## 6. Exact forward structure

### Stage 1 — down1

**Input:**

```text
A*                           : (B, 1, 27, 27)
```

**Operation:**

```text
d1 = ReLU( Conv3x3(1 -> 64)(A*) )
```

**Output:**

```text
d1                           : (B, 64, 27, 27)
```

### Stage 2 — down2

**Resize:**

```text
d1_ds = Interpolate(d1, size=(14, 14))
```

**Operation:**

```text
d2_pre = ReLU( Conv3x3(64 -> 128)(d1_ds) )
d2     = CrossAttn(d2_pre, T_hw*)
```

**Output:**

```text
d2                           : (B, 128, 14, 14)
```

### Stage 3 — bottleneck

**Resize:**

```text
d2_ds = Interpolate(d2, size=(7, 7))
```

**Operation:**

```text
b_pre = ReLU( Conv3x3(128 -> 256)(d2_ds) )
b     = CrossAttn(b_pre, T_hw*)
```

**Output:**

```text
b                            : (B, 256, 7, 7)
```

### Stage 4 — up1

**Resize:**

```text
b_us = Interpolate(b, size=(14, 14))
```

**Operation:**

```text
u1_pre  = ReLU( Conv3x3(256 -> 128)(b_us) )
u1_skip = u1_pre + d2
u1      = CrossAttn(u1_skip, T_hw*)
```

**Output:**

```text
u1                           : (B, 128, 14, 14)
```

### Stage 5 — up2 / head

**Resize:**

```text
u1_us = Interpolate(u1, size=(27, 27))
```

**Operation:**

```text
S* = Conv3x3(128 -> 1)(u1_us)
```

**Output:**

```text
S*                           : (B, 1, 27, 27)
```

No activation is applied after the final output conv.

---

## 7. ASCII diagram

```text
A* : (B,1,27,27)
  │
  ├─ Conv3x3(1→64) + ReLU
  ▼
d1 : (B,64,27,27)
  │
  ├─ Interpolate → (14,14)
  ├─ Conv3x3(64→128) + ReLU
  ├─ CrossAttn(T_hw*)   [4 heads]
  ▼
d2 : (B,128,14,14)
  │
  ├─ Interpolate → (7,7)
  ├─ Conv3x3(128→256) + ReLU
  ├─ CrossAttn(T_hw*)   [4 heads]
  ▼
 b : (B,256,7,7)
  │
  ├─ Interpolate → (14,14)
  ├─ Conv3x3(256→128) + ReLU
  ├─ Add skip from d2
  ├─ CrossAttn(T_hw*)   [4 heads]
  ▼
u1 : (B,128,14,14)
  │
  ├─ Interpolate → (27,27)
  ├─ Conv3x3(128→1)
  ▼
S* : (B,1,27,27)
```

---

## 8. Cross-attention block

Cross-attention is injected at exactly 3 points:

- after down2
- after bottleneck
- after up1 skip fusion

### Attention dimensions

Use:

```text
token_dim      = 128
attn_dim       = 128
num_heads      = 4
head_dim       = 32
```

since:

```text
128 / 4 = 32
```

Use the same head count at all three injection points.

### Why 4 heads are locked

4 heads are the chosen compromise because they provide enough specialization without making each head too narrow.

- 2 heads are too coarse
- 8 heads are too thin for this model size
- 4 heads with 32 dims/head is the best balanced default

### Cross-attention rule

For a feature map F : (B, C, H, W):

```text
F_flat = reshape(F) to (B, H*W, C)

Q = Linear(C   -> 128)(F_flat)
K = Linear(128 -> 128)(T_hw*)
V = Linear(128 -> 128)(T_hw*)
```

Then split into 4 heads:

```text
Q -> (B, heads=4, H*W, 32)
K -> (B, heads=4, 27,  32)
V -> (B, heads=4, 27,  32)
```

Per head:

```text
Attn_h = softmax( Q_h K_h^T / sqrt(32) ) V_h
```

Concatenate all heads:

```text
AttnOut = concat(Attn_1, ..., Attn_4)   # (B, H*W, 128)
```

Project back to the feature-map channel dimension:

```text
DeltaF_flat = Linear(128 -> C)(AttnOut)
DeltaF      = reshape(DeltaF_flat) back to (B, C, H, W)
```

Residual gated update:

```text
F_out = F + alpha_attn * DeltaF
```

where:

```text
alpha_attn
```

is a learnable scalar per attention block, initialized to:

```text
0.0
```

This makes attention start as a no-op and become useful only if training learns to use it.

---

## 9. Skip-connection policy

Use exactly this skip policy:

```text
u1_skip = u1_pre + d2
```

Rules:

- additive skip only
- only one skip
- skip happens at the 14 x 14 level
- no concatenation skip
- no down1 -> up2 skip
- no full classical U-Net skip ladder

---

## 10. What is intentionally NOT included

The following are intentionally excluded in v1.1.1:

- no 5-channel spatial input
- no logical tokens
- no token type embeddings
- no extra down/up levels
- no copy-and-crop skip ladder
- no attention at every block
- no decode inside the backbone
- no final Hungarian/Sinkhorn inside the backbone
- no D inside the hardware token used by the mapper

D stays in loss / routing proxy logic, not in mapper token conditioning.

---

## 11. Decode and assignment are outside the backbone

The backbone stops at latent-frame logits:

```text
S* : (B, 1, 27, 27)
```

Then outside the backbone:

```text
S_nat = R_L^T S* R_H
P_map = LogSinkhorn(S_nat / tau_m)          # training
M_map = Hungarian(S_nat)                    # inference
```

So the U-Net is a latent-frame score generator, not the full mapping pipeline.

---

## 12. Reimplementation notes

### Must preserve

- exact stage count
- exact channel schedule
- exact attention placement
- exact single skip structure
- exact interpolation policy
- exact 4-head attention choice

### Allowed implementation freedom

- helper function names
- internal class organization
- whether each stage is split into helper methods
- whether attention is implemented in a separate class or inline inside model.py

### Not allowed without a later ablation revision

- adding more blocks
- adding more skips
- changing channel widths
- changing number of attention heads
- changing attention insertion points
- changing the spatial input back to multi-channel form

---

## Short update summary

- 4 attention heads are now locked
- the converted U-Net stays shallow
- input is `A*` only
- conditioning is `T_hw*` only
- one additive skip only
- three attention blocks only
- ready to drop directly into `Design Plan v1.1.1.md`


---

## 6. Practical Interpretation for Implementation

This section restates how the combined plan should be read during coding.

### 6.1 What is fixed

- fixed physical size: `27`
- batch size default: `1`
- native-frame label-free training
- learned soft logical and hardware reindexing
- mapper input: reordered circuit matrix `A*` only
- mapper conditioning: reordered hardware tokens `T_hw*` only
- decode to native frame before task loss
- Sinkhorn during training, Hungarian during inference
- finite-only policy across preprocessing, attention, and losses
- strict fail-fast checks for NaN / Inf
- pure `src/kmw/` package implementation
- first milestone: stable MQT smoke pipeline

### 6.2 What is intentionally deferred

- richer gate-type weighting for `A`
- extra circuit channels beyond `A`
- inclusion of `D` in mapper hardware tokens
- full activation of reindexer auxiliary losses in the first stable run
- direction-aware `c2` modeling beyond symmetric reduction
- GraphQMap label supervision
- multi-backend / variable-hardware training

### 6.3 Immediate implementation order

1. preprocessing pipeline,
2. manifest builder and dataset loader,
3. model file with reindexer + token encoder + mapper + assignment helpers,
4. loss file,
5. trainer with two-pass optimization,
6. tests,
7. evaluation and CLI.

---

## 7. End of Combined Plan

This file intentionally retains overlapping material from the source documents so that no implementation-relevant detail is dropped.


