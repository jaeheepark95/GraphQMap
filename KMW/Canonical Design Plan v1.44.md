> **Update note (2026-04-02, Canonical v1.43 — Real-Circuit Training Revision):** This revision incorporates the **Modified Dataset + Training Plan (Label-Free, Real-Circuit Training Branch)** into the canonical-hardware branch while preserving the same model family, canonicalization logic, and Option A runtime semantics.
>
> **What changed in this revision**
> - Real circuits from **QASMBench** and **RevLib** are now part of training, not evaluation-only.
> - Training remains **label-free** with no supervised stage.
> - Dataset splitting is now explicitly defined **per source** for both synthetic and real datasets, with clean real held-out test sets.
> - Training is now staged:
>   - **Stage 0** smoke test
>   - **Stage 1** synthetic warmup
>   - **Stage 2** mixed synthetic + real training
>   - **Stage 3** real-focused calibration.
> - Validation now uses two pools:
>   - `val_synth_macro`
>   - `val_real_macro`
> - Checkpoint selection is now:
>   - primary = `val_real_macro`
>   - secondary guardrail = `val_synth_macro`
> - Epochs are no longer defined as raw concatenated passes; the branch now uses **fixed-length epochs** with a recommended default of **6000 balanced draws per epoch**.
> - Final evaluation is now explicitly partitioned into:
>   - `test_real_core20`
>   - `test_real_broad80`
>   - `test_benchmarks_ext`
>
> **Purpose of this revision**
> - To make the canonical branch a proper **real-circuit label-free training branch** rather than a synthetic-only warmup branch, while preserving leakage prevention, mandatory routed metrics, and the existing canonical-hardware model design.

> **Update note (2026-04-01, Canonical v1.42):** This revision updates the canonical-hardware branch to address three implementation-priority issues while keeping the streamlined mapper representation unchanged.
>
> **What changed in this revision**
> - Source-balanced sampling is no longer deferred; it is now a required part of the main-run training protocol across `queko`, `mlqd`, and `mqt_bench`.
> - Assignment is now defined only over **active logical rows**. Padded dummy rows must not participate in Sinkhorn during training or Hungarian during inference.
> - The route term is now explicitly reweighted with a configurable `lambda_route`, and loss-decomposition logging is mandatory.
>
> **Required new training diagnostics**
> - `L_1Q`
> - `L_RO`
> - `L_2Q`
> - `L_native`
> - `L_route`
> - `L_task`
> - `route_to_native_ratio = L_route / max(L_native, eps)`
>
> **Purpose of this revision**
> - To close three likely performance-degradation gaps that remained after removing the learned reindexer: train-distribution skew, wasted assignment capacity on padded rows, and underweighted routing pressure.

> **Update note (2026-03-31, Canonical v1.41):** This document is the **hardware-canonical ablation replacement** of `Combined Design Plan v1.4`.
>
> **What changed relative to the learned-reindexer v1.4 plan**
> - Removed the learned soft reindexer entirely.
> - Removed both learned permutation branches:
>   - logical reindexer `R_L`
>   - hardware reindexer `R_H`
> - Removed all learned-reindexer-specific machinery:
>   - `tau_r` schedule
>   - reindexer Pass-A / Pass-B split
>   - locality loss
>   - consistency loss
>   - entropy / sharpness diagnostics
>   - `freeze_hardware_reindex`
>   - logical canonical-teacher pretraining
> - Replaced the learned reindexer with the **fixed canonical hardware reindexer** from the old canonical design.
> - Locked the experiment to **hardware canonical reindexing only**:
>   - logical order remains identity
>   - circuit tensor is **not** canonically reordered
> - Kept the rest of the v1.4 mapper design as unchanged as possible:
>   - same shallow conditional U-Net backbone
>   - same 1-channel circuit spatial input
>   - same hardware-only token conditioning
>   - same Sinkhorn / Hungarian assignment policy
>   - same v1.4.1 execution-surrogate loss family
> - Locked the runtime to the corrected **Option A** semantics:
>   - `native -> canonical reindexing -> mapper -> decode -> native-frame loss`
> - Updated the implementation layout to match the **one-main-file-per-subsystem** structure requested for `src/kmw1`.
>
> **What did not change**
> - This is **not** a rollback to full v1_1.
> - We do **not** restore:
>   - the old 5-channel spatial U-Net input
>   - logical tokens
>   - the old lambda-weighted PST / SWAP / depth loss family
>
> **Purpose of this amendment**
> - To test whether the main performance drop in learned-reindexer v1.4 is caused primarily by the reindexer path rather than by the streamlined mapper backbone itself.

# Canonical Design Plan v1.43

## 0. Document Status

This document is the implementation authority for the **canonical-hardware v1.4 ablation**.

It supersedes the learned-reindexer sections of `Combined Design Plan v1.4` where the two documents conflict.

This branch is intentionally implemented under a separate package namespace:

```text
src/kmw1
```

so that the canonical ablation does not contaminate the current learned-reindexer implementation in:

```text
src/kmw
```

This document is written for implementation, not presentation.

### 0.1 Precedence rule for conflicts

If two source documents disagree, use this order of authority:

1. explicit implementation decisions locked for **Canonical Design Plan v1.43**,
2. the v1.4.1 loss-replacement rule in this document,
3. the locked v1.4 shallow conditional U-Net backbone authority,
4. the older `Combined Design Plan v1.4`,
5. the older canonical / learned-reindexer plans only as historical references.

### 0.2 One-line description

This project is now:

> a stability-first, label-free initial qubit mapper that canonicalizes **hardware only**, keeps the **streamlined v1.4 circuit-as-image + hardware-as-attention** mapper design, decodes canonical logits back to native hardware order before assignment/loss, uses the **v1.4.1 execution-surrogate loss** without any learned reindexer, and now supports staged **real-circuit label-free training**.

---

## 1. High-Level Project Overview
## 1. High-Level Project Overview

### 1.1 What the project is trying to do

The project is a learned initial qubit mapper for a fixed 27-qubit IBM BackendV2 device.

Its job is **not** to perform full routing.

Its job is to choose a strong initial logical-to-physical placement so that later routing incurs less performance loss, especially in terms of:

- execution-quality degradation,
- SWAP overhead,
- depth increase.

The output is still a one-to-one mapping between logical qubits and physical qubits.

### 1.2 The main conceptual change in this canonical branch

The learned-reindexer v1.4 path was:

```text
native circuit/backend tensors -> learned soft reindexing -> mapper -> decode -> native-frame loss
```

That path is gone in this branch.

The canonical-hardware v1.43 path is now:

```text
native circuit/backend tensors -> fixed hardware canonicalization -> mapper in canonical hardware frame -> deterministic decode back to native hardware order -> native-frame loss
```

So the model no longer learns an internal permutation basis.

Instead:

- the circuit remains in original logical order,
- the hardware is normalized once into a deterministic canonical order,
- the mapper predicts logits over canonical hardware slots,
- those logits are deterministically decoded back to native hardware order,
- assignment and task loss are evaluated in the native frame after decode.

### 1.3 How to think about the model architecture
### 1.3 How to think about the model architecture

The design remains close to an Imagen-style conditioning idea:

- the circuit is treated like the main spatial object, similar to an "image"
- the hardware is treated like conditioning information, similar to "text"

Concretely:

- the U-Net input is the circuit-side matrix `A`
- the queries in attention come from the circuit-side feature maps
- the keys/values come from **hardware-only tokens**
- those hardware tokens are built in **canonical hardware order**

So the hardware is still not mixed into the U-Net input as the old 5-channel image.
Instead, the hardware conditions the circuit representation through cross-attention.

### 1.4 What the main inputs mean now

The notation remains centered around:

- `A = circuit representation`
- `B = hardware adjacency / connectivity`
- `c1 = per-physical-qubit cost`
- `c2 = per-physical-edge cost`
- `D = shortest-path distance matrix on the hardware graph`

But in this canonical branch:

- the circuit tensor `A` stays in original logical order
- the hardware tensors are canonicalized to:
  - `B_can`
  - `c1_can`
  - `c2_can`
  - `D_can`

### 1.5 How the hardware is represented

Each physical qubit becomes one hardware token.

For canonical hardware slot `j`, the token is built from:

\[
x^{(\mathrm{hw})}_{\mathrm{can}}[j]
=
\big[B_{\mathrm{can}}[j,:] \;\Vert\; c2_{\mathrm{can}}[j,:] \;\Vert\; c1_{\mathrm{can}}[j]\big]
\]

So each token contains:

- which other canonical hardware slots `j` is connected to
- how costly those physical couplings are
- how costly qubit `j` itself is

These raw token vectors are embedded with an MLP and used as keys/values in cross-attention.

### 1.6 How the circuit is represented

The circuit is not represented as a simple binary logical adjacency matrix.

Instead, `A` is a weighted logical interaction matrix:

- off-diagonal entries describe how strongly pairs of logical qubits interact, mainly through 2Q gate activity
- diagonal entries summarize per-logical-qubit burden

The circuit remains in its original logical order.
This branch does **not** introduce a logical canonicalizer.

### 1.7 How the model learns

The model remains label-free.

The training assignment head remains:

- Sinkhorn during training
- Hungarian during inference

The task objective remains the v1.4.1 execution-surrogate family.

The crucial change in this branch is:

- the mapper operates in **canonical hardware space**,
- then logits are **deterministically decoded back to native hardware order**,
- then assignment and task loss are computed in the **native frame after decode**.

This is the locked **Option A** choice for the ablation.

### 1.8 Stability philosophy
### 1.8 Stability philosophy

A major theme of this branch is numerical stability first.

That is why the branch locks:

- batch size = 1
- no infinities anywhere
- explicit validation of each major loss term
- fail loudly on NaN / Inf
- keep `D` finite
- keep non-edge handling explicit instead of encoding impossibility with `∞`
- remove the learned reindexer entirely

This is not just an optimization detail.
It is part of the design philosophy:
the model should learn from a stable hardware coordinate system and cleanly separated signals.

### 1.9 One-sentence summary

At a high level, this project is now:

> a stability-first, label-free initial qubit mapper that uses deterministic hardware canonicalization, treats the circuit as the main U-Net object and the hardware as attention conditioning, and is trained with the v1.4.1 execution-surrogate loss in the native frame after deterministic decode.

---

## 2. Final Workspace and File Structure

### 2.1 Project tree

```text
KMW/
├── readme.md
├── Canonical Design Plan v1.42.md
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
│   ├── kmw/
│   │   └── ...                       # existing learned-reindexer v1.4 branch
│   └── kmw1/
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

### 2.2 One-line purpose for each `src/kmw1` subsystem

- `src/kmw1/data/dataset.py` — manifest-driven dataset loading, split selection, circuit-file lookup, per-sample return contract, and source-balanced sampling support metadata
- `src/kmw1/preprocessing/pipeline.py` — backend extraction, circuit featurization, deterministic hardware canonicalization, normalization, finite-value validation, and deterministic canonical→native decode helpers for Option A
- `src/kmw1/models/model.py` — hardware token encoder, shallow conditional U-Net mapper, cross-attention blocks, assignment helpers, and any model-side helper layers formerly split out into smaller files
- `src/kmw1/losses/loss.py` — v1.4.1 execution-surrogate loss in the **native frame after deterministic decode**, active-row-only assignment/loss support, explicit `lambda_route` reweighting, and loss-side validation helpers
- `src/kmw1/training/trainer.py` — optimizer/scheduler setup, train/validation loops, checkpointing, source-balanced sampling hookup, mandatory loss-decomposition logging, fail-fast checks, and Option A training orchestration
- `src/kmw1/evaluation/evaluate.py` — inference path, deterministic decode, active-row-only native-frame assignment, proxy metrics, and **mandatory** routed evaluation outputs (`real_pst_gate_readout`, SWAP/depth/compile-time metrics), plus CSV/JSON summary generation
- `src/kmw1/cli/main.py` — unified CLI entrypoint for train/eval/full-run evaluation
- `src/kmw1/utils.py` — shared utility functions that are not model-, preprocessing-, loss-, training-, or evaluation-specific

### 2.3 Design rule

This structure is intentionally locked to the **same one-main-file-per-subsystem layout** as the learned-reindexer v1.4 project.

That means the following helper files must **not** exist in `src/kmw1`:

```text
extractor.py
featurizer.py
canonical_indexer.py
layers.py
```

Their functionality must be folded into:

- `preprocessing/pipeline.py`
- `models/model.py`

This keeps the canonical branch faithful to the requested file structure while changing only the internal logic required by the hardware-canonical ablation.

---

## 3. Full Base Implementation Specification (Canonical v1.43)
## 3. Full Base Implementation Specification (Canonical v1.43)

# Canonical Design Plan v1.43 — Implementation Specification

## Table of Contents

- [0. Document Status](#0-document-status)
- [1. Objective](#1-objective)
- [2. Non-Goals](#2-non-goals)
- [3. Locked Design Decisions](#3-locked-design-decisions)
  - [3.1 Hardware canonical indexing is restored](#31-hardware-canonical-indexing-is-restored)
  - [3.2 Logical canonical indexing is not used](#32-logical-canonical-indexing-is-not-used)
  - [3.3 Mapper conditioning style](#33-mapper-conditioning-style)
  - [3.4 U-Net input](#34-u-net-input)
  - [3.5 Hardware token definition](#35-hardware-token-definition)
  - [3.6 Distance matrix role](#36-distance-matrix-role)
  - [3.7 No infinities anywhere](#37-no-infinities-anywhere)
  - [3.8 Batch size](#38-batch-size)
  - [3.9 No learned-reindexer auxiliary losses](#39-no-learned-reindexer-auxiliary-losses)
- [4. Notation](#4-notation)
  - [4.1 Dimensions](#41-dimensions)
  - [4.2 Circuit tensors](#42-circuit-tensors)
  - [4.3 Hardware tensors](#43-hardware-tensors)
  - [4.4 Canonical permutation outputs](#44-canonical-permutation-outputs)
  - [4.5 Canonicalized tensors](#45-canonicalized-tensors)
  - [4.6 Mapper outputs](#46-mapper-outputs)
- [5. Dataset Protocol](#5-dataset-protocol)
  - [5.1 Phase A — stability smoke test](#51-phase-a--stability-smoke-test)
  - [5.2 Phase B — main comparison training](#52-phase-b--main-comparison-training)
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
  - [6.7 Raw per-qubit error preservation](#67-raw-per-qubit-error-preservation)
  - [6.8 Raw per-edge error preservation](#68-raw-per-edge-error-preservation)
  - [6.9 Raw distance preservation](#69-raw-distance-preservation)
  - [6.10 Backend extractor invariants](#610-backend-extractor-invariants)
- [7. Circuit Featurization](#7-circuit-featurization)
  - [7.1 Logical qubit count](#71-logical-qubit-count)
  - [7.2 Logical mask m](#72-logical-mask-m)
  - [7.3 Off-diagonal entries of A](#73-off-diagonal-entries-of-a)
  - [7.4 Default 2Q gate-type weights](#74-default-2q-gate-type-weights)
  - [7.5 Diagonal entries of A](#75-diagonal-entries-of-a)
  - [7.6 Padding](#76-padding)
  - [7.7 Symmetry](#77-symmetry)
  - [7.8 Degenerate circuit rejection](#78-degenerate-circuit-rejection)
  - [7.9 Logical count vectors](#79-logical-count-vectors)
- [8. Normalization and Finite-Value Policy](#8-normalization-and-finite-value-policy)
  - [8.1 Global epsilon](#81-global-epsilon)
  - [8.2 Circuit matrix A](#82-circuit-matrix-a)
  - [8.3 Hardware adjacency B](#83-hardware-adjacency-b)
  - [8.4 Per-qubit cost c1](#84-per-qubit-cost-c1)
  - [8.5 Per-edge cost c2](#85-per-edge-cost-c2)
  - [8.6 Distance matrix D](#86-distance-matrix-d)
  - [8.7 Summary of finite rules](#87-summary-of-finite-rules)
- [9. Canonical Hardware Indexer](#9-canonical-hardware-indexer)
  - [9.1 File](#91-file)
  - [9.2 Inputs and outputs](#92-inputs-and-outputs)
  - [9.3 Undirected neighborhood graph](#93-undirected-neighborhood-graph)
  - [9.4 Degree](#94-degree)
  - [9.5 Mean incident edge cost](#95-mean-incident-edge-cost)
  - [9.6 Root score](#96-root-score)
  - [9.7 BFS ordering](#97-bfs-ordering)
  - [9.8 Canonicalized hardware tensors](#98-canonicalized-hardware-tensors)
  - [9.9 Canonicalizer invariants](#99-canonicalizer-invariants)
- [10. Mapper](#10-mapper)
  - [10.1 File](#101-file)
  - [10.2 Input contract](#102-input-contract)
  - [10.3 Hardware token encoder](#103-hardware-token-encoder)
  - [10.4 No logical tokens](#104-no-logical-tokens)
  - [10.5 U-Net backbone](#105-u-net-backbone)
  - [10.6 Cross-attention placement](#106-cross-attention-placement)
  - [10.7 Cross-attention rule](#107-cross-attention-rule)
  - [10.8 No timestep embedding](#108-no-timestep-embedding)
- [11. Assignment and Mapping Output](#11-assignment-and-mapping-output)
  - [11.1 Training assignment](#111-training-assignment)
  - [11.2 Inference assignment](#112-inference-assignment)
  - [11.3 Mapping output](#113-mapping-output)
  - [11.4 Mapper Sinkhorn default](#114-mapper-sinkhorn-default)
- [12. Task Loss (v1.4.1 in native frame after deterministic decode)](#12-task-loss-v141-in-native-frame-after-deterministic-decode)
  - [12.1 Active logical set and masses](#121-active-logical-set-and-masses)
  - [12.2 Native raw error tensors and canonical frame conversion](#122-native-raw-error-tensors-and-canonical-frame-conversion)
  - [12.3 Reliability-weighted routing quantities](#123-reliability-weighted-routing-quantities)
  - [12.4 Soft survival and route-surrogate definitions](#124-soft-survival-and-route-surrogate-definitions)
  - [12.5 Final loss definitions](#125-final-loss-definitions)
  - [12.6 Dummy logical rows](#126-dummy-logical-rows)
- [13. Removed Learned-Reindexer Components](#13-removed-learned-reindexer-components)
- [14. Training Loop](#14-training-loop)
  - [14.1 Files](#141-files)
  - [14.2 Optimizer](#142-optimizer)
  - [14.3 Gradient clipping](#143-gradient-clipping)
  - [14.4 Single-pass mapper update](#144-single-pass-mapper-update)
  - [14.5 Temperature scheduling](#145-temperature-scheduling)
  - [14.6 Batch size](#146-batch-size)
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
- [18. Default Config](#18-default-config)
- [19. Implementation Order](#19-implementation-order)
- [20. Acceptance Criteria](#20-acceptance-criteria)
- [21. Explicitly Deferred to Later Revisions](#21-explicitly-deferred-to-later-revisions)
- [22. One-Paragraph Execution Summary](#22-one-paragraph-execution-summary)

---

## 0. Document Status

This document is the implementation authority for the canonical-hardware branch.

If this document conflicts with:

- the older learned-reindexer v1.4 specification,
- the older canonical-indexing v1_1 specification,
- or the earlier combined plan wording,

this document wins.

This document is written for implementation, not for presentation.

---

## 1. Objective

Build a label-free learned initial qubit mapper for a fixed 27-qubit IBM BackendV2 target.

The system must:

- accept a quantum circuit with `K <= 27` logical qubits,
- extract a fixed-size circuit representation,
- extract fixed native hardware tensors from the backend,
- compute a deterministic canonical hardware permutation,
- canonicalize hardware tensors once before the mapper sees them,
- run a U-Net-style mapper in the canonical hardware frame,
- produce a one-to-one initial mapping using Sinkhorn during training and Hungarian during inference,
- train against the v1.4.1 execution-surrogate objective in the canonical hardware frame,
- convert the final hard mapping back to native hardware IDs for output and routed evaluation.

The one-to-one assignment requirement and Sinkhorn/Hungarian contract are preserved from the prior mapper design; the fixed canonical hardware frame replaces the learned reindexer entirely.

---

## 2. Non-Goals

This revision does not do the following:

- full routing / explicit SWAP insertion during training,
- diffusion or timestep conditioning,
- backend-specific supervised imitation of GraphQMap labels,
- multi-backend training in a single run,
- runtime logical canonicalization,
- learned logical or hardware reindexing,
- restoration of the old 5-channel mapper input,
- restoration of logical tokens in the mapper,
- restoration of the old lambda-weighted PST / SWAP / depth loss family.

No diffusion/timestep embedding is used.

---

## 3. Locked Design Decisions

### 3.1 Hardware canonical indexing is restored

Do implement and use a deterministic canonical hardware indexer.

Compute a hardware permutation \(p\) from native hardware tensors and canonicalize hardware tensors once before the mapper sees them.

Use only:

- original logical circuit order,
- canonicalized hardware tensors for the mapper,
- deterministic decode from canonical logits back to native hardware order,
- native-frame assignment,
- native-frame task loss after decode.

### 3.2 Logical canonical indexing is not used
### 3.2 Logical canonical indexing is not used

Do **not** canonically reorder the circuit-side logical tensor.

Do **not** compute any logical permutation.

Use only:

- original logical tensor order,
- original logical mask order,
- original logical count-vector order.

### 3.3 Mapper conditioning style

Use Imagen-style circuit-to-hardware conditioning:

- the circuit tensor is the main U-Net spatial input,
- hardware is injected as cross-attention conditioning,
- queries come from U-Net feature maps,
- keys and values come from hardware-only tokens.

Do not use logical tokens in the mapper.

### 3.4 U-Net input

The U-Net input is only the circuit matrix `A`.

Do not use the old 5-channel input contract from the canonical v1_1 design as the mapper input.

### 3.5 Hardware token definition

For each canonical hardware slot `j`, define the raw hardware token as:

\[
x^{(\mathrm{hw})}_{\mathrm{can}}[j]
=
\mathrm{concat}(B_{\mathrm{can}}[j,:],\; c2_{\mathrm{can}}[j,:],\; c1_{\mathrm{can}}[j])
\]

with shape:

- `B_can[j, :] -> (27,)`
- `c2_can[j, :] -> (27,)`
- `c1_can[j]    -> (1,)`
- `x_hw_can[j]  -> (55,)`

This is the exact mapper hardware-token contract for this branch.

### 3.6 Distance matrix role

Keep `D` in:

- loss computation,
- routing proxy computation,

but do **not** include `D[j,:]` in the mapper hardware token.

### 3.7 No infinities anywhere

Never use `inf`, `-inf`, or large artificial sentinels in:

- `B`,
- `c1`,
- `c2`,
- `D`,
- attention inputs,
- loss computation,
- routing proxy computation.

### 3.8 Batch size

Use:

```text
batch_size = 1
```

as the default and recommended setting.

### 3.9 No learned-reindexer auxiliary losses

Do not implement or optimize:

- locality loss,
- consistency loss,
- any learned-permutation auxiliary objective.

There is no learned reindexer in this branch.

---

## 4. Notation

Use the following notation consistently in code and docs.

### 4.1 Dimensions

```text
n   = 27     # fixed physical qubit count
K   <= 27    # logical qubit count for a given circuit
Bsz = 1      # batch size
```

### 4.2 Circuit tensors

```text
A      : (27, 27)   # circuit interaction matrix, original logical order
m      : (27,)      # logical-valid mask
n1Q    : (27,)      # per-logical-qubit 1Q count
nmeas  : (27,)      # per-logical-qubit measurement count
```

### 4.3 Hardware tensors

Native-frame tensors:

```text
B_nat      : (27, 27)   # binary hardware adjacency / connectivity
c1_nat     : (27,)      # per-physical-qubit cost
e2q_nat    : (27, 27)   # raw 2Q edge error
c2_nat     : (27, 27)   # per-physical-edge cost
D_nat      : (27, 27)   # normalized shortest-path distance
D_raw_nat  : (27, 27)   # raw shortest-path distance
e1q_nat    : (27,)      # raw 1Q gate error
ero_nat    : (27,)      # raw readout error
```

Canonicalized tensors:

```text
B_can      : (27, 27)
c1_can     : (27,)
e2q_can    : (27, 27)
c2_can     : (27, 27)
D_can      : (27, 27)
D_raw_can  : (27, 27)
e1q_can    : (27,)
ero_can    : (27,)
```

### 4.4 Canonical permutation outputs

```text
p      : (27,)   # canonical slot -> native hardware ID
p_inv  : (27,)   # native hardware ID -> canonical slot
```

### 4.5 Canonicalized tensors

Canonicalization is defined as:

\[
B_{\mathrm{can}} = B_{\mathrm{nat}}[p][:,p]
\]

\[
c1_{\mathrm{can}} = c1_{\mathrm{nat}}[p]
\]

\[
e2q_{\mathrm{can}} = e2q_{\mathrm{nat}}[p][:,p]
\]

\[
c2_{\mathrm{can}} = c2_{\mathrm{nat}}[p][:,p]
\]

\[
D_{\mathrm{can}} = D_{\mathrm{nat}}[p][:,p]
\]

\[
D_{\mathrm{raw,can}} = D_{\mathrm{raw,nat}}[p][:,p]
\]

\[
e1q_{\mathrm{can}} = e1q_{\mathrm{nat}}[p]
\]

\[
ero_{\mathrm{can}} = ero_{\mathrm{nat}}[p]
\]

### 4.6 Mapper outputs

```text
S_can   : (27, 27)   # canonical-frame mapping logits
S_nat   : (27, 27)   # native-frame logits after deterministic decode
P_nat   : (27, 27)   # soft assignment in training, native frame
M_nat   : (27, 27)   # hard assignment in inference, native frame
```

Interpretation:

\[
S_{\mathrm{can}}[u, j]
=
	ext{score for mapping original logical qubit }u
	ext{ to canonical hardware slot }j
\]

Deterministic decode to native hardware order is defined by the inverse permutation:

\[
S_{\mathrm{nat}}[u, i] = S_{\mathrm{can}}[u, p^{-1}[i]]
\]

There is no learned decode rule in this branch.

---

\
## 5. Dataset Protocol

The dataset protocol remains manifest-driven and circuit-v2 compatible, but this revision now includes **real circuits in training** while remaining fully **label-free**. Real training is added from **QASMBench** and **RevLib**, while `benchmarks/` remains external-only evaluation.

### 5.1 Dataset sources

#### Synthetic sources
- `queko` = 900
- `mlqd` = 4,443
- `mqt_bench` = 1,219

#### Real sources
- `qasmbench` = 94
- `revlib` = 231

#### External-only evaluation
- `benchmarks/`

This branch therefore trains on both **synthetic** and **real** circuits, but keeps the entire pipeline **label-free** with no supervised target stage.

### 5.2 Per-source split definition

Splits are done **per source**, not after concatenation.

#### Synthetic sources
- `queko`: train = 810, val = 90
- `mlqd`: train = 3,999, val = 444
- `mqt_bench`: train = 1,097, val = 122

Synthetic totals:
- `train_synth = 5,906`
- `val_synth = 656`

#### Real sources
- `qasmbench`: train = 60, val = 10, test = 24
- `revlib`: train = 145, val = 30, test = 56

Real totals:
- `train_real = 205`
- `val_real = 40`
- `test_real_broad80 = 80`

### 5.3 Detailed real test subset and external benchmark test

Define:

- `test_real_core20` = curated subset of `test_real_broad80`
- recommended composition:
  - 10 `qasmbench`
  - 10 `revlib`

Use `test_real_core20` for:
- detailed per-circuit analysis
- tables
- representative examples
- qualitative comparisons

Keep:
- `test_benchmarks_ext` = all circuits in `benchmarks/`

Rules:
- never use `benchmarks/` for training or validation
- use `benchmarks/` only for final external stress/generalization evaluation

### 5.4 Rules for real test selection and leakage prevention

For `test_real_core20`, choose circuits by **stratified coverage**, not by convenience.

Recommended qubit-bin composition:
- 3–4Q: 4 circuits
- 5–6Q: 4 circuits
- 7–8Q: 4 circuits
- 9–10Q: 4 circuits
- 11–12Q: 4 circuits

Within those bins, vary:
- gate count
- depth
- 2Q gate density
- interaction-graph difficulty
- circuit family/type

Before splitting:
- remove duplicates / near-duplicates
- group related variants if possible
- keep grouped variants in the same split
- do not allow similar family members to leak between train and test

### 5.5 Stage 0 — smoke test

Use a small MQT smoke subset only.

Purpose:
- verify preprocessing
- verify canonicalization
- verify decode path
- verify assignment path
- verify logging
- verify routed evaluation

This stage is for correctness/stability only, not performance reporting.

### 5.6 Stage 1 — synthetic warmup

Train only on:
- `queko`
- `mlqd`
- `mqt_bench`

Sampling rule:
- choose source first
- then sample uniformly within that source bucket

Use equal-probability source-balanced sampling:
- `P(queko) = 1/3`
- `P(mlqd) = 1/3`
- `P(mqt_bench) = 1/3`

Purpose:
- learn stable placement behavior from larger synthetic sources
- avoid early overfitting to the small real datasets
- stabilize the model before mixed real+synthetic training

Recommended duration:
- about 20–30% of total training steps

### 5.7 Stage 2 — mixed synthetic + real training

Train on all five train sources:
- `queko`
- `mlqd`
- `mqt_bench`
- `qasmbench`
- `revlib`

Sampling uses **two-level group-balanced sampling**.

Group choice:
- synthetic group: 70%
- real group: 30%

Within synthetic group:
- uniform over `queko / mlqd / mqt_bench`

Within real group:
- proportional over train counts:
  - `qasmbench = 60`
  - `revlib = 145`

Approximate effective source probabilities:
- `queko`: 23.3%
- `mlqd`: 23.3%
- `mqt_bench`: 23.3%
- `qasmbench`: 8.8%
- `revlib`: 21.2%

Purpose:
- introduce real-circuit training signal
- prevent synthetic sources from fully dominating
- avoid oversampling tiny real sets too aggressively

Recommended duration:
- about 60–70% of total training steps

### 5.8 Stage 3 — real-focused calibration

Start from the best Stage-2 checkpoint.

Run a short final adaptation stage at a **lower learning rate**.

Group choice:
- synthetic group: 50%
- real group: 50%

Within groups:
- synthetic: uniform over `queko / mlqd / mqt_bench`
- real: proportional over `qasmbench / revlib`

Approximate effective source probabilities:
- `queko`: 16.7%
- `mlqd`: 16.7%
- `mqt_bench`: 16.7%
- `qasmbench`: 14.7%
- `revlib`: 35.3%

Purpose:
- bias the final model more strongly toward real-circuit behavior
- still retain enough synthetic exposure to avoid collapse or overfitting

Recommended duration:
- about 10–15% of total training steps
- lower LR than Stage 2

### 5.9 Validation strategy and checkpointing

Use two validation pools:

- `val_synth`
- `val_real`

#### Synthetic validation
Sources:
- `queko` val = 90
- `mlqd` val = 444
- `mqt_bench` val = 122

Define:

\[
\mathrm{val\_synth\_macro}
=
\frac{
L(\mathrm{queko\ val}) + L(\mathrm{mlqd\ val}) + L(\mathrm{mqt\ val})
}{3}
\]

#### Real validation
Sources:
- `qasmbench` val = 10
- `revlib` val = 30

Define:

\[
\mathrm{val\_real\_macro}
=
\frac{
L(\mathrm{qasmbench\ val}) + L(\mathrm{revlib\ val})
}{2}
\]

Checkpoint rule:
- primary criterion = `val_real_macro`
- secondary guardrail = `val_synth_macro`

Interpretation:
- final model should improve on real validation
- but should not catastrophically collapse on synthetic validation

### 5.10 Epoch definition

Do **not** define epoch as a raw pass over concatenated data.

Use a **fixed-length epoch** instead.

Recommended default:
- `1 epoch = 6000 balanced draws`

Reason:
- training uses oversampling / balanced source draws
- raw concatenated-pass semantics no longer reflect the actual sampling behavior
- fixed-length epochs make logging and run comparison easier

### 5.11 Final evaluation plan

After training, freeze the selected checkpoint and evaluate on:

#### Detailed real comparison
- `test_real_core20`

Use for:
- per-circuit tables
- representative examples
- detailed comparisons

#### Headline real performance
- `test_real_broad80`

Use for:
- primary real-world result
- aggregate headline metrics

#### External stress/generalization
- `test_benchmarks_ext`

Use for:
- out-of-distribution / external benchmark testing
- supplementary transfer analysis

### 5.12 Metrics to report

Keep routed downstream evaluation mandatory.

Required reported metrics include:
- proxy/task loss summary
- routed PST-related metrics
- SWAP overhead
- depth increase
- compile / routing runtime
- per-circuit detailed results on `test_real_core20`
- aggregate summary on `test_real_broad80`

These routed metrics remain mandatory outputs for official comparison and must not be treated as optional extras.

### 5.13 Labels

Do not use GraphQMap labels as direct training targets.

There is **no supervised stage** in this branch.
This method remains fully label-free and uses proxy-loss training only.

### 5.14 Hard inclusion rules

A circuit is included iff all are true:

- `2 <= K <= 27`
- contains at least one 2Q gate, unless a manifest builder explicitly enables `--allow-no-2q`
- parses successfully
- hardware preprocessing succeeds
- circuit interaction matrix is not all-zero off-diagonal

Keep disconnected logical graphs.
Only remove degenerate cases with no useful 2Q structure.

### 5.15 Train/val/test reproducibility rule

Use deterministic per-source splitting inside the full manifest builder.

Authoritative split storage remains under:

- `data/manifests/full/source_manifests/<source>/`
- `data/manifests/full/recipes/<recipe_name>/`

Rules:
- primitive source manifests are split independently per source using deterministic source-specific seeds
- combined recipe manifests are built from already-split primitive source manifests
- manifests must be reused for reproducibility
- no identical circuit file may appear across `train / val / test`

## 6. Backend Extraction

This module converts an IBM BackendV2 target into the native hardware tensors.

### 6.1 Input

```text
backend: IBM BackendV2
```

### 6.2 Output

```text
B_raw       : (27, 27)
c1_raw      : (27,)
c2_raw      : (27, 27)
D_raw       : (27, 27)
e1q_raw     : (27,)
ero_raw     : (27,)
e2q_raw     : (27, 27)
```

### 6.3 Hardware adjacency B

Construct a symmetric binary adjacency:

```text
B[i, j] = 1 if at least one usable 2Q direction exists between i and j
B[i, j] = 0 otherwise
B[i, i] = 0
```

Then enforce symmetry:

```text
B = max(B, B^T)
```

Use `B` only as topology/validity.
Do not encode hardware quality into `B`.

### 6.4 Per-qubit cost c1

Define:

- `e_ro[j]` = readout error for qubit `j` if available, else `0.0`
- `e_1q[j]` = mean error over supported 1Q basis gates on qubit `j`; if unavailable, `0.0`

Then:

\[
c1_{\mathrm{raw}}[j] = 0.5 \cdot e_{ro}[j] + 0.5 \cdot e_{1q}[j]
\]

This yields a scalar per physical qubit.

### 6.5 Per-edge cost c2

For each unordered pair `(i, j)`:

collect valid directional 2Q gate errors:

- `e_ij` if `(i -> j)` exists
- `e_ji` if `(j -> i)` exists

Then define:

```text
if no valid direction exists:
    c2_raw[i, j] = 0
elif only one valid direction exists:
    c2_raw[i, j] = that direction's error
else:
    c2_raw[i, j] = min(e_ij, e_ji)
```

Then enforce symmetry:

```text
c2_raw[j, i] = c2_raw[i, j]
c2_raw[i, i] = 0
```

This intentionally treats `c2` as a symmetric edge badness for initial mapping.

### 6.6 Distance matrix D

Build the shortest-path distance matrix on the graph defined by `B`.

Use either:

- Floyd-Warshall, or
- BFS from every node.

Define:

```text
D_raw[i, i] = 0
D_raw[i, j] = shortest hop count if reachable
D_raw[i, j] = 28 if unreachable
```

because `n + 1 = 28`.

### 6.7 Raw per-qubit error preservation

In addition to `c1_raw`, preserve:

- `e1q_raw[j]`
- `ero_raw[j]`

where:

- `e1q_raw[j]` is the mean raw 1Q gate error over supported 1Q basis gates on qubit `j`
- `ero_raw[j]` is the raw readout error on qubit `j`

These raw tensors are cached and returned for the loss/evaluation path.

### 6.8 Raw per-edge error preservation

In addition to `c2_raw`, preserve:

- `e2q_raw[i,j]`

where `e2q_raw[i,j]` is the symmetric raw native 2Q error used to build `c2_raw[i,j]`.

### 6.9 Raw distance preservation

In addition to normalized `D`, preserve raw shortest-path distance as:

- `D_raw`

The mapper continues to use normalized `D` only if needed in preprocessing.
The v1.4.1 loss uses raw `D_raw`.

### 6.10 Backend extractor invariants

Two sets of invariants must be checked.

#### 6.10.1 Raw tensor invariants

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

#### 6.10.2 Normalized tensor invariants

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

---

## 7. Circuit Featurization

This module converts a circuit into `(A, m, n1Q, nmeas)`.

### 7.1 Logical qubit count

Let:

```text
K = number of logical qubits in the circuit
```

Reject the circuit if:

```text
K > 27
```

### 7.2 Logical mask m

Define:

```text
m[u] = 1 for u < K
m[u] = 0 for u >= K
```

### 7.3 Off-diagonal entries of A

For `u != v`, define:

\[
A_{\mathrm{raw}}[u, v]
=
\log\Big(1 + \sum_{g \in G_{2Q}(u,v)} w_{2q}(\mathrm{type}(g))\Big)
\]

where:

- `G_2Q(u,v)` = all 2Q gates involving logical qubits `u` and `v`
- `w_2q(type)` = gate-type-specific default weight

Then enforce symmetry:

\[
A_{\mathrm{raw}}[v,u] = A_{\mathrm{raw}}[u,v]
\]

### 7.4 Default 2Q gate-type weights

Use the default weight map:

```text
cx      -> 1.0
cz      -> 1.0
ecr     -> 1.0
iswap   -> 1.2
swap    -> 3.0
other 2Q -> 1.0
```

This choice is intentionally simple and stable.

### 7.5 Diagonal entries of A

For each logical qubit `u`, define:

\[
A_{\mathrm{raw}}[u,u]
=
\log(1 + n_{1Q}[u] + n_{meas}[u])
\]

where:

- `n1Q[u]` = number of 1Q gates applied to logical qubit `u`
- `nmeas[u]` = number of measurement ops touching logical qubit `u`

### 7.6 Padding

Pad `A`, `m`, `n1Q`, and `nmeas` to size 27 using zeros.

### 7.7 Symmetry

Require:

```text
A[u, v] == A[v, u] for all u != v
```

### 7.8 Degenerate circuit rejection

Reject circuits with no meaningful 2Q structure unless explicitly allowed for a diagnostic-only run.

### 7.9 Logical count vectors

Preserve and return:

- `n1Q`
- `nmeas`

These are used by the v1.4.1 execution-surrogate loss.

---

## 8. Normalization and Finite-Value Policy

### 8.1 Global epsilon

Use:

```text
eps = 1e-8
```

for all divisions, normalizations, and logs.

### 8.2 Circuit matrix A

Normalize by a robust positive scalar.

Recommended default:

```text
A = A_raw / max(max(A_raw), eps)
```

If `A_raw` is already small and stable, keeping it raw is allowed as an ablation, but the default should be normalized.

### 8.3 Hardware adjacency B

`B` is binary and is not normalized.

### 8.4 Per-qubit cost c1

Normalize with:

```text
c1 = c1_raw / max(max(c1_raw), eps)
```

### 8.5 Per-edge cost c2

Normalize with:

```text
c2 = c2_raw / max(max(c2_raw), eps)
```

### 8.6 Distance matrix D

Normalize with:

```text
D = D_raw / max(max(D_raw), 1)
```

This keeps `D` finite, bounded, and well-scaled for auxiliary use.

### 8.7 Summary of finite rules

Must be true after normalization:

- all tensors finite
- no NaN
- no Inf
- no negative distances
- no invalid self-edges

---

## 9. Canonical Hardware Indexer

### 9.1 File

```text
src/kmw1/preprocessing/pipeline.py
```

### 9.2 Inputs and outputs

Input:

- `B_nat`
- `c1_nat`
- `c2_nat`

Output:

- `p`
- `p_inv`
- canonicalized hardware tensors

### 9.3 Undirected neighborhood graph

Construct:

\[
\text{adj}(i,j) = \mathbf{1}(B_{\mathrm{nat}}[i,j] + B_{\mathrm{nat}}[j,i] > 0)
\]

### 9.4 Degree

\[
\deg(i) = \sum_j \text{adj}(i,j)
\]

### 9.5 Mean incident edge cost

For each node `i`, define:

\[
\text{meanEdgeCost}(i)
=
\frac{1}{|\mathcal N(i)|}
\sum_{j \in \mathcal N(i)}
\min(c2_{\mathrm{nat}}[i,j], c2_{\mathrm{nat}}[j,i])
\]

If a node is isolated, use a fixed finite fallback.

### 9.6 Root score

Define:

\[
qscore(i)
=
z(c1_{\mathrm{nat}}[i])
+
z(\text{meanEdgeCost}(i))
-
0.3\,z(\deg(i))
\]

where `z(·)` is z-score normalization over the node set.

### 9.7 BFS ordering

Build the canonical order by repeated BFS:

1. choose the lowest-`qscore` unvisited root,
2. run BFS,
3. sort neighbors by:
   - lower edge cost first,
   - lower `c1` first,
   - higher degree first,
   - lower native ID first,
4. append visited nodes to the order,
5. if disconnected components remain, restart from the next lowest-`qscore` node.

This yields the deterministic permutation `p`.

### 9.8 Canonicalized hardware tensors

Use:

\[
B_{\mathrm{can}} = B_{\mathrm{nat}}[p][:,p]
\]

\[
c1_{\mathrm{can}} = c1_{\mathrm{nat}}[p]
\]

\[
c2_{\mathrm{can}} = c2_{\mathrm{nat}}[p][:,p]
\]

\[
D_{\mathrm{can}} = D_{\mathrm{nat}}[p][:,p]
\]

and likewise for raw error tensors.

### 9.9 Canonicalizer invariants

Must be true:

- `p` is a true permutation of `{0, ..., 26}`
- `p_inv[p[j]] = j`
- canonicalized tensors preserve symmetry and finiteness
- canonicalization is deterministic for fixed inputs

---

## 10. Mapper

### 10.1 File

```text
src/kmw1/models/model.py
```

### 10.2 Input contract

Mapper inputs are:

- spatial input: `A`
- conditioning input: `T_hw_can`

with shapes:

```text
A         : (Bsz, 1, 27, 27)
T_hw_can  : (Bsz, 27, d_model)
```

### 10.3 Hardware token encoder

The hardware token encoder embeds each raw token:

\[
x^{(\mathrm{hw})}_{\mathrm{can}}[j]
\rightarrow
t^{(\mathrm{hw})}_{\mathrm{can}}[j]
\]

using a small MLP.

### 10.4 No logical tokens

Do **not** build or use logical tokens.

The mapper sees the logical structure only through the circuit image `A`.

### 10.5 U-Net backbone

Keep the shallow conditional U-Net backbone already locked by v1.4.

Required properties:

- shallow channel schedule
- cross-attention injection at the same locked locations
- no timestep embedding
- no diffusion-specific blocks
- output head returns a `27 x 27` logit matrix

### 10.6 Cross-attention placement

Keep the same locked placement as the v1.4 backbone.

Queries come from circuit feature maps.
Keys and values come from canonical hardware tokens.

### 10.7 Cross-attention rule

Use standard multi-head cross-attention between:

- spatial circuit features as queries,
- hardware-token embeddings as keys/values.

### 10.8 No timestep embedding

There is no timestep, no noise schedule, and no diffusion embedding in this mapper.

---

## 11. Assignment and Mapping Output

### 11.1 Deterministic decode

The mapper outputs canonical-frame logits:

\[
S_{\mathrm{can}} \in \mathbb{R}^{27 	imes 27}
\]

Decode them deterministically back to native hardware order before assignment:

\[
S_{\mathrm{nat}}[u, i] = S_{\mathrm{can}}[u, p^{-1}[i]]
\]

Equivalently, the native-frame columns are just the canonical-frame columns permuted back by the inverse canonical mapping.

### 11.2 Training assignment

During training, assignment is defined **only over active logical rows**.

Let:

\[
K = \sum_u m[u]
\]

and define the active native-frame logits:

\[
S_{\mathrm{nat}}^{\mathrm{act}} = S_{\mathrm{nat}}[0:K, :]
\]

Apply Sinkhorn only to these active rows:

\[
P_{\mathrm{nat}}^{\mathrm{act}} = \mathrm{Sinkhorn}(S_{\mathrm{nat}}^{\mathrm{act}} / 	au)
\]

Interpretation:

\[
P_{\mathrm{nat}}^{\mathrm{act}}[u,i]
pprox
	ext{soft probability that active logical qubit }u
	ext{ maps to native hardware ID }i
\]

Padded dummy rows must **not** participate in Sinkhorn.

### 11.3 Inference assignment

During inference, compute a hard one-to-one assignment using Hungarian only on the active decoded native-frame logits:

\[
S_{\mathrm{nat}}^{\mathrm{act}} = S_{\mathrm{nat}}[0:K, :]
\]

This yields a hard active-row assignment:

```text
M_nat_act : (K, 27)
```

Padded dummy rows must **not** participate in Hungarian.

### 11.4 Mapping output

If Hungarian assigns active logical qubit `u` to native hardware ID `i`, then that native ID is the final reported placement for that active logical qubit.

If needed for debugging or comparison to canonical quantities, the corresponding canonical hardware slot is:

\[
j = p^{-1}[i]
\]

### 11.5 Mapper Sinkhorn default

Recommended default:

```text
sinkhorn_tau = 0.5
sinkhorn_iters = 30
```

These values may be tuned later, but they are the default starting point for the canonical branch.

---

## 12. Task Loss (v1.4.1 in native frame after deterministic decode)

The authoritative task loss is the **v1.4.1 execution-surrogate loss family**, evaluated in the **native frame after deterministic decode**.

This branch does **not** restore the old lambda-weighted PST/SWAP/depth loss.

### 12.1 Active logical set and masses

Let:

\[
\mathcal U = \{u \mid m[u] = 1\}
\]

Only active logical rows contribute to the main loss terms.

Define active logical mass vectors from:

- `n1Q[u]`
- `nmeas[u]`

### 12.2 Native raw error tensors

Raw native error tensors are extracted and preserved in native hardware order:

- `e1q_nat`
- `ero_nat`
- `e2q_nat`
- `D_raw_nat`
- `B_nat`

The v1.4.1 loss is evaluated using these **native-frame tensors** together with the decoded native-frame assignment/logits.

### 12.3 Reliability-weighted routing quantities

Define native-frame survival terms:

\[
s_{1Q}[i] = 1 - e1q_{\mathrm{nat}}[i]
\]

\[
s_{RO}[i] = 1 - ero_{\mathrm{nat}}[i]
\]

\[
s_{2Q}[i,j] = 1 - e2q_{\mathrm{nat}}[i,j]
\]

Then define native-frame hazard quantities using the v1.4.1 rule:

\[
q_{1Q}[i] = -\log(\max(s_{1Q}[i], arepsilon))
\]

\[
q_{RO}[i] = -\log(\max(s_{RO}[i], arepsilon))
\]

\[
q_{2Q}[i,j] = -\log(\max(s_{2Q}[i,j], arepsilon))
\]

Use raw native shortest-path distance `D_raw_nat` and the reliability-weighted route-cost construction from v1.4.1 to form the route hazard matrix `Croute_nat`.

### 12.4 Soft survival and route-surrogate definitions

Given `P_nat`, define soft per-logical-qubit survival terms:

\[
S_{1Q}[u] = \sum_i P_{\mathrm{nat}}[u,i] \cdot s_{1Q}[i]
\]

\[
S_{RO}[u] = \sum_i P_{\mathrm{nat}}[u,i] \cdot s_{RO}[i]
\]

For interacting logical pairs, define soft 2Q survival:

\[
S_{2Q}[u,v]
=
\sum_{i,j}
P_{\mathrm{nat}}[u,i]
P_{\mathrm{nat}}[v,j]
B_{\mathrm{nat}}[i,j]
s_{2Q}[i,j]
\]

Define native-frame route hazard:

\[
H_{route}[u,v]
=
\sum_{i,j}
P_{\mathrm{nat}}[u,i]
P_{\mathrm{nat}}[v,j]
Croute_{\mathrm{nat}}[i,j]
\]

and its route survival surrogate:

\[
S_{route}[u,v] = \exp(-H_{route}[u,v])
\]

### 12.5 Final loss definitions

The exact aggregation rule follows the v1.4.1 execution-surrogate family, evaluated in native frame using:

- `P_nat_act`
- `B_nat`
- `e1q_nat`
- `ero_nat`
- `e2q_nat`
- `D_raw_nat`
- `n1Q`
- `nmeas`
- `m`

The implementation must expose and log the following decomposed terms:

- `L_1Q`
- `L_RO`
- `L_2Q`
- `L_native = L_1Q + L_RO + L_2Q`
- `L_route`
- `L_task`
- `route_to_native_ratio = L_route / max(L_native, eps)`

The task loss must use an explicit route weight:

\[
L_{\mathrm{task}} = L_{\mathrm{native}} + \lambda_{\mathrm{route}} L_{\mathrm{route}}
\]

where `lambda_route` is a required config parameter for the canonical v1.42 branch.

Implementation rule:

- keep the v1.4.1 semantics for native and route subterms,
- do **not** fall back to old `L_PST_total + λ_S L_swap + λ_D L_depth`,
- do **not** silently switch back to canonical-frame loss,
- use canonicalization only to define the mapper frame, not the final task-loss frame,
- make the route contribution explicit and inspectable through `lambda_route` and the logged decomposition above.

### 12.6 Dummy logical rows

Dummy rows must not contribute real mass and must not participate in assignment.

Use `m`, `n1Q`, and `nmeas` to ensure padded logical rows have zero contribution to the meaningful task terms.

Implementation rule:
- slice to active logical rows before Sinkhorn
- slice to active logical rows before Hungarian
- build all meaningful task terms from active logical rows only
- do not allow padded rows to consume hardware columns in either training or inference

---

## 13. Removed Learned-Reindexer Components
## 13. Removed Learned-Reindexer Components

The following components are completely absent from this branch:

- learned logical reindexer `R_L`
- learned hardware reindexer `R_H`
- learned reindexer Sinkhorn
- `tau_r` schedule
- reindexer stage-specific locality loss
- reindexer stage-specific consistency loss
- reindexer entropy / sharpness diagnostics
- `freeze_hardware_reindex`
- logical canonical-teacher pretraining
- native-frame decode through learned permutations

There is no hidden partial reindexer.
The branch is truly no-learned-reindexer.

---

## 14. Training Loop

### 14.1 Files

```text
src/kmw1/training/trainer.py
src/kmw1/cli/main.py
```

### 14.2 Optimizer

Recommended default:
- `AdamW`
- base LR = `1e-4`
- weight decay = `1e-4`

Stage 3 must use a **lower LR** than Stage 2.

### 14.3 Gradient clipping

Use:

```text
grad_clip_norm = 1.0
```

### 14.4 Training stages

There is no Pass-A / Pass-B reindexer split.

The canonical v1.43 training flow is instead:

1. **Stage 0 — smoke test**
2. **Stage 1 — synthetic warmup**
3. **Stage 2 — mixed synthetic + real training**
4. **Stage 3 — real-focused calibration**

### 14.5 Stage-specific sampling rules

#### Stage 1
Use equal-probability source-balanced sampling across:
- `queko`
- `mlqd`
- `mqt_bench`

#### Stage 2
Use two-level group-balanced sampling:
- synthetic group = 70%
- real group = 30%

Within synthetic:
- uniform over `queko / mlqd / mqt_bench`

Within real:
- proportional over `qasmbench / revlib`

#### Stage 3
Use two-level group-balanced sampling:
- synthetic group = 50%
- real group = 50%

Within synthetic:
- uniform over `queko / mlqd / mqt_bench`

Within real:
- proportional over `qasmbench / revlib`

### 14.6 Single-step mapper update contract

For one training step:

1. draw a balanced sample according to the active stage sampler
2. load circuit / backend tensors
3. canonicalize hardware
4. forward mapper to obtain `S_can`
5. deterministically decode to `S_nat`
6. slice to active logical rows
7. run Sinkhorn on active rows only
8. compute native-frame-after-decode v1.4.1 loss with explicit `lambda_route`
9. log the required loss decomposition
10. backprop mapper parameters
11. update optimizer

### 14.7 Validation during training

Every validation interval, compute at least:

- `val_synth_macro`
- `val_real_macro`

Checkpointing rule:
- primary = best `val_real_macro`
- secondary guardrail = `val_synth_macro`

A checkpoint that improves real validation but catastrophically collapses synthetic validation must not be silently treated as an unquestioned winner.

### 14.8 Epoch semantics

Epoch is a **fixed-length balanced training unit**, not a literal pass over concatenated datasets.

Recommended default:
- `1 epoch = 6000 balanced draws`

This rule applies because staged oversampling and group-balanced sampling break the semantics of raw concatenated passes.

### 14.9 Loss-decomposition logging

Mandatory logged terms:
- `L_1Q`
- `L_RO`
- `L_2Q`
- `L_native`
- `L_route`
- `L_task`
- `route_to_native_ratio = L_route / max(L_native, eps)`

These must be logged during both training and validation.

## 15. Inference Protocol

### 15.1 Steps

For one circuit:

1. build `A, m, n1Q, nmeas`
2. extract native backend tensors
3. compute canonical permutation `p`
4. build canonical hardware tensors
5. build canonical hardware tokens
6. run mapper to obtain `S_can`
7. deterministically decode `S_can` to `S_nat`
8. run Hungarian on `S_nat`
9. emit native-ID logical-to-physical mapping
10. run routed downstream evaluation on that native-ID mapping

### 15.2 Output format

The canonical branch must output:

- canonical assignment if diagnostic mode is enabled,
- native-ID final mapping always,
- proxy evaluation metrics,
- and **mandatory** routed evaluation metrics for official final evaluation.

For official experiment reporting, the routed output must include at minimum:

- `real_pst_gate_readout`
- `swap_inserted_count`
- `added_2q_ops`
- `routing_compile_time_s`
- `routing_total_eval_time_s`
- `original_depth`
- `routed_depth`
- `depth_increase_abs`

These fields must appear in the per-circuit outputs and in dataset-level summary outputs.

---

## 16. Validation and Fail-Safes

### 16.1 Tensor checks

At preprocessing and model entry:

- check shapes
- check symmetry where required
- check finiteness
- check valid permutation properties

### 16.2 Loss checks

At each loss call:

- assert all major subterms are finite
- abort on NaN / Inf
- log the failing batch identifier

### 16.3 Sinkhorn sanity checks

Check:

- row sums approximately 1
- column sums approximately 1
- entries finite
- no mode collapse to invalid values

### 16.4 Gradient checks

After backward:

- detect NaN gradients
- detect Inf gradients
- optionally detect gross exploding norms

### 16.5 Hard-fail conditions

Immediately stop the run if any of the following occur:

- NaN / Inf in model outputs
- NaN / Inf in loss components
- invalid canonical permutation
- invalid assignment tensor
- invalid routed evaluation output due to malformed mapping

### 16.6 Warning conditions

Warn but continue if:

- Sinkhorn is only mildly imprecise
- batch contains weak interaction structure but still passes inclusion rules
- routed evaluation is skipped intentionally in a smoke run

### 16.7 Failure report

If a run aborts, log:

- run ID
- epoch / step
- circuit ID
- backend ID
- canonical permutation `p`
- tensor stats
- failing loss component name

---

## 17. File and Module Layout

Required canonical branch files:

```text
src/kmw1/data/dataset.py
src/kmw1/preprocessing/pipeline.py
src/kmw1/models/model.py
src/kmw1/losses/loss.py
src/kmw1/training/trainer.py
src/kmw1/evaluation/evaluate.py
src/kmw1/cli/main.py
src/kmw1/utils.py
```

Files that must **not** exist in `src/kmw1`:

```text
src/kmw1/preprocessing/extractor.py
src/kmw1/preprocessing/featurizer.py
src/kmw1/preprocessing/canonical_indexer.py
src/kmw1/models/layers.py
```

Integration rule:

- move backend extraction, circuit featurization, deterministic hardware canonicalization, tensor normalization, validators, and deterministic decode helpers into `preprocessing/pipeline.py`,
- move helper layers and any mapper-side helper classes into `models/model.py`,
- keep loss logic isolated in `loss.py`,
- do not scatter permutation/decode logic throughout unrelated modules.

---

## 18. Default Config

Recommended initial config:

```yaml
backend: ibm_backendv2_fixed_27q
batch_size: 1
optimizer: adamw
lr_stage1: 1.0e-4
lr_stage2: 1.0e-4
lr_stage3: 5.0e-5
weight_decay: 1.0e-4
grad_clip_norm: 1.0
sinkhorn_tau: 0.5
sinkhorn_iters: 30
use_learned_reindexer: false
use_logical_canonicalizer: false
use_hardware_canonicalizer: true
loss_family: v1.4.1_execution_surrogate
loss_frame: native_after_decode
lambda_route: 1.0
log_loss_decomposition: true

stage0_recipe: smoke_mqt
stage1_train_sources: [queko, mlqd, mqt_bench]
stage2_train_sources: [queko, mlqd, mqt_bench, qasmbench, revlib]
stage3_train_sources: [queko, mlqd, mqt_bench, qasmbench, revlib]

stage1_sampling: synth_uniform
stage2_sampling:
  synthetic_group_prob: 0.70
  real_group_prob: 0.30
  synthetic_within_group: uniform
  real_within_group: proportional_train_count
stage3_sampling:
  synthetic_group_prob: 0.50
  real_group_prob: 0.50
  synthetic_within_group: uniform
  real_within_group: proportional_train_count

epoch_draws: 6000
primary_checkpoint_metric: val_real_macro
secondary_guardrail_metric: val_synth_macro

real_train_enabled: true
external_only_eval_sources: [benchmarks]
detailed_real_eval_split: test_real_core20
headline_real_eval_split: test_real_broad80
project_namespace: kmw1
```

The exact LR values may be tuned later, but the structural rules above are part of the default branch contract.

## 19. Implementation Order

Implement in this order:

1. backend extraction
2. circuit featurization
3. canonical hardware indexer logic (inside `preprocessing/pipeline.py`)
4. canonical preprocessing pipeline
5. shallow conditional U-Net mapper
6. deterministic decode from `S_can` to `S_nat`
7. active-row-only assignment path (`S_nat_act`, `P_nat_act`, active-row Hungarian)
8. native-frame-after-decode v1.4.1 loss with explicit `lambda_route`
9. staged sampling engine:
   - Stage 1 synthetic uniform
   - Stage 2 70/30 synthetic-real
   - Stage 3 50/50 synthetic-real
10. dual validation reporting:
    - `val_synth_macro`
    - `val_real_macro`
11. checkpoint selection logic:
    - primary = `val_real_macro`
    - secondary guardrail = `val_synth_macro`
12. fixed-length epoch accounting (`6000 balanced draws`)
13. evaluation:
    - `test_real_core20`
    - `test_real_broad80`
    - `test_benchmarks_ext`
14. routed final evaluation hookup

Reason:
- the canonical pipeline must be correct before staged training has meaning,
- staged samplers and validation logic must exist before the real-circuit branch is comparable,
- final evaluation must preserve mandatory routed metrics on the held-out real sets and external benchmarks.

## 20. Acceptance Criteria

### 20.1 Data and splits

- manifests load correctly
- train / val / test separation is reproducible
- no illegal circuit sizes pass through
- no train/val/test leakage across per-source manifests
- real-train sources are present:
  - `qasmbench`
  - `revlib`
- `benchmarks/` remains external-only evaluation

### 20.2 Stage sampling

- Stage 1 uses equal-probability source-balanced sampling over `queko / mlqd / mqt_bench`
- Stage 2 uses 70% synthetic / 30% real group-balanced sampling
- Stage 3 uses 50% synthetic / 50% real group-balanced sampling
- within real stages, `qasmbench / revlib` are sampled proportionally to train counts

### 20.3 Validation and checkpointing

- `val_synth_macro` is computed and logged
- `val_real_macro` is computed and logged
- checkpoint selection uses `val_real_macro` as the primary criterion
- synthetic validation acts as a guardrail against catastrophic collapse

### 20.4 Numerics

- all tensors finite
- no NaN / Inf in forward or loss
- no invalid `D`

### 20.5 Assignment

- Sinkhorn produces approximately doubly-stochastic matrices over **active logical rows only**
- Hungarian yields valid one-to-one hard assignments over **active logical rows only**
- deterministic decode from canonical to native order is correct
- padded dummy rows do not participate in assignment

### 20.6 Logging

- train loss is logged
- validation metrics are logged
- failure reports are informative
- required loss decomposition is logged
- stage identity is logged
- the run record indicates which stage scheduler / sampler produced each checkpoint

### 20.7 Full-run routed evaluation

- final evaluation is performed on:
  - `test_real_core20`
  - `test_real_broad80`
  - `test_benchmarks_ext`
- the canonical branch can produce native-ID mappings that flow through routed evaluation
- per-circuit routed outputs include at minimum:
  - `real_pst_gate_readout`
  - `swap_inserted_count`
  - `added_2q_ops`
  - `routing_compile_time_s`
  - `routing_total_eval_time_s`
  - `original_depth`
  - `routed_depth`
  - `depth_increase_abs`
- dataset-level summary outputs include aggregates for the same mandatory routed metrics
- a run is not considered complete for official comparison unless these routed metrics are present

## 21. Explicitly Deferred to Later Revisions

The following are intentionally deferred:

- runtime logical canonicalization
- both-sides canonicalization experiments
- Option B native-frame loss evaluation
- reintroduction of any learned reindexer hybrid
- restoring logical tokens as a mapper ablation
- restoring 5-channel spatial input as a mapper ablation
- multi-backend experiments

These are separate ablations and are **not** part of the current canonical-hardware branch.

---

## 22. One-Paragraph Execution Summary

The canonical-hardware v1.43 branch keeps the streamlined v1.4 mapper architecture — circuit matrix as the only spatial U-Net input, hardware-only tokens as cross-attention conditioning, Sinkhorn for training, Hungarian for inference, and the v1.4.1 execution-surrogate loss family — but removes the learned reindexer entirely. Instead, the hardware is canonicalized once using a deterministic permutation computed from backend topology and costs, while the circuit remains in original logical order. This revision extends the branch into a **real-circuit label-free training branch**: Stage 1 performs synthetic warmup on `queko / mlqd / mqt_bench`, Stage 2 performs mixed synthetic + real training with 70/30 synthetic-real group-balanced sampling, and Stage 3 performs lower-LR real-focused calibration with 50/50 synthetic-real group balancing. Validation is split into `val_synth_macro` and `val_real_macro`, checkpoint selection is driven primarily by real validation, and epochs are defined as fixed-length balanced-draw units rather than raw concatenated passes. Final evaluation is mandatory on `test_real_core20`, `test_real_broad80`, and `test_benchmarks_ext`, with routed downstream metrics retained as required outputs for official comparison.

