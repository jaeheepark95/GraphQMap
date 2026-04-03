# Design Plan: Full-Run Routed Final Evaluation with Real Metrics

## 0. Document Status

This document is the implementation specification for adding **true routed downstream evaluation** to the KMW full run.

It is intended to remove ambiguity for code review and implementation.

This plan does **not** cover `build_manifest_full.py` implementation. That work is explicitly out of scope here.

This plan covers only:

1. switching final evaluation to use the full `circuit_v2` manifests already being produced elsewhere,
2. adding **real post-routing evaluation** to the existing evaluation pipeline,
3. reporting the requested real metrics after routing:
   - PST (**gate + readout included**)
   - compile time
   - SWAP overhead
   - depth increase

---

## 1. Why this belongs in evaluation, not training

The current project is an **initial mapper**, not a full router. The current design authority says the model’s job is to choose a strong initial logical-to-physical placement so that later routing incurs less loss, especially in PST, SWAP overhead, and depth increase. The model still trains with proxy terms and uses Hungarian only at inference. The existing `evaluate.py` already owns inference, hard reindexing, dataset-level evaluation, CSV writing, JSON summary writing, and currently contains routing placeholders for a future routed phase. Therefore, routed downstream evaluation must be implemented as an **evaluation-time extension**, not as a change to the training loop or loss formulation. 

---

## 2. Grounding in the current codebase

### 2.1 Current responsibility split

The current codebase already assigns these responsibilities:

- `src/kmw/evaluation/evaluate.py`
  - inference pipeline,
  - hard reindexing,
  - final Hungarian assignment,
  - per-circuit metrics,
  - CSV + summary writing,
  - routing placeholders.

- `src/kmw/cli/main.py`
  - thin CLI only,
  - builds dataset / loader,
  - builds model,
  - builds `EvalConfig`,
  - dispatches evaluation.

- `src/kmw/data/dataset.py`
  - manifest-driven sample loading,
  - returns native-frame tensors only,
  - keeps `qasm_relpath`, source, split, and other manifest metadata available in `metadata`.

### 2.2 Consequence for implementation

Therefore:

- **real routed metrics go into `evaluate.py`**,
- **new eval flags go into `main.py`**,
- **dataset semantics remain native-only and do not perform routing**,
- **training code remains unchanged**.

---

## 3. Locked design decisions for this revision

These decisions are fixed for this revision and are not optional.

### 3.1 Scope

This revision changes **final evaluation only**.

It must **not**:

- modify the mapper architecture,
- modify the reindexer,
- modify the loss formulas used in training,
- modify the two-pass training loop,
- modify dataset tensor semantics,
- turn training into routed training.

### 3.2 Dataset policy

The full run uses the full `circuit_v2` manifests produced by the separate manifest builder. Evaluation continues to treat the manifest as split authority.

### 3.3 Official transpilation setting

The official downstream transpilation setting for the research run is:

- `optimization_level = 0`

This is the official/default setting for the routed evaluation path in this revision.

### 3.4 Official PST definition

The real PST metric for this revision must include:

- gate error contributions, and
- readout error contributions.

### 3.5 Official SWAP-overhead reporting

The evaluation must report **both**:

1. exact inserted SWAP count, and
2. added two-qubit operation count.

### 3.6 Backend convention

This project remains fixed to a 27-qubit `BackendV2`-style backend for the current phase.
The current environment lock is Qiskit `2.3.1`.

---

## 4. Non-goals and explicit exclusions

This revision does **not** attempt to solve or include any of the following:

- routed training loss,
- reinforcement learning over routing,
- multiple routing methods in the official comparison table,
- variable-hardware training,
- live hardware execution,
- calibration refresh,
- timing-aware scheduling metrics,
- fidelity simulation,
- stochastic noise simulation,
- per-layer EPLG-style hardware metrics,
- replacing proxy metrics with real metrics in training.

Proxy metrics remain in evaluation output because they are still useful diagnostics.

---

## 5. Current behavior that must be preserved

The following current behavior must remain unchanged when routed evaluation is disabled:

1. hard reindexing in inference,
2. Hungarian final assignment in the native frame,
3. current proxy metrics,
4. current per-circuit CSV writing,
5. current summary JSON writing,
6. current console summary behavior,
7. current failure-row behavior,
8. current CLI eval flow.

In other words:

- routed evaluation is an **extension**, not a rewrite.
- existing smoke/proxy evaluation must still run successfully.

---

## 6. Official routed evaluation protocol

This section defines the exact routed-evaluation algorithm.

### 6.1 Per-sample high-level flow

For each evaluation sample, the system must execute the following steps in this exact conceptual order:

1. load native tensors and metadata from the dataset,
2. run the current inference path exactly as now,
3. obtain the final hard mapping `M_map` in the native frame,
4. convert `M_map` into a logical-to-physical assignment vector,
5. load the original circuit from `qasm_relpath`,
6. construct a Qiskit `initial_layout` from the assignment,
7. run backend-targeted transpilation/routing using that fixed initial layout,
8. capture routing-stage SWAP statistics,
9. compute final post-routing metrics from the routed circuit,
10. append the real metrics to the row together with the existing proxy metrics,
11. include the new real metrics in CSV and summary outputs.

### 6.2 Mapping convention

The current inference code uses a permutation matrix `M_map` whose row-wise argmax gives the assigned physical index for each logical index.

This revision locks the assignment convention as:

- row index `u` = logical qubit index,
- column index `j` = physical qubit index,
- `assign[u] = j` means logical qubit `u` is placed on physical qubit `j`.

This exact convention must be used when building the Qiskit `initial_layout`.

### 6.3 Initial-layout construction

Let:

- `assign[u] = j`
- the original circuit have logical qubits in Qiskit order `qc.qubits[u]`

Then the `initial_layout` must map:

- `qc.qubits[u] -> j`

for every logical qubit `u` actually present in the original circuit.

#### 6.3.1 Important rule for circuits with `K < 27`

Only the logical qubits that exist in the loaded original circuit may be placed into the `initial_layout`.

The layout builder must **not** create fake virtual qubits for padded rows.

#### 6.3.2 Important validation rule

Before transpilation, the layout helper must validate:

- all assigned physical indices are integers,
- all assigned physical indices are unique for the circuit’s active logical qubits,
- all assigned physical indices are in `[0, n_qubits-1]`,
- the number of mapped virtual qubits equals the circuit’s logical qubit count.

If validation fails, the sample must become a failure row unless `fail_fast=True`.

---

## 7. Transpilation and routing settings

### 7.1 Official pass-manager construction

The implementation must use Qiskit’s preset transpilation pipeline for BackendV2-style targets.

The official path is:

- build a preset pass manager targeted to the backend,
- provide `initial_layout`,
- provide the requested `routing_method`,
- use `optimization_level=0`,
- run the pass manager on the original circuit.

### 7.2 Official settings for this revision

Default routed-eval settings:

- `optimization_level = 0`
- `routing_method = "sabre"`
- `seed_transpiler = <eval seed>`
- `layout_method = None`
- `translation_method = default`
- `scheduling_method = None`

### 7.3 Why layout must not be re-chosen

This evaluation is specifically testing the quality of the model’s predicted initial mapping.
Therefore the transpiler must be given the model-derived `initial_layout` directly.
The implementation must not introduce a second layout-selection policy that overrides or re-chooses the initial placement.

### 7.4 Compile-time definition

`routing_compile_time_s` is defined as:

- wall-clock elapsed time for the transpilation call itself,
- measured using `time.perf_counter()`,
- starting immediately before `pass_manager.run(circuit)`,
- ending immediately after the returned routed circuit is available.

This metric must exclude:

- model inference time,
- CSV writing time,
- summary generation time,
- PST postprocessing time.

### 7.5 Additional timing metric

In addition to `routing_compile_time_s`, the row should also report:

- `routing_total_eval_time_s`

which measures the full routed-evaluation overhead for that sample, starting immediately before the circuit is loaded for routing and ending after all routed metrics are computed.

---

## 8. Exact real-metric definitions

This section is the authority for how the real metrics are computed.

### 8.1 Real PST

#### 8.1.1 Name

The official CSV/summary metric name is:

- `real_pst_gate_readout`

#### 8.1.2 Definition

`real_pst_gate_readout` is the multiplicative estimated success probability of the **final routed executable circuit**.

It must be computed as:

- multiply the success factor of each executable instruction in the routed circuit,
- where success factor = `1 - error_rate(instruction, physical_qubits)`.

This includes:

- 1Q gate errors,
- 2Q gate errors,
- readout errors for `measure` instructions.

This excludes:

- barriers,
- delays,
- metadata-only or non-semantic instructions with no executable error model.

#### 8.1.3 Error-source lookup

Error lookup must follow BackendV2 / Target semantics.

Use the backend target as the primary source of instruction error values.

Rules:

- for a 1Q or 2Q gate named `op_name` applied on physical qubits `q_tuple`, lookup the instruction property on `backend.target[op_name][q_tuple]` and read its `.error` field,
- for measurement on qubit `q`, lookup `backend.target["measure"][(q,)] .error`,
- if the backend reports a valid executable instruction but the error is missing for a non-virtual/non-meta gate, treat this as a metric-computation failure for that sample.

#### 8.1.4 Handling virtual / zero-error operations

For operations that are effectively virtual or modeled with zero error on the backend, the helper may use `0.0` if that is how the backend target reports them.

It must **not** silently invent error values for unknown executable gates.

#### 8.1.5 Value range

For successful rows, `real_pst_gate_readout` must satisfy:

- `0.0 <= real_pst_gate_readout <= 1.0`

---

### 8.2 Exact inserted SWAP count

#### 8.2.1 Name

The official CSV/summary metric name is:

- `swap_inserted_count`

#### 8.2.2 Definition

This metric is the exact number of literal `swap` instructions inserted by the **routing stage**.

#### 8.2.3 Critical implementation rule

This metric must **not** be estimated from the final basis-translated circuit, because the final translated circuit may decompose swaps away.

Instead, the implementation must capture the circuit **at the end of the routing stage and before translation removes literal swaps**.

#### 8.2.4 Required implementation behavior

The routed-eval implementation must therefore extract or preserve a routing-stage circuit snapshot such that:

- `swap_inserted_count = routing_stage_circuit.count_ops().get("swap", 0)`

If that routing-stage snapshot cannot be obtained reliably, the implementation is incomplete.
Using only the post-translation circuit for exact swap count is not acceptable for this revision.

---

### 8.3 Added two-qubit operation count

#### 8.3.1 Name

The official CSV/summary metric name is:

- `added_2q_ops`

#### 8.3.2 Definition

Let:

- `original_2q_count` = number of 2-qubit operations in the original loaded input circuit,
- `routed_2q_count` = number of 2-qubit operations in the final routed executable circuit.

Then:

- `added_2q_ops = routed_2q_count - original_2q_count`

#### 8.3.3 Counting rule

A counted instruction is a two-qubit instruction if:

- `inst.operation.num_qubits == 2`

Measurements are not counted as 2Q operations.

No clamping is applied.

The value is expected to be nonnegative in normal use, but the implementation must record the exact arithmetic result.

---

### 8.4 Depth increase

#### 8.4.1 Official metric names

The evaluation must report all of the following:

- `original_depth`
- `routed_depth`
- `depth_increase_abs`
- `depth_increase_ratio`

#### 8.4.2 Original depth definition

`original_depth` is defined as the depth of the original loaded input circuit using Qiskit depth semantics, excluding `barrier` instructions.

#### 8.4.3 Routed depth definition

`routed_depth` is defined as the depth of the final routed executable circuit using Qiskit depth semantics, excluding `barrier` instructions.

#### 8.4.4 Increase formulas

- `depth_increase_abs = routed_depth - original_depth`
- `depth_increase_ratio = routed_depth / max(original_depth, 1)`

This intentionally measures end-to-end downstream compiled-depth growth relative to the original input circuit.

---

### 8.5 Compile time

#### 8.5.1 Official metric name

- `routing_compile_time_s`

Definition already given in Section 7.4.

---

## 9. Real-metric row schema

When routed evaluation is enabled, each successful row must contain the existing proxy metrics plus the following real routed fields.

### 9.1 Required routed columns

- `routing_attempted`
- `routing_status`
- `routing_compile_time_s`
- `routing_total_eval_time_s`
- `real_pst_gate_readout`
- `swap_inserted_count`
- `original_2q_count`
- `routed_2q_count`
- `added_2q_ops`
- `original_depth`
- `routed_depth`
- `depth_increase_abs`
- `depth_increase_ratio`

### 9.2 Success-row conventions

For successful routed rows:

- `routing_attempted = True`
- `routing_status = "success"`
- all required routed numeric fields are finite

### 9.3 Failure-row conventions

For routed failures:

- `routing_attempted = True`
- `routing_status` must be one of a small fixed set, see Section 13
- routed numeric fields become `NaN` or `None` according to the failure-row policy used elsewhere in evaluation
- the pre-existing `error_type` field remains populated with the Python exception class name when applicable

---

## 10. Summary JSON requirements

The summary JSON must be extended so the real routed metrics are summarized alongside the proxy metrics.

### 10.1 Add these keys to numeric summaries

The summary numeric key list must include:

- `routing_compile_time_s`
- `routing_total_eval_time_s`
- `real_pst_gate_readout`
- `swap_inserted_count`
- `original_2q_count`
- `routed_2q_count`
- `added_2q_ops`
- `original_depth`
- `routed_depth`
- `depth_increase_abs`
- `depth_increase_ratio`

### 10.2 Grouped summaries

These routed metrics must be summarized in:

- overall summary,
- per-source summary,
- per-`k_logical` summary.

### 10.3 Routing summary block

Replace the current placeholder routing summary with a real routing summary block containing at minimum:

- `implemented: true`
- official settings used:
  - backend name
  - routing method
  - optimization level
  - include readout in PST = true
- number of rows with successful routing metrics
- number of rows with routing failures
- routing status breakdown

---

## 11. File-by-file implementation specification

This section is the operational core of the plan.

## 11.1 `src/kmw/evaluation/evaluate.py`

### 11.1.1 Keep existing proxy path intact

Do **not** remove or rewrite the existing inference path. The current sequence:

- soft reindexer forward,
- harden `R_L` and `R_H`,
- reorder tensors,
- mapper forward,
- decode to native frame,
- Hungarian on `S_nat`,
- proxy metrics,

must remain the first part of per-sample evaluation.

### 11.1.2 Extend `EvalConfig`

Add the following fields to `EvalConfig`:

- `project_root: str = "."`
- `backend_name: str = "fake_toronto_v2"`
- `route_final_eval: bool = False`
- `routing_method: str = "sabre"`
- `transpile_optimization_level: int = 0`
- `seed_transpiler: int | None = None`
- `include_readout_in_pst: bool = True`
- `save_routed_qasm_dir: str | None = None`
- `save_routed_qpy_dir: str | None = None`

Do not remove the existing config fields.

### 11.1.3 Remove placeholder-only behavior when real routing is enabled

Current placeholder helper functions may remain for backward compatibility, but when `route_final_eval=True` the row must contain **real routed fields**, not placeholder-only fields.

### 11.1.4 Add required helpers

Add these new helpers to `evaluate.py`.

#### A. Metadata / circuit loading

- `resolve_qasm_path(sample: dict[str, Any], eval_config: EvalConfig) -> Path`
- `load_original_circuit(sample: dict[str, Any], eval_config: EvalConfig) -> QuantumCircuit`

Behavior:

- read `qasm_relpath` from sample metadata,
- resolve absolute path as `Path(eval_config.project_root) / qasm_relpath`,
- load the circuit from disk,
- raise a clear error if the file is missing or unreadable.

#### B. Mapping conversion

- `mapping_matrix_to_assignment(M_map: torch.Tensor) -> list[int]`
- `build_initial_layout_for_circuit(circuit: QuantumCircuit, assignment: list[int]) -> Layout`

Behavior:

- convert row-wise argmax of `M_map` into `assign[u] = j`,
- use only active logical qubits from the loaded circuit,
- validate uniqueness and range.

#### C. Backend resolution

- `resolve_backend_for_eval(backend_name: str) -> BackendV2`

Behavior:

- use the shared backend resolver used by preprocessing if available,
- do not duplicate backend-resolution logic in multiple places if a shared helper already exists,
- support at minimum the current `fake_toronto_v2` path.

#### D. Routed transpilation

- `run_routed_transpile(circuit: QuantumCircuit, backend: BackendV2, initial_layout: Layout, eval_config: EvalConfig) -> RoutedEvalArtifacts`

`RoutedEvalArtifacts` should be a small dataclass containing at least:

- `final_circuit`
- `routing_stage_circuit`
- `compile_time_s`
- `routing_method`
- `optimization_level`

#### E. Counting helpers

- `count_two_qubit_ops(circuit: QuantumCircuit) -> int`
- `compute_circuit_depth_no_barrier(circuit: QuantumCircuit) -> int`
- `extract_swap_inserted_count(routing_stage_circuit: QuantumCircuit) -> int`

#### F. Backend-target error lookup

- `lookup_instruction_error(backend: BackendV2, op_name: str, qargs: tuple[int, ...]) -> float`

Behavior:

- use backend target instruction properties,
- support measure on `(q,)`,
- raise a clear error when an executable instruction has no valid error model.

#### G. Real PST

- `estimate_real_pst_gate_readout(circuit: QuantumCircuit, backend: BackendV2, include_readout: bool = True) -> float`

#### H. Routed metric aggregation

- `compute_real_routed_metrics(original_circuit: QuantumCircuit, final_circuit: QuantumCircuit, routing_stage_circuit: QuantumCircuit, backend: BackendV2, eval_config: EvalConfig, compile_time_s: float) -> dict[str, Any]`

This function must return exactly the routed fields defined in Section 9.

### 11.1.5 Integrate routed evaluation into `run_single_sample_inference()`

Current behavior:

- compute proxy metrics,
- return row.

Required new behavior:

- compute the row exactly as today first,
- if `eval_config.route_final_eval is False`, return unchanged behavior,
- if `eval_config.route_final_eval is True`, then:
  1. load original circuit,
  2. resolve backend,
  3. build initial layout from `M_map`,
  4. run transpilation/routing,
  5. compute real routed metrics,
  6. merge them into the row,
  7. optionally save routed circuit artifacts.

### 11.1.6 Optional artifact saving

If `save_routed_qasm_dir` or `save_routed_qpy_dir` is set, write per-circuit routed outputs using the circuit id as filename stem.

This is optional for the first implementation pass, but the config hooks must exist.

### 11.1.7 Update CSV column policy

Update the minimum column policy so that when routed evaluation is enabled the routed fields are part of the standard ordered output.

Do **not** keep the old placeholder columns as the official routed output in routed mode.

### 11.1.8 Update summary policy

Replace the placeholder routing summary block with the real block described in Section 10 when routed evaluation is enabled.

---

## 11.2 `src/kmw/cli/main.py`

### 11.2.1 Keep the CLI thin

No routing logic or metric formulas should be implemented in the CLI.
The CLI remains a config/dispatch layer only.

### 11.2.2 Extend eval arguments

Add these eval CLI arguments:

- `--route-final-eval` (flag)
- `--routing-method` (default `sabre`)
- `--transpile-optimization-level` (default `0`)
- `--seed-transpiler` (default `None`; if omitted, use `args.seed` inside config build)
- `--save-routed-qasm-dir` (optional path)
- `--save-routed-qpy-dir` (optional path)

Do **not** add a CLI flag for excluding readout in the official setting for this revision. Readout must be included by default and remain the official behavior.

### 11.2.3 Extend `build_eval_config(args)`

Populate these additional config fields:

- `cfg.route_final_eval = args.route_final_eval`
- `cfg.routing_method = args.routing_method`
- `cfg.transpile_optimization_level = args.transpile_optimization_level`
- `cfg.seed_transpiler = args.seed_transpiler if args.seed_transpiler is not None else args.seed`
- `cfg.save_routed_qasm_dir = args.save_routed_qasm_dir`
- `cfg.save_routed_qpy_dir = args.save_routed_qpy_dir`
- `cfg.include_readout_in_pst = True`
- `cfg.project_root = str(Path.cwd())`

### 11.2.4 Pass backend identity into eval config

After the dataset is built in `cmd_eval`, set:

- `eval_config.backend_name = dataset.backend_name`

This prevents `evaluate.py` from guessing backend identity implicitly.

### 11.2.5 Placeholder-column policy

When `--route-final-eval` is enabled, the CLI/config should set:

- `include_routing_placeholders_in_csv = False`

because real routed columns are now available.

---

## 11.3 `src/kmw/data/dataset.py`

### 11.3.1 No semantic data-path change required

No mandatory semantic change is required in `dataset.py` for this revision.

The current metadata already includes:

- `id`
- `source`
- `split`
- `qasm_relpath`
- manifest contents
- backend metadata

which is sufficient for routed evaluation.

### 11.3.2 Allowed optional cleanup

An optional cleanup is allowed:

- expose `project_root` or resolved qasm path in metadata for convenience.

But this is **not required** if `EvalConfig.project_root` is passed explicitly from CLI.

---

## 11.4 Shared backend helper location

If backend resolution is currently implemented in preprocessing/pipeline code, routed evaluation must reuse that resolver.

If no reusable public resolver exists yet, add one in a shared place such as:

- `src/kmw/preprocessing/pipeline.py`

and import it from both preprocessing and evaluation.

Do **not** create two divergent backend resolvers.

---

## 12. Routing-stage SWAP capture requirement

This is one of the most important implementation rules in the entire document.

### 12.1 Problem

The final backend-translated circuit may not contain literal `swap` instructions, because translation can decompose them into basis gates.

### 12.2 Required behavior

The implementation must capture the circuit **after the routing stage but before translation destroys literal swap gates**.

### 12.3 Acceptable implementation strategies

Any of the following are acceptable if implemented correctly and reproducibly:

1. run the staged pass manager in stage-wise fashion and keep the routing-stage output circuit,
2. instrument the pass-manager execution so the routing-stage output circuit is captured before translation,
3. construct a functionally equivalent two-pass transpilation procedure where routing-stage swap instructions are preserved for counting and the final official executable circuit is still produced from the backend-targeted path.

### 12.4 Not acceptable

These are **not** acceptable:

- estimating exact swap count only from final `count_ops()` on the executable circuit,
- inferring swap count from `added_2q_ops`,
- inferring swap count from final layout permutation alone.

---

## 13. Failure handling policy

The current evaluation code already supports failure rows and `fail_fast` behavior. That policy must be preserved.

### 13.1 Allowed routed status values

Lock the routed status vocabulary to:

- `success`
- `circuit_load_error`
- `layout_build_error`
- `backend_resolve_error`
- `transpile_error`
- `routing_capture_error`
- `pst_compute_error`
- `metric_compute_error`

Do not use free-form status strings.

### 13.2 Failure-row rules

If routed evaluation fails for a sample and `fail_fast=False`:

- still emit a row,
- keep the existing proxy metrics if they were already computed,
- mark routed fields as missing / NaN,
- set `routing_attempted = True`,
- set `routing_status` to the locked code,
- preserve `error_type`.

### 13.3 Hard-fail mode

If `fail_fast=True`, the first routed-eval exception must still raise immediately.

---

## 14. Testing requirements

Implementation is not complete without tests.

## 14.1 Required tests

Add tests under `tests/`.

### A. Unit tests

1. **assignment-to-layout test**
   - given a hard permutation matrix, verify the produced assignment and `initial_layout` are correct.

2. **metric-count helper test**
   - verify 2Q counting and depth counting rules on a tiny synthetic circuit.

3. **PST helper test**
   - verify the PST helper multiplies instruction success factors correctly when mock instruction errors are provided.

### B. Integration tests

4. **routed eval one-sample integration test**
   - run `evaluate_model()` with routed evaluation enabled on a tiny sample and confirm that routed fields appear.

5. **CSV schema test**
   - confirm routed-mode CSV contains real routed columns and not only placeholders.

6. **summary test**
   - confirm the summary JSON includes the routed numeric metrics.

### C. Regression test

7. **proxy-only compatibility test**
   - confirm evaluation still works when `route_final_eval=False`.

---

## 15. Acceptance criteria

This revision is acceptable only if **all** of the following are true.

### 15.1 Functional

- the full-run evaluation can run on full manifests,
- `--route-final-eval` triggers real routed evaluation,
- proxy-only evaluation still works when the flag is absent,
- routed evaluation uses the model’s hard mapping as the transpiler initial layout.

### 15.2 Metric completeness

Each successful routed row includes all of:

- real PST with readout,
- exact inserted swap count,
- added 2Q op count,
- compile time,
- original and routed depth,
- absolute and relative depth increase.

### 15.3 Numerical sanity

For successful routed rows:

- `0 <= real_pst_gate_readout <= 1`
- `swap_inserted_count >= 0`
- `routing_compile_time_s >= 0`
- `original_depth >= 0`
- `routed_depth >= 0`
- all routed numeric fields are finite.

### 15.4 Architectural cleanliness

- training code is unchanged,
- mapper architecture is unchanged,
- dataset tensor semantics are unchanged,
- CLI remains thin,
- routed logic is centered in `evaluate.py`.

### 15.5 Reproducibility

Given the same:

- checkpoint,
- manifest,
- backend,
- routing method,
- optimization level,
- transpiler seed,

routed evaluation is reproducible at the metric level.

---

## 16. Official CLI shape after implementation

The official final-eval command should look like this structurally:

```bash
python -m kmw.cli.main eval \
  --manifest "$PWD/data/manifests_full/test.jsonl" \
  --batch-size 1 \
  --num-workers 0 \
  --device cuda \
  --checkpoint "$PWD/checkpoints/kmw_full_epoch_xxxx.pt" \
  --eval-split-name full_test \
  --per-circuit-csv "$PWD/artifacts/eval/full_test_metrics.csv" \
  --summary-json "$PWD/artifacts/eval/full_test_summary.json" \
  --route-final-eval \
  --routing-method sabre \
  --transpile-optimization-level 0 \
  --seed-transpiler 20260323
```

This is the official research-eval mode for this revision.

---

## 17. Implementation order

Implement in this exact order.

### Phase 1 — config and CLI wiring

1. extend `EvalConfig`
2. add new eval CLI flags
3. pass `project_root` and `backend_name` into eval config

### Phase 2 — routed helpers in evaluation

4. circuit loading helper
5. backend resolver hookup
6. assignment-to-layout helper
7. depth / 2Q counting helpers
8. instruction-error lookup helper
9. PST helper

### Phase 3 — transpilation path

10. official routed transpile helper
11. routing-stage swap-capture path
12. returned routed artifact bundle

### Phase 4 — row integration

13. extend `run_single_sample_inference()`
14. merge routed metrics into rows
15. update failure-row behavior

### Phase 5 — report integration

16. update CSV schema
17. update summary keys
18. replace routing placeholder summary with real summary when enabled

### Phase 6 — tests

19. unit tests
20. integration tests
21. proxy-only regression test

---

## 18. Final one-paragraph execution summary

Keep the current KMW model exactly as an initial mapper. During final evaluation only, take the hard native-frame mapping produced by the current inference pipeline, convert it into a Qiskit `initial_layout`, transpile the original circuit against the BackendV2 target using `optimization_level=0` and `routing_method="sabre"`, capture the routing-stage circuit to count exact inserted swaps, compute the final executable circuit’s gate+readout success estimate from backend target instruction errors, compute compile time / added 2Q ops / depth increase, and write these real routed metrics beside the existing proxy metrics into the per-circuit CSV and summary JSON.
