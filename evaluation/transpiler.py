"""Custom transpiler (PassManager builder) for GraphQMap evaluation.

Supports layout_method × routing_method combinations with per-stage timing.
Follows the MQM project's build_transpiler() pattern.

Supported layout methods: graphqmap, sabre, dense, noise_adaptive, trivial, vf2
Supported routing methods: sabre, nassc
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from math import pi
from time import perf_counter_ns
from typing import Any

from qiskit import QuantumCircuit
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.circuit.library.standard_gates import (
    CXGate,
    CZGate,
    ECRGate,
    RXXGate,
    iSwapGate,
)
from qiskit.passmanager.flow_controllers import (
    ConditionalController,
    DoWhileController,
)
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.synthesis.one_qubit import one_qubit_decompose
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler import passes

from evaluation.prev_methods.nassc import NASSCSwap
from evaluation.prev_methods.noise_adaptive import NoiseAdaptiveLayout

logger = logging.getLogger(__name__)


def _choose_kak_gate(basis_gates: list[str]):
    """Choose the first available 2q gate for KAK decomposition."""
    kak_gate_names = {
        "cx": CXGate(),
        "cz": CZGate(),
        "iswap": iSwapGate(),
        "rxx": RXXGate(pi / 2),
        "ecr": ECRGate(),
    }
    kak_gates = set(basis_gates or []).intersection(kak_gate_names.keys())
    if kak_gates:
        return kak_gate_names[kak_gates.pop()]
    return None


def _choose_euler_basis(basis_gates: list[str]):
    """Choose the first available 1q basis for Euler decomposition."""
    basis_set = set(basis_gates or [])
    for basis, gates in one_qubit_decompose.ONE_QUBIT_EULER_BASIS_GATES.items():
        if set(gates).issubset(basis_set):
            return basis
    return None


def build_transpiler(
    backend: Any,
    initial_layout: dict | list | None = None,
    layout_method: str = "sabre",
    routing_method: str = "sabre",
    seed: int = 43,
    optimization_level: int = 3,
) -> PassManager:
    """Build a customizable PassManager for transpilation.

    Args:
        backend: FakeBackendV2 instance.
        initial_layout: Pre-computed layout (used when layout_method='given').
        layout_method: One of 'sabre', 'dense', 'noise_adaptive', 'trivial', 'vf2', 'given'.
                       Use 'given' with initial_layout for GraphQMap model layouts.
        routing_method: One of 'sabre', 'nassc'.
        seed: Random seed for transpiler.
        optimization_level: Qiskit optimization level (2 or 3).

    Returns:
        Configured PassManager.
    """
    basis_gates = backend.configuration().basis_gates
    coupling_map = CouplingMap(backend.configuration().coupling_map)
    target = backend.target

    # --- Pass definitions ---
    _unroll3q = [passes.Unroll3qOrMore()]
    _given_layout = [passes.SetLayout(initial_layout)] if initial_layout else []

    def _choose_layout_condition(property_set):
        return not property_set["layout"]

    _choose_layout_0 = [
        passes.TrivialLayout(coupling_map),
        passes.Layout2qDistance(coupling_map, property_name="trivial_layout_score"),
    ]

    def _trivial_not_perfect(property_set):
        if property_set["trivial_layout_score"] is not None:
            if property_set["trivial_layout_score"] != 0:
                property_set["layout"] = None
                return True
        return False

    try:
        _choose_layout_1 = [
            passes.CSPLayout(coupling_map, call_limit=1000, time_limit=60, seed=seed)
        ]
    except Exception:
        # CSPLayout requires optional python-constraint library
        _choose_layout_1 = []

    def _csp_not_found_match(property_set):
        if property_set["layout"] is None:
            return True
        if (
            property_set.get("CSPLayout_stop_reason") is not None
            and property_set["CSPLayout_stop_reason"] != "solution found"
        ):
            return True
        return False

    _choose_layout_2 = [
        passes.SabreLayout(
            coupling_map=coupling_map,
            max_iterations=2,
            seed=seed,
            layout_trials=10,
            skip_routing=True,
        )
    ]

    _embed = [
        passes.FullAncillaAllocation(coupling_map=coupling_map),
        passes.EnlargeWithAncilla(),
        passes.ApplyLayout(),
    ]

    _swap_check = [passes.CheckMap(coupling_map=coupling_map)]

    def _swap_condition(property_set):
        return not property_set["is_swap_mapped"]

    # Routing method selection
    if routing_method == "sabre":
        _swap_method = passes.SabreSwap(
            coupling_map, heuristic="decay", seed=seed, fake_run=False,
        )
    elif routing_method == "nassc":
        _swap_method = NASSCSwap(
            coupling_map=coupling_map,
            heuristic="decay",
            enable_factor_block=True,
            enable_factor_commute_0=True,
            enable_factor_commute_1=True,
            factor_block=1,
            factor_commute_0=1,
            factor_commute_1=1,
            decomposer2q=TwoQubitBasisDecomposer(
                _choose_kak_gate(basis_gates),
                euler_basis=_choose_euler_basis(basis_gates),
            ),
            fake_run=False,
            seed=seed,
            approximation_degree=1.0,
        )
    else:
        raise ValueError(f"Unknown routing_method: {routing_method}")

    _swap = [passes.BarrierBeforeFinalMeasurements(), _swap_method]

    _unroll = [
        passes.UnrollCustomDefinitions(SessionEquivalenceLibrary, basis_gates),
        passes.BasisTranslator(SessionEquivalenceLibrary, basis_gates),
    ]
    _depth_check = [passes.Depth(), passes.FixedPoint("depth")]
    _reset = [passes.RemoveResetInZeroState()]
    _meas = [
        passes.OptimizeSwapBeforeMeasure(),
        passes.RemoveDiagonalGatesBeforeMeasure(),
    ]

    def _opt_control(property_set):
        return not property_set["depth_fixed_point"]

    _opt = [
        passes.Collect2qBlocks(),
        passes.ConsolidateBlocks(basis_gates=basis_gates),
        passes.UnitarySynthesis(
            basis_gates,
            approximation_degree=1.0,
            coupling_map=coupling_map,
            backend_props=backend.properties(),
        ),
        passes.Optimize1qGatesDecomposition(basis_gates),
        passes.CommutativeCancellation(),
    ]

    _opt_before_routing = [
        passes.Collect2qBlocks(),
        passes.ConsolidateBlocks(basis_gates=basis_gates),
        passes.UnitarySynthesis(
            basis_gates,
            approximation_degree=1.0,
            coupling_map=coupling_map,
            backend_props=backend.properties(),
        ),
        passes.Optimize1qGatesDecomposition(basis_gates),
    ]

    _direction_check = [passes.CheckGateDirection(coupling_map)]

    def _direction_condition(property_set):
        return not property_set["is_direction_mapped"]

    _direction = [passes.GateDirection(coupling_map)]

    instruction_durations = (
        target._instruction_durations
        if hasattr(target, "_instruction_durations")
        else None
    )
    granularity = (
        target.timing_constraints().granularity
        if hasattr(target, "timing_constraints")
        else 1
    )
    min_length = (
        target.timing_constraints().min_length
        if hasattr(target, "timing_constraints")
        else 1
    )
    acquire_alignment = (
        target.timing_constraints().acquire_alignment
        if hasattr(target, "timing_constraints")
        else 1
    )
    _sched = [
        passes.TimeUnitConversion(instruction_durations),
        passes.ValidatePulseGates(granularity=granularity, min_length=min_length),
        passes.AlignMeasures(alignment=acquire_alignment),
    ]

    # --- Build PassManager ---
    pm = PassManager()

    # Initial unroll
    pm.append(_unroll3q)
    pm.append(_reset + _meas)

    # Layout
    if initial_layout is not None:
        pm.append(_given_layout)
    elif layout_method == "sabre":
        pm.append(
            passes.SabreLayout(
                coupling_map=coupling_map,
                max_iterations=10,
                seed=seed,
                layout_trials=20,
                skip_routing=True,
            )
        )
    elif layout_method == "noise_adaptive":
        pm.append(
            NoiseAdaptiveLayout(
                backend.properties(),
                coupling_map,
            )
        )
    elif layout_method == "dense":
        pm.append(
            passes.DenseLayout(
                coupling_map,
                backend_prop=backend.properties(),
                target=target,
            )
        )
    elif layout_method == "trivial":
        pm.append(passes.TrivialLayout(coupling_map))
        pm.append(
            passes.Layout2qDistance(
                coupling_map, property_name="trivial_layout_score"
            )
        )
    elif layout_method == "vf2":
        from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason

        _ly = [passes.VF2Layout(coupling_map, call_limit=1000, seed=seed)]

        def _vf2_not_found_match(property_set):
            return (
                property_set["VF2Layout_stop_reason"]
                == VF2LayoutStopReason.no_matching_subgraph
            )

        pm.append(_ly)
        pm.append(ConditionalController(_ly, condition=_vf2_not_found_match))
    elif layout_method == "given":
        pass  # initial_layout already set above
    else:
        # Default fallback: Trivial → CSP → SABRE
        pm.append(
            ConditionalController(
                _choose_layout_0, condition=_choose_layout_condition
            )
        )
        pm.append(
            ConditionalController(
                _choose_layout_1, condition=_trivial_not_perfect
            )
        )
        pm.append(
            ConditionalController(
                _choose_layout_2, condition=_csp_not_found_match
            )
        )

    # Embed + Route
    pm.append(_embed)
    pm.append(_swap_check)
    if routing_method == "nassc":
        pm.append(_unroll + _opt_before_routing)
    pm.append(ConditionalController(_swap, condition=_swap_condition))

    # Optimization
    pm.append(_unroll)
    if routing_method == "nassc":
        pm.append([passes.CommutativeCancellation()])
    if coupling_map and not coupling_map.is_symmetric:
        pm.append(_direction_check)
        pm.append(
            ConditionalController(_direction, condition=_direction_condition)
        )
    pm.append(_reset)
    pm.append(
        DoWhileController(_depth_check + _opt + _unroll, do_while=_opt_control)
    )

    # Scheduling
    pm.append(_sched)

    return pm


class _ElapsedTimer:
    """Simple timer for measuring elapsed time."""

    def __init__(self):
        self.start = 0
        self.end = 0

    @property
    def elapsed_ms(self) -> float:
        return (self.end - self.start) / 1e6


@contextmanager
def _timed(timer: _ElapsedTimer):
    timer.start = perf_counter_ns()
    yield
    timer.end = perf_counter_ns()


def transpile_with_timing(
    circuit: QuantumCircuit,
    backend: Any,
    initial_layout: dict | list | None = None,
    layout_method: str = "sabre",
    routing_method: str = "sabre",
    seed: int = 43,
    optimization_level: int = 3,
) -> tuple[QuantumCircuit, dict[str, Any]]:
    """Transpile a circuit and return timing metadata.

    Uses staged PassManagers for per-stage timing measurement,
    following MQM's transpile_and_run() pattern.

    Args:
        circuit: Quantum circuit to transpile.
        backend: FakeBackendV2 instance.
        initial_layout: Pre-computed layout (for GraphQMap model).
        layout_method: Layout method name.
        routing_method: Routing method name.
        seed: Random seed.
        optimization_level: Optimization level.

    Returns:
        Tuple of (transpiled_circuit, metadata_dict).
        metadata_dict has keys: layout_time, routing_time, optimization_time,
        scheduling_time, total_time, depth, map_depth, map_cx.
    """
    basis_gates = backend.configuration().basis_gates
    coupling_map = CouplingMap(backend.configuration().coupling_map)
    target = backend.target

    # Build individual stage pass managers
    init_pm = PassManager()
    layout_pm = PassManager()
    routing_pm = PassManager()
    optimization_pm = PassManager()
    scheduling_pm = PassManager()

    # --- Init stage ---
    init_pm.append([passes.Unroll3qOrMore()])
    init_pm.append([passes.RemoveResetInZeroState()])
    init_pm.append([passes.OptimizeSwapBeforeMeasure()])
    init_pm.append([passes.RemoveDiagonalGatesBeforeMeasure()])

    # --- Layout stage ---
    if initial_layout is not None:
        layout_pm.append([passes.SetLayout(initial_layout)])
    elif layout_method == "sabre":
        layout_pm.append(
            passes.SabreLayout(
                coupling_map=coupling_map,
                max_iterations=10,
                seed=seed,
                layout_trials=20,
                skip_routing=True,
            )
        )
    elif layout_method == "noise_adaptive":
        layout_pm.append(
            NoiseAdaptiveLayout(backend.properties(), coupling_map)
        )
    elif layout_method == "dense":
        layout_pm.append(
            passes.DenseLayout(
                coupling_map,
                backend_prop=backend.properties(),
                target=target,
            )
        )
    elif layout_method == "trivial":
        layout_pm.append(passes.TrivialLayout(coupling_map))
    elif layout_method == "given":
        pass

    layout_pm.append(
        [
            passes.FullAncillaAllocation(coupling_map=coupling_map),
            passes.EnlargeWithAncilla(),
            passes.ApplyLayout(),
        ]
    )

    # --- Routing stage ---
    routing_pm.append([passes.CheckMap(coupling_map=coupling_map)])

    def _swap_condition(property_set):
        return not property_set["is_swap_mapped"]

    if routing_method == "nassc":
        _unroll_passes = [
            passes.UnrollCustomDefinitions(
                SessionEquivalenceLibrary, basis_gates
            ),
            passes.BasisTranslator(SessionEquivalenceLibrary, basis_gates),
        ]
        _opt_before = [
            passes.Collect2qBlocks(),
            passes.ConsolidateBlocks(basis_gates=basis_gates),
            passes.UnitarySynthesis(
                basis_gates,
                approximation_degree=1.0,
                coupling_map=coupling_map,
                backend_props=backend.properties(),
            ),
            passes.Optimize1qGatesDecomposition(basis_gates),
        ]
        routing_pm.append(_unroll_passes + _opt_before)

        _swap_method = NASSCSwap(
            coupling_map=coupling_map,
            heuristic="decay",
            enable_factor_block=True,
            enable_factor_commute_0=True,
            enable_factor_commute_1=True,
            factor_block=1,
            factor_commute_0=1,
            factor_commute_1=1,
            decomposer2q=TwoQubitBasisDecomposer(
                _choose_kak_gate(basis_gates),
                euler_basis=_choose_euler_basis(basis_gates),
            ),
            fake_run=False,
            seed=seed,
            approximation_degree=1.0,
        )
    else:
        _swap_method = passes.SabreSwap(
            coupling_map, heuristic="decay", seed=seed, fake_run=False,
        )

    _swap = [passes.BarrierBeforeFinalMeasurements(), _swap_method]
    routing_pm.append(ConditionalController(_swap, condition=_swap_condition))

    # --- Optimization stage ---
    _unroll = [
        passes.UnrollCustomDefinitions(SessionEquivalenceLibrary, basis_gates),
        passes.BasisTranslator(SessionEquivalenceLibrary, basis_gates),
    ]
    optimization_pm.append(_unroll)
    if routing_method == "nassc":
        optimization_pm.append([passes.CommutativeCancellation()])
    if coupling_map and not coupling_map.is_symmetric:
        _dir_check = [passes.CheckGateDirection(coupling_map)]

        def _direction_condition(property_set):
            return not property_set["is_direction_mapped"]

        _dir = [passes.GateDirection(coupling_map)]
        optimization_pm.append(_dir_check)
        optimization_pm.append(
            ConditionalController(_dir, condition=_direction_condition)
        )
    optimization_pm.append([passes.RemoveResetInZeroState()])

    _depth_check = [passes.Depth(), passes.FixedPoint("depth")]

    def _opt_control(property_set):
        return not property_set["depth_fixed_point"]

    _opt = [
        passes.Collect2qBlocks(),
        passes.ConsolidateBlocks(basis_gates=basis_gates),
        passes.UnitarySynthesis(
            basis_gates,
            approximation_degree=1.0,
            coupling_map=coupling_map,
            backend_props=backend.properties(),
        ),
        passes.Optimize1qGatesDecomposition(basis_gates),
        passes.CommutativeCancellation(),
    ]
    optimization_pm.append(
        DoWhileController(_depth_check + _opt + _unroll, do_while=_opt_control)
    )

    # --- Scheduling stage ---
    instruction_durations = (
        target._instruction_durations
        if hasattr(target, "_instruction_durations")
        else None
    )
    granularity = (
        target.timing_constraints().granularity
        if hasattr(target, "timing_constraints")
        else 1
    )
    min_length = (
        target.timing_constraints().min_length
        if hasattr(target, "timing_constraints")
        else 1
    )
    acquire_alignment = (
        target.timing_constraints().acquire_alignment
        if hasattr(target, "timing_constraints")
        else 1
    )
    scheduling_pm.append(
        [
            passes.TimeUnitConversion(instruction_durations),
            passes.ValidatePulseGates(
                granularity=granularity, min_length=min_length
            ),
            passes.AlignMeasures(alignment=acquire_alignment),
        ]
    )

    # --- Run with timing ---
    init_timer = _ElapsedTimer()
    layout_timer = _ElapsedTimer()
    routing_timer = _ElapsedTimer()
    optimization_timer = _ElapsedTimer()
    scheduling_timer = _ElapsedTimer()
    total_timer = _ElapsedTimer()

    with _timed(total_timer):
        with _timed(init_timer):
            icirc = init_pm.run(circuit)
        with _timed(layout_timer):
            lcirc = layout_pm.run(icirc)
        with _timed(routing_timer):
            rcirc = routing_pm.run(lcirc)
        with _timed(optimization_timer):
            ocirc = optimization_pm.run(rcirc)
        with _timed(scheduling_timer):
            final_circ = scheduling_pm.run(ocirc)

    # Collect 2q gate count after routing (before optimization)
    map_cx = 0
    for name in ("cx", "ecr", "cz"):
        count = rcirc.count_ops().get(name, 0)
        if count > 0:
            map_cx = count
            break

    metadata = {
        "init_time": init_timer.elapsed_ms,
        "layout_time": layout_timer.elapsed_ms,
        "routing_time": routing_timer.elapsed_ms,
        "optimization_time": optimization_timer.elapsed_ms,
        "scheduling_time": scheduling_timer.elapsed_ms,
        "total_time": total_timer.elapsed_ms,
        "depth": final_circ.depth(),
        "map_depth": rcirc.depth(),
        "map_cx": map_cx,
    }
    return final_circ, metadata
