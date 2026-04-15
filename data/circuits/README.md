# GraphQMap Dataset

Quantum circuit dataset for training and evaluating the GraphQMap qubit mapping model.

## Overview

| | Circuits | Labels |
|---|:---:|:---:|
| **Total** | **7,165** | **4,269** |
| QUEKO | 900 | 540 |
| MLQD | 4,443 | 3,729 |
| MQT Bench | 1,448 | — |
| QASMBench | 111 | — |
| RevLib | 263 | — |

## Directory Structure

```
data/circuits/
├── qasm/                           # All .qasm circuit files (OPENQASM 2.0)
│   ├── queko/                      # QUEKO benchmark circuits
│   ├── mlqd/                       # MLQD (Machine Learning Quantum Dataset)
│   ├── mqt_bench/                  # MQT Bench generated circuits
│   ├── qasmbench/                  # QASMBench circuits
│   └── revlib/                     # RevLib reversible circuits
├── labels/                         # Initial layout labels
│   ├── queko/labels.json
│   └── mlqd/labels.json
├── backends/                       # Synthetic backend noise profiles
│   ├── queko_aspen4.json
│   ├── queko_tokyo.json
│   ├── queko_rochester.json
│   ├── queko_sycamore.json
│   └── mlqd_grid5x5.json
├── splits/                         # Train/val split definitions
│   ├── train_all.json
│   ├── val.json
│   └── val_queko_only.json
└── README.md                       # This file
```

## Dataset Details

### 1. QUEKO (900 circuits)

**Source**: [QUEKO-benchmark](https://github.com/tbcdebug/QUEKO-benchmark)

Artificially generated circuits with known optimal qubit mappings (τ⁻¹ = 0, meaning zero SWAP overhead). Three categories:

| Category | Circuits | Labels | Description |
|---|:---:|:---:|---|
| BNTF | 180 | 180 | Near-term feasible (depth 5-45) |
| BSS | 360 | 360 | Scalability study (depth 100-900) |
| BIGD | 360 | 0 | Gate density impact study (no solutions provided) |

**Hardware topologies** (4 backends, not available as Qiskit FakeBackendV2):
- `queko_aspen4`: Rigetti Aspen-4 (16 qubits, 18 edges)
- `queko_tokyo`: IBM Tokyo (20 qubits, 43 edges)
- `queko_rochester`: IBM Rochester (53 qubits, 58 edges)
- `queko_sycamore`: Google Sycamore (54 qubits, 88 edges)

**Label format**: `layout[i]` = physical qubit that logical qubit `i` is mapped to. This mapping guarantees zero SWAP overhead on the target topology.

### 2. MLQD (4,443 circuits)

**Source**: [MLQD](https://github.com/WanHsuanLin/MLQD) (Machine Learning Quantum Dataset)

Circuits derived from QASMBench, processed through OLSQ2 optimal qubit mapper on 5 hardware backends. Labels extracted by reverse-engineering SWAP patterns from OLSQ2 result circuits.

| Backend | Circuits | Labels | Noise Source |
|---|:---:|:---:|---|
| Aspen-4 (16Q) | 943 | 779 | Synthetic (`queko_aspen4.json`) |
| Grid 5x5 (25Q) | 992 | 879 | Synthetic (`mlqd_grid5x5.json`) |
| Melbourne (15Q) | 797 | 666 | Qiskit `FakeMelbourneV2` |
| Rochester (53Q) | 881 | 677 | Qiskit `FakeRochesterV2` |
| Sycamore (54Q) | 830 | 728 | Synthetic (`queko_sycamore.json`) |

**Label extraction method**: For each OLSQ2 result circuit:
1. Parse measurement lines → final mapping (logical → physical after all SWAPs)
2. Detect SWAP patterns (3-CNOT decomposition: `cx a,b; cx b,a; cx a,b`)
3. Reverse SWAPs chronologically → recover initial layout
4. Circuits where SWAP count doesn't match detected patterns are kept unlabeled

### 3. MQT Bench (1,448 circuits)

**Source**: [MQT Bench](https://github.com/munich-quantum-toolkit/bench) (Python API, v2.1.1)

Generated via `scripts/generate_mqt_bench.py` with 29 algorithm types across 2-127 qubits. No labels (unsupervised only).

**Algorithm types**: ae, bmw_quark_cardinality, bmw_quark_copula, bv, cdkm_ripple_carry_adder, dj, draper_qft_adder, full_adder, ghz, graphstate, grover, half_adder, hhl, modular_adder, multiplier, qaoa, qft, qftentangled, qnn, qpeexact, qpeinexact, qwalk, randomcircuit, rg_qft_multiplier, vbe_ripple_carry_adder, vqe_real_amp, vqe_su2, vqe_two_local, wstate

**Variants**: VQE circuits have reps=1/2/3, QAOA has seed=0/1/2. Per-circuit 30s timeout to skip excessively large circuits.

### 4. QASMBench (111 circuits)

**Source**: [QASMBench](https://github.com/pnnl/QASMBench)

Filtered to 2Q-127Q. Includes small (43), medium (25), and large (43) categories. No labels.

**Naming**: `{size}_{directory}_{filename}.qasm` (e.g., `small_ghz_n4_ghz_n4.qasm`)

### 5. RevLib (263 circuits)

**Source**: [RevLib](http://www.informatik.uni-bremen.de/rev_lib/) via [Real2QASM](https://github.com/changkyu-u/Real2QASM)

Reversible circuits converted from `.real` format to `.qasm` without basis-gate unrolling (preserves `cx`, `ccx` etc.). Filtered to ≤127Q.

## Label Format

Labels are stored as JSON in `labels/{source}/labels.json`:

```json
{
  "circuit_filename.qasm": {
    "backend": "backend_name",
    "layout": [5, 13, 1, 9, 14, ...]
  }
}
```

- `backend`: Name of the hardware backend this layout is optimal for
- `layout`: List where `layout[i]` = physical qubit assigned to logical qubit `i`

## Synthetic Backend Noise Profiles

For hardware topologies not available as Qiskit FakeBackendV2, synthetic noise profiles are stored in `backends/`:

```json
{
  "backend_name": "queko_aspen4",
  "num_qubits": 16,
  "coupling_map": [[0, 1], [1, 2], ...],
  "qubit_properties": {
    "0": {"t1": 0.000236, "t2": 3.17e-05, "frequency": 5.01e+09, "readout_error": 0.087, "sq_gate_error": 7.1e-05},
    ...
  },
  "edge_properties": {
    "(0, 1)": {"cx_error": 0.228, "cx_duration": 4.46e-07},
    ...
  }
}
```

Noise values are sampled from clipped normal distributions fitted to 11 real Qiskit FakeBackendV2 hardware. Generated once with fixed seed for reproducibility.

## Splits

Split files in `splits/` define which circuits participate in each training stage:

| File | Entries | Description |
|---|:---:|---|
| `val.json` | 28 | Labeled validation set |
| `val_queko_only.json` | 2 | QUEKO validation |
| `train_all.json` | 969 | All training circuits (surrogate loss) |

**Split format**:
```json
[
  {"source": "queko", "file": "16QBT_05CYC_TFL_0.qasm"},
  {"source": "mqt_bench", "file": "ghz_n10.qasm"},
  ...
]
```

## Hardware Backends

### Training Backends (60 total)

**55 Qiskit FakeBackendV2**:
- 5Q (15): Athens, Belem, Bogota, Burlington, Essex, Lima, London, Manila, Ourense, Quito, Rome, Santiago, Valencia, Vigo, Yorktown
- 7Q (6): Casablanca, Jakarta, Lagos, Nairobi, Oslo, Perth
- 15-16Q (2): Melbourne, Guadalupe
- 20Q (5): Almaden, Boeblingen, Johannesburg, Poughkeepsie, Singapore
- 27-28Q (12): Algiers, Auckland, Cairo, Cambridge, Geneva, Hanoi, Kolkata, Montreal, Mumbai, Paris, Peekskill, Sydney
- 33Q (1): Prague
- 53Q (1): Rochester
- 65Q (1): Manhattan
- 127Q (9): Brisbane, Cusco, Kawasaki, Kyiv, Kyoto, Osaka, Quebec, Sherbrooke, Washington

**5 Synthetic backends** (JSON in `backends/`):
- queko_aspen4 (16Q), queko_tokyo (20Q), queko_rochester (53Q), queko_sycamore (54Q), mlqd_grid5x5 (25Q)

### Test Backends (UNSEEN — excluded from training)
- FakeToronto (27Q)
- FakeBrooklyn (65Q)
- FakeTorino (133Q)

## Reproduction Scripts

| Script | Purpose |
|---|---|
| `scripts/generate_queko_noise.py` | Generate synthetic noise profiles for QUEKO backends |
| `scripts/process_mlqd.py` | Process MLQD: copy circuits + extract labels from OLSQ2 results |
| `scripts/generate_mqt_bench.py` | Generate MQT Bench circuits via Python API |

## Citation

If you use this dataset, please cite the original sources:

- **QUEKO**: Tan & Cong, "Optimality Study of Existing Quantum Computing Layout Synthesis Tools", IEEE TC 2020
- **MLQD/MLQM**: "MLQM: Machine Learning Approach for Accelerating Optimal Qubit Mapping", Future Generation Computer Systems 2025
- **MQT Bench**: Quetschlich et al., "MQT Bench: Benchmarking Software and Design Automation Tools for Quantum Computing", Quantum 2023
- **QASMBench**: Li et al., "QASMBench: A Low-level Quantum Benchmark Suite for NISQ Evaluation and Simulation", ACM TQCI 2023
- **RevLib**: Wille et al., "RevLib: An Online Resource for Reversible Functions and Reversible Circuits", ISMVL 2008
