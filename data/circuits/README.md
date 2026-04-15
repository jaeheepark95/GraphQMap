# GraphQMap Dataset

Quantum circuit dataset for training and evaluating the GraphQMap qubit mapping model.

## Overview

| | Circuits |
|---|:---:|
| **Total** | **7,165** |
| QUEKO | 900 |
| MLQD | 4,443 |
| MQT Bench | 1,448 |
| QASMBench | 111 |
| RevLib | 263 |

## Directory Structure

```
data/circuits/
├── qasm/                           # All .qasm circuit files (OPENQASM 2.0)
│   ├── queko/                      # QUEKO benchmark circuits
│   ├── mlqd/                       # MLQD (Machine Learning Quantum Dataset)
│   ├── mqt_bench/                  # MQT Bench generated circuits
│   ├── qasmbench/                  # QASMBench circuits
│   └── revlib/                     # RevLib reversible circuits
├── splits/                         # Train/val split definitions
│   ├── train_all.json
│   ├── val.json
│   └── val_queko_only.json
├── cache/                          # Preprocessed .pt files (per-circuit features)
└── README.md                       # This file
```

## Dataset Details

### 1. QUEKO (900 circuits)

**Source**: [QUEKO-benchmark](https://github.com/tbcdebug/QUEKO-benchmark)

Artificially generated circuits. Three categories:

| Category | Circuits | Description |
|---|:---:|---|
| BNTF | 180 | Near-term feasible (depth 5-45) |
| BSS | 360 | Scalability study (depth 100-900) |
| BIGD | 360 | Gate density impact study |

### 2. MLQD (4,443 circuits)

**Source**: [MLQD](https://github.com/WanHsuanLin/MLQD) (Machine Learning Quantum Dataset)

Circuits derived from QASMBench, originally processed through OLSQ2 optimal qubit mapper on 5 hardware backends.

### 3. MQT Bench (1,448 circuits)

**Source**: [MQT Bench](https://github.com/munich-quantum-toolkit/bench) (Python API, v2.1.1)

Generated via `scripts/generate_mqt_bench.py` with 29 algorithm types across 2-127 qubits.

**Algorithm types**: ae, bmw_quark_cardinality, bmw_quark_copula, bv, cdkm_ripple_carry_adder, dj, draper_qft_adder, full_adder, ghz, graphstate, grover, half_adder, hhl, modular_adder, multiplier, qaoa, qft, qftentangled, qnn, qpeexact, qpeinexact, qwalk, randomcircuit, rg_qft_multiplier, vbe_ripple_carry_adder, vqe_real_amp, vqe_su2, vqe_two_local, wstate

### 4. QASMBench (111 circuits)

**Source**: [QASMBench](https://github.com/pnnl/QASMBench)

Filtered to 2Q-127Q. Includes small (43), medium (25), and large (43) categories.

### 5. RevLib (263 circuits)

**Source**: [RevLib](http://www.informatik.uni-bremen.de/rev_lib/) via [Real2QASM](https://github.com/changkyu-u/Real2QASM)

Reversible circuits converted from `.real` format to `.qasm`. Filtered to ≤127Q.

## Splits

Split files in `splits/` define which circuits participate in training and validation:

| File | Entries | Description |
|---|:---:|---|
| `train_all.json` | 969 | All training circuits (unsupervised surrogate loss) |
| `val.json` | 28 | Validation set |
| `val_queko_only.json` | 2 | QUEKO validation |

**Split format**:
```json
[
  {"source": "queko", "file": "16QBT_05CYC_TFL_0.qasm"},
  {"source": "mqt_bench", "file": "ghz_n10.qasm"},
  ...
]
```

## Hardware Backends

All training uses **Qiskit FakeBackendV2** backends (49 total). QUEKO/MLQD circuits are randomly re-assigned to real Qiskit backends at data load time. See `configs/base.yaml` for the active training/validation/test backend list.

## Reproduction Scripts

| Script | Purpose |
|---|---|
| `scripts/generate_mqt_bench.py` | Generate MQT Bench circuits via Python API |
| `scripts/normalize_gates.py` | Normalize all QASM to basis gates `{cx, id, rz, sx, x}` |
| `scripts/preprocess_circuits.py` | Build cache/*.pt with node/edge features |

## Citation

If you use this dataset, please cite the original sources:

- **QUEKO**: Tan & Cong, "Optimality Study of Existing Quantum Computing Layout Synthesis Tools", IEEE TC 2020
- **MLQD/MLQM**: "MLQM: Machine Learning Approach for Accelerating Optimal Qubit Mapping", Future Generation Computer Systems 2025
- **MQT Bench**: Quetschlich et al., "MQT Bench: Benchmarking Software and Design Automation Tools for Quantum Computing", Quantum 2023
- **QASMBench**: Li et al., "QASMBench: A Low-level Quantum Benchmark Suite for NISQ Evaluation and Simulation", ACM TQCI 2023
- **RevLib**: Wille et al., "RevLib: An Online Resource for Reversible Functions and Reversible Circuits", ISMVL 2008
