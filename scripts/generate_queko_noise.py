"""Generate synthetic noise profiles for QUEKO hardware backends.

QUEKO circuits are designed for specific hardware topologies (Aspen-4, Tokyo,
Rochester, Sycamore) but only provide coupling maps without noise data.
This script generates realistic random noise values sampled from distributions
observed in real Qiskit FakeBackendV2 hardware, then saves them as JSON files
that can be loaded during training.

The noise is generated ONCE with a fixed seed for reproducibility.
"""

import json
import os
import numpy as np
from pathlib import Path

# Fixed seed for reproducibility
SEED = 42

# QUEKO hardware definitions from CONNECTION.py
QUEKO_BACKENDS = {
    "queko_aspen4": {
        "num_qubits": 16,
        "coupling_map": [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
            (0, 8), (3, 11), (4, 12), (7, 15),
            (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15),
        ],
    },
    "queko_tokyo": {
        "num_qubits": 20,
        "coupling_map": [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (1, 6), (1, 7), (2, 6), (2, 7), (3, 8), (3, 9), (4, 8), (4, 9),
            (5, 6), (6, 7), (7, 8), (8, 9),
            (5, 10), (5, 11), (6, 10), (6, 11), (7, 12), (7, 13), (8, 12), (8, 13), (9, 14),
            (10, 11), (11, 12), (12, 13), (13, 14),
            (10, 15), (11, 16), (11, 17), (12, 16), (12, 17), (13, 18), (13, 19), (14, 18), (14, 19),
            (15, 16), (16, 17), (17, 18), (18, 19),
        ],
    },
    "queko_rochester": {
        "num_qubits": 53,
        "coupling_map": [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (4, 6), (5, 9), (6, 13),
            (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15),
            (7, 16), (11, 17), (15, 18), (16, 19), (17, 23), (18, 27),
            (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27),
            (21, 28), (25, 29), (28, 32), (29, 36),
            (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), (37, 38),
            (30, 39), (34, 40), (38, 41), (39, 42), (40, 46), (41, 50),
            (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 48), (48, 49), (49, 50),
            (44, 51), (48, 52),
        ],
    },
    "queko_sycamore": {
        "num_qubits": 54,
        "coupling_map": [
            (0, 6), (1, 6), (1, 7), (2, 7), (2, 8), (3, 8), (3, 9), (4, 9), (4, 10), (5, 10), (5, 11),
            (6, 12), (6, 13), (7, 13), (7, 14), (8, 14), (8, 15), (9, 15), (9, 16), (10, 16), (10, 17), (11, 17),
            (12, 18), (13, 18), (13, 19), (14, 19), (14, 20), (15, 20), (15, 21), (16, 21), (16, 22), (17, 22), (17, 23),
            (18, 24), (18, 25), (19, 25), (19, 26), (20, 26), (20, 27), (21, 27), (21, 28), (22, 28), (22, 29), (23, 29),
            (24, 30), (25, 30), (25, 31), (26, 31), (26, 32), (27, 32), (27, 33), (28, 33), (28, 34), (29, 34), (29, 35),
            (30, 36), (30, 37), (31, 37), (31, 38), (32, 38), (32, 39), (33, 39), (33, 40), (34, 40), (34, 41), (35, 41),
            (36, 42), (37, 42), (37, 43), (38, 43), (38, 44), (39, 44), (39, 45), (40, 45), (40, 46), (41, 46), (41, 47),
            (42, 48), (42, 49), (43, 49), (43, 50), (44, 50), (44, 51), (45, 51), (45, 52), (46, 52), (46, 53), (47, 53),
        ],
    },
}

# Noise distributions observed from real FakeBackendV2 hardware
# (collected from 11 backends: Athens, Belem, Manila, Guadalupe,
#  Montreal, Cairo, Rochester, Manhattan, Brisbane, Osaka, Sherbrooke)
NOISE_DISTRIBUTIONS = {
    "t1":              {"mean": 1.993e-04, "std": 1.209e-04, "min": 7.717e-06, "max": 5.149e-04},
    "t2":              {"mean": 1.371e-04, "std": 1.013e-04, "min": 2.637e-06, "max": 4.888e-04},
    "frequency":       {"mean": 4.894e+09, "std": 1.555e+08, "min": 4.455e+09, "max": 5.460e+09},
    "readout_error":   {"mean": 3.671e-02, "std": 5.391e-02, "min": 2.930e-03, "max": 5.000e-01},
    "sq_gate_error":   {"mean": 9.608e-04, "std": 7.015e-03, "min": 7.104e-05, "max": 1.409e-01},
    "cx_error":        {"mean": 1.017e-01, "std": 2.758e-01, "min": 2.653e-03, "max": 1.000e+00},
    "cx_duration":     {"mean": 5.790e-07, "std": 2.014e-07, "min": 1.671e-07, "max": 1.999e-06},
}


def sample_clipped_normal(rng: np.random.Generator, dist: dict, size: int) -> list[float]:
    """Sample from a normal distribution, clipped to [min, max]."""
    values = rng.normal(dist["mean"], dist["std"], size=size)
    values = np.clip(values, dist["min"], dist["max"])
    return values.tolist()


def generate_backend_noise(backend_name: str, config: dict, rng: np.random.Generator) -> dict:
    """Generate synthetic noise profile for a single QUEKO backend."""
    n = config["num_qubits"]
    edges = config["coupling_map"]

    # Per-qubit noise
    qubit_properties = {}
    for q in range(n):
        qubit_properties[str(q)] = {
            "t1": sample_clipped_normal(rng, NOISE_DISTRIBUTIONS["t1"], 1)[0],
            "t2": sample_clipped_normal(rng, NOISE_DISTRIBUTIONS["t2"], 1)[0],
            "frequency": sample_clipped_normal(rng, NOISE_DISTRIBUTIONS["frequency"], 1)[0],
            "readout_error": sample_clipped_normal(rng, NOISE_DISTRIBUTIONS["readout_error"], 1)[0],
            "sq_gate_error": sample_clipped_normal(rng, NOISE_DISTRIBUTIONS["sq_gate_error"], 1)[0],
        }

    # Per-edge noise (bidirectional)
    edge_properties = {}
    for p, q in edges:
        cx_err = sample_clipped_normal(rng, NOISE_DISTRIBUTIONS["cx_error"], 1)[0]
        cx_dur = sample_clipped_normal(rng, NOISE_DISTRIBUTIONS["cx_duration"], 1)[0]
        edge_properties[f"({p}, {q})"] = {"cx_error": cx_err, "cx_duration": cx_dur}
        edge_properties[f"({q}, {p})"] = {"cx_error": cx_err, "cx_duration": cx_dur}

    return {
        "backend_name": backend_name,
        "num_qubits": n,
        "coupling_map": [[p, q] for p, q in edges],
        "qubit_properties": qubit_properties,
        "edge_properties": edge_properties,
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    output_dir = Path(__file__).resolve().parent.parent / "data" / "circuits" / "backends"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, config in QUEKO_BACKENDS.items():
        noise_profile = generate_backend_noise(name, config, rng)
        out_path = output_dir / f"{name}.json"
        with open(out_path, "w") as f:
            json.dump(noise_profile, f, indent=2)
        print(f"Generated {out_path} ({config['num_qubits']}Q, {len(config['coupling_map'])} edges)")

    print(f"\nAll synthetic noise profiles saved to {output_dir}/")


if __name__ == "__main__":
    main()
