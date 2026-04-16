"""
Train Attention_Qubit_Mapping model on GraphQMap's training dataset.

Uses mini-batch sampling to handle 969 circuits × 52 backends efficiently
with the reference model's numerical gradient optimizer.

Preprocessing is done once for all circuits/backends, then mini-batches
are sampled per epoch for SGD-style training.
"""
import sys
import os
import json
import time
import random
import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit_ibm_runtime import fake_provider

# Add reference code to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'references', 'Attention_Qubit_Mapping'))
from attn_map.model import AttentionQAPModel
from attn_map.attention import extract_mapping
from attn_map.embedding import parse_circuit_to_features

# GraphQMap evaluation infrastructure
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evaluation.pst import compute_pst
from evaluation.transpiler import transpile_with_timing

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'circuits'
QASM_DIR = DATA_DIR / 'qasm'
SPLITS_DIR = DATA_DIR / 'splits'
BENCHMARK_DIR = QASM_DIR / 'benchmarks'
RUNS_DIR = PROJECT_ROOT / 'runs_AQM'

# ── Stage 2 training backends (Qiskit FakeBackendV2 only) ────────
TRAINING_BACKENDS = [
    'FakeTorontoV2',
]

TEST_BACKENDS = ['FakeTorontoV2', 'FakeRochesterV2', 'FakeWashingtonV2']

SHOTS = 8192


# ── Helpers ──────────────────────────────────────────────────────
def get_backend_instance(name: str):
    """Instantiate a FakeBackendV2 by class name."""
    cls = getattr(fake_provider, name)
    return cls()


def load_qasm_circuit(source: str, filename: str) -> QuantumCircuit | None:
    """Load a QASM circuit file and return QuantumCircuit."""
    path = QASM_DIR / source / filename
    if not path.exists():
        return None
    try:
        qc = QuantumCircuit.from_qasm_file(str(path))
        if qc.count_ops().get('measure', 0) == 0:
            qc.measure_all()
        qc = qc.decompose()
        return qc
    except Exception as e:
        print(f"  [WARN] Failed to load {source}/{filename}: {e}")
        return None


# ── Data loading & preprocessing ─────────────────────────────────
def load_training_circuits(split_path: str, max_qubits: int = None) -> list[dict]:
    """Load circuits from a GraphQMap split file.

    Returns list of dicts with keys: name, source, qc, dag, circ_data, n_qubits.
    """
    with open(split_path) as f:
        split = json.load(f)

    circuits = []
    skipped = 0
    for entry in split:
        source = entry['source']
        filename = entry['file']
        name = filename.replace('.qasm', '')

        qc = load_qasm_circuit(source, filename)
        if qc is None:
            skipped += 1
            continue

        n_qubits = qc.num_qubits
        if max_qubits and n_qubits > max_qubits:
            skipped += 1
            continue

        circuits.append({
            'name': name,
            'source': source,
            'qc': qc,
            'n_qubits': n_qubits,
        })

    print(f"  Loaded {len(circuits)} circuits ({skipped} skipped)")
    return circuits


def preprocess_circuits(model: AttentionQAPModel, circuits: list[dict]) -> list[dict]:
    """Preprocess all circuits: DAG → features. Adds 'circ_data' key."""
    t0 = time.time()
    processed = []
    for i, c in enumerate(circuits):
        try:
            dag = circuit_to_dag(c['qc'])
            circ_data = model.preprocess_circuit(dag)
            c['circ_data'] = circ_data
            processed.append(c)
        except Exception as e:
            print(f"  [WARN] Preprocess failed for {c['name']}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  Preprocessed {i+1}/{len(circuits)} circuits...")

    elapsed = time.time() - t0
    print(f"  Circuit preprocessing done: {len(processed)}/{len(circuits)} in {elapsed:.1f}s")
    return processed


def preprocess_backends(model: AttentionQAPModel, backend_names: list[str]) -> list[dict]:
    """Preprocess all backends: noise extraction + C_eff + features."""
    t0 = time.time()
    backends = []
    for name in backend_names:
        try:
            backend = get_backend_instance(name)
            hw_data = model.preprocess_hardware(backend)
            backends.append({
                'name': name,
                'backend': backend,
                'hw_data': hw_data,
                'n_qubits': backend.num_qubits,
            })
        except Exception as e:
            print(f"  [WARN] Backend {name} failed: {e}")

    elapsed = time.time() - t0
    print(f"  Backend preprocessing done: {len(backends)}/{len(backend_names)} in {elapsed:.1f}s")
    return backends


def build_compatibility_index(circuits: list[dict], backends: list[dict]) -> dict:
    """Build index: backend_name → list of compatible circuit indices."""
    compat = {}
    for b in backends:
        indices = [i for i, c in enumerate(circuits) if c['n_qubits'] <= b['n_qubits']]
        compat[b['name']] = indices
    return compat


# ── Training ─────────────────────────────────────────────────────
def compute_full_gradient(model, circ_data, hw_data, eps=1e-4):
    """Compute numerical gradient for all parameters."""
    grad_W_Q = np.zeros_like(model.W_Q)
    grad_W_K = np.zeros_like(model.W_K)

    for i in range(model.d_0):
        for j in range(model.d_k):
            model.W_Q[i, j] += eps
            Pp, _ = model.forward(circ_data, hw_data)
            lp = model.compute_loss(Pp, circ_data, hw_data)
            model.W_Q[i, j] -= 2 * eps
            Pm, _ = model.forward(circ_data, hw_data)
            lm = model.compute_loss(Pm, circ_data, hw_data)
            model.W_Q[i, j] += eps
            grad_W_Q[i, j] = (lp - lm) / (2 * eps)

    for i in range(model.d_0):
        for j in range(model.d_k):
            model.W_K[i, j] += eps
            Pp, _ = model.forward(circ_data, hw_data)
            lp = model.compute_loss(Pp, circ_data, hw_data)
            model.W_K[i, j] -= 2 * eps
            Pm, _ = model.forward(circ_data, hw_data)
            lm = model.compute_loss(Pm, circ_data, hw_data)
            model.W_K[i, j] += eps
            grad_W_K[i, j] = (lp - lm) / (2 * eps)

    model.lam += eps
    Pp, _ = model.forward(circ_data, hw_data)
    lp = model.compute_loss(Pp, circ_data, hw_data)
    model.lam -= 2 * eps
    Pm, _ = model.forward(circ_data, hw_data)
    lm = model.compute_loss(Pm, circ_data, hw_data)
    model.lam += eps
    grad_lam = (lp - lm) / (2 * eps)

    return grad_W_Q, grad_W_K, grad_lam


def train_model(model, circuits, backends, compat_index, *,
                epochs=100, lr=1e-2, circuits_per_epoch=16,
                backends_per_epoch=4, seed=42):
    """Train model with mini-batch Adam optimizer.

    Each epoch samples `backends_per_epoch` backends and `circuits_per_epoch`
    compatible circuits per backend for SGD-style training.
    """
    rng = random.Random(seed)

    # Adam state
    m_Q, v_Q = np.zeros_like(model.W_Q), np.zeros_like(model.W_Q)
    m_K, v_K = np.zeros_like(model.W_K), np.zeros_like(model.W_K)
    m_l, v_l = 0.0, 0.0
    b1, b2, ae = 0.9, 0.999, 1e-8

    best_loss = float('inf')
    best_model_state = None
    loss_history = []

    n_backends = min(backends_per_epoch, len(backends))
    total_params = model.W_Q.size + model.W_K.size + 1

    print(f"\n{'='*70}")
    print(f"Training: {len(circuits)} circuits, {len(backends)} backends")
    print(f"Mini-batch: {circuits_per_epoch} circuits × {n_backends} backends/epoch")
    print(f"Total params: {total_params}, Epochs: {epochs}, LR: {lr}")
    print(f"{'='*70}")

    start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss, count = 0.0, 0
        acc_gQ = np.zeros_like(model.W_Q)
        acc_gK = np.zeros_like(model.W_K)
        acc_gl = 0.0

        # Sample backends for this epoch
        sampled_backends = rng.sample(backends, n_backends)

        for binfo in sampled_backends:
            hw_data = binfo['hw_data']
            compatible_indices = compat_index[binfo['name']]

            if len(compatible_indices) == 0:
                continue

            # Sample circuits compatible with this backend
            n_sample = min(circuits_per_epoch, len(compatible_indices))
            sampled_indices = rng.sample(compatible_indices, n_sample)

            for idx in sampled_indices:
                circ_data = circuits[idx]['circ_data']
                P, _ = model.forward(circ_data, hw_data)
                loss = model.compute_loss(P, circ_data, hw_data)
                total_loss += loss

                gQ, gK, gl = compute_full_gradient(model, circ_data, hw_data)
                acc_gQ += gQ
                acc_gK += gK
                acc_gl += gl
                count += 1

        if count == 0:
            continue

        acc_gQ /= count
        acc_gK /= count
        acc_gl /= count
        avg_loss = total_loss / count
        t = epoch + 1

        # Adam update
        m_Q = b1 * m_Q + (1 - b1) * acc_gQ
        v_Q = b2 * v_Q + (1 - b2) * acc_gQ ** 2
        model.W_Q -= lr * (m_Q / (1 - b1 ** t)) / (np.sqrt(v_Q / (1 - b2 ** t)) + ae)

        m_K = b1 * m_K + (1 - b1) * acc_gK
        v_K = b2 * v_K + (1 - b2) * acc_gK ** 2
        model.W_K -= lr * (m_K / (1 - b1 ** t)) / (np.sqrt(v_K / (1 - b2 ** t)) + ae)

        m_l = b1 * m_l + (1 - b1) * acc_gl
        v_l = b2 * v_l + (1 - b2) * acc_gl ** 2
        model.lam -= lr * (m_l / (1 - b1 ** t)) / (np.sqrt(v_l / (1 - b2 ** t)) + ae)
        model.lam = max(model.lam, 0.01)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = (model.W_Q.copy(), model.W_K.copy(), model.lam)

        epoch_time = time.time() - epoch_start
        loss_history.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - start
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Best: {best_loss:.4f} | λ: {model.lam:.4f} | "
                  f"Samples: {count} | Time: {epoch_time:.1f}s | "
                  f"Total: {elapsed:.0f}s")

    total_time = time.time() - start
    print(f"\n  Training complete. Best loss: {best_loss:.4f} | "
          f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Restore best model
    if best_model_state:
        model.W_Q, model.W_K, model.lam = best_model_state
        print(f"  Restored best model (loss={best_loss:.4f})")

    return model, loss_history


# ── Evaluation ───────────────────────────────────────────────────
def create_ideal_simulator(backend):
    from qiskit_aer import AerSimulator
    return AerSimulator.from_backend(backend, noise_model=None,
                                     method='tensor_network', device='GPU')


def create_noisy_simulator(backend):
    from qiskit_aer import AerSimulator
    return AerSimulator.from_backend(backend, method='tensor_network', device='GPU')


def get_attn_layout(model, qc, backend):
    """Get initial layout from attention model."""
    dag = circuit_to_dag(qc)
    circ_data = model.preprocess_circuit(dag)
    hw_data = model.preprocess_hardware(backend)
    P, mapping = model.forward(circ_data, hw_data)
    return list(mapping)


def load_benchmark_circuits() -> list[tuple[str, QuantumCircuit]]:
    """Load benchmark circuits for evaluation."""
    circuits = []
    for f in sorted(BENCHMARK_DIR.glob('*.qasm')):
        if f.name == 'rd84_253.qasm':
            continue  # skip — too large for simulation
        qc = QuantumCircuit.from_qasm_file(str(f))
        if qc.count_ops().get('measure', 0) == 0:
            qc.measure_all()
        qc = qc.decompose()
        circuits.append((f.stem, qc))
    return circuits


def evaluate_backend(model, backend, circuits, baselines, label):
    """Evaluate model + baselines on a single backend."""
    print(f"\n{'='*70}")
    print(f"{label}: {backend.name} ({backend.num_qubits}Q)")
    print(f"{'='*70}")

    ideal_sim = create_ideal_simulator(backend)
    noisy_sim = create_noisy_simulator(backend)

    method_labels = [m[0] for m in baselines] + ['Ours+SABRE', 'Ours+NASSC']

    hdr = f"{'Circuit':<18}"
    for lb in method_labels:
        hdr += f" | {lb:>12}"
    print(hdr)
    print('-' * len(hdr))

    all_psts = {lb: [] for lb in method_labels}

    def run_sim(tc, ideal_sim, noisy_sim, backend):
        try:
            ideal_result = ideal_sim.run(tc, shots=SHOTS).result()
            if not ideal_result.success:
                raise RuntimeError(str(ideal_result.status))
            noisy_result = noisy_sim.run(tc, shots=SHOTS).result()
            if not noisy_result.success:
                raise RuntimeError(str(noisy_result.status))
            return ideal_result.get_counts(), noisy_result.get_counts(), ideal_sim, noisy_sim
        except Exception:
            ideal_sim = create_ideal_simulator(backend)
            noisy_sim = create_noisy_simulator(backend)
            ideal_result = ideal_sim.run(tc, shots=SHOTS).result()
            noisy_result = noisy_sim.run(tc, shots=SHOTS).result()
            if not ideal_result.success or not noisy_result.success:
                raise
            return ideal_result.get_counts(), noisy_result.get_counts(), ideal_sim, noisy_sim

    for name, qc in circuits:
        row = f"{name:<18}"

        for method_label, layout_m, routing_m in baselines:
            try:
                tc, meta = transpile_with_timing(
                    qc, backend,
                    layout_method=layout_m,
                    routing_method=routing_m,
                )
                ideal_counts, noisy_counts, ideal_sim, noisy_sim = \
                    run_sim(tc, ideal_sim, noisy_sim, backend)
                pst = compute_pst(noisy_counts, ideal_counts)
                if isinstance(pst, list):
                    pst = sum(pst) / len(pst)
                cx = tc.count_ops().get('cx', 0) + tc.count_ops().get('ecr', 0) + tc.count_ops().get('cz', 0)
                row += f" | {pst*100:5.1f}% {cx:3d}cx"
                all_psts[method_label].append(pst)
            except Exception as e:
                row += f" | {'ERROR':>12}"
                print(f"  [WARN] {name}/{method_label}: {e}")

        for routing_label, routing_m in [('Ours+SABRE', 'sabre'), ('Ours+NASSC', 'nassc')]:
            try:
                layout = get_attn_layout(model, qc, backend)
                tc, meta = transpile_with_timing(
                    qc, backend,
                    initial_layout=layout,
                    layout_method='given',
                    routing_method=routing_m,
                )
                ideal_counts, noisy_counts, ideal_sim, noisy_sim = \
                    run_sim(tc, ideal_sim, noisy_sim, backend)
                pst = compute_pst(noisy_counts, ideal_counts)
                if isinstance(pst, list):
                    pst = sum(pst) / len(pst)
                cx = tc.count_ops().get('cx', 0) + tc.count_ops().get('ecr', 0) + tc.count_ops().get('cz', 0)
                row += f" | {pst*100:5.1f}% {cx:3d}cx"
                all_psts[routing_label].append(pst)
            except Exception as e:
                row += f" | {'ERROR':>12}"
                print(f"  [WARN] {name}/{routing_label}: {e}")

        print(row)

    print('-' * len(hdr))
    row = f"{'AVERAGE':<18}"
    for lb in method_labels:
        vals = all_psts[lb]
        if vals:
            row += f" | {np.mean(vals)*100:5.1f}%      "
        else:
            row += f" | {'N/A':>12}"
    print(row)

    return all_psts


# ── Results saving ───────────────────────────────────────────────
def setup_run_dir(name: str = None) -> Path:
    RUNS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    dirname = f"{ts}_{name}" if name else ts
    run_dir = RUNS_DIR / dirname
    run_dir.mkdir(exist_ok=True)
    return run_dir


def save_results_csv(run_dir: Path, all_results: dict, eval_circuit_names: list[str]):
    path = run_dir / 'eval_results.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['backend', 'circuit', 'method', 'pst'])
        for bname, backend_results in all_results.items():
            for method, pst_list in backend_results.items():
                for i, pst in enumerate(pst_list):
                    writer.writerow([bname, eval_circuit_names[i], method, f'{pst:.6f}'])
    print(f"  Results CSV saved to {path}")


def save_loss_csv(run_dir: Path, loss_history: list[float]):
    path = run_dir / 'loss_history.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss'])
        for i, loss in enumerate(loss_history):
            writer.writerow([i + 1, f'{loss:.6f}'])


def save_summary(run_dir: Path, all_results: dict, method_labels: list[str],
                 train_info: dict):
    path = run_dir / 'summary.md'
    lines = []
    lines.append("# AQM Model — GraphQMap Dataset Training Results\n")
    lines.append("## Training")
    lines.append(f"- Dataset: GraphQMap train_all split ({train_info['n_circuits']} circuits)")
    lines.append(f"- Training backends: {train_info['n_backends']}")
    lines.append(f"- Mini-batch: {train_info['circuits_per_epoch']} circuits × "
                 f"{train_info['backends_per_epoch']} backends/epoch")
    lines.append(f"- Epochs: {train_info['epochs']}")
    lines.append(f"- Best loss: {train_info['best_loss']:.4f}")
    lines.append(f"- Final λ: {train_info['final_lam']:.4f}")
    lines.append(f"- Training time: {train_info['train_time']:.0f}s ({train_info['train_time']/60:.1f} min)")
    lines.append(f"- Seed: {train_info['seed']}")
    lines.append(f"- Hyperparams: d_0={train_info['d_0']}, d_k={train_info['d_k']}, "
                 f"T={train_info['T']}, tau_0={train_info['tau_0']}, beta={train_info['beta']}\n")

    lines.append("## Average PST (%)\n")
    lines.append(f"| {'Backend':<20} |" + " | ".join(f" {lb:>12}" for lb in method_labels) + " |")
    lines.append(f"|{'-'*22}|" + "|".join(f"{'-'*14}" for _ in method_labels) + "|")
    for bname in ['fake_toronto', 'fake_rochester', 'fake_washington']:
        row = f"| {bname:<20} |"
        for lb in method_labels:
            vals = all_results.get(bname, {}).get(lb, [])
            if vals:
                row += f" {np.mean(vals)*100:11.1f}% |"
            else:
                row += f" {'N/A':>12} |"
        lines.append(row)

    # 3-backend average
    row = f"| {'**AVERAGE**':<20} |"
    for lb in method_labels:
        vals = []
        for bname in ['fake_toronto', 'fake_rochester', 'fake_washington']:
            v = all_results.get(bname, {}).get(lb, [])
            if v:
                vals.append(np.mean(v))
        if vals:
            row += f" {np.mean(vals)*100:11.1f}% |"
        else:
            row += f" {'N/A':>12} |"
    lines.append(row)

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Summary saved to {path}")


# ── Main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Train AQM model on GraphQMap dataset')
    parser.add_argument('--name', type=str, default='graphqmap_dataset',
                        help='Run directory name suffix')
    parser.add_argument('--split', type=str, default='train_curated.json',
                        help='Split file name (under data/circuits/splits/)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--circuits-per-epoch', type=int, default=16,
                        help='Number of circuits sampled per backend per epoch')
    parser.add_argument('--backends-per-epoch', type=int, default=4,
                        help='Number of backends sampled per epoch')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-only', type=str, default=None,
                        help='Path to saved model .npz for eval-only mode')
    parser.add_argument('--skip-eval', action='store_true',
                        help='Skip evaluation (training only)')
    parser.add_argument('--max-qubits', type=int, default=None,
                        help='Max circuit qubits to include (default: no limit)')
    # Model hyperparameters
    parser.add_argument('--d0', type=int, default=10, help='Raw feature dim (s+2)')
    parser.add_argument('--dk', type=int, default=8, help='Projection dim')
    parser.add_argument('--s', type=int, default=8, help='HKS time scales')
    parser.add_argument('--T', type=int, default=15, help='Iterative refinement steps')
    parser.add_argument('--tau0', type=float, default=1.0, help='Initial temperature')
    parser.add_argument('--beta', type=float, default=0.85, help='Temperature decay')
    args = parser.parse_args()

    run_dir = setup_run_dir(args.name)
    print(f"Run directory: {run_dir}")

    # ── Model ────────────────────────────────────────────────────
    model = AttentionQAPModel(
        d_0=args.d0, d_k=args.dk, s=args.s,
        T=args.T, tau_0=args.tau0, beta=args.beta,
        sinkhorn_iters=20, seed=args.seed,
    )
    print(f"Model: d_0={args.d0}, d_k={args.dk}, T={args.T}, "
          f"params={model.W_Q.size + model.W_K.size + 1}")

    if args.eval_only:
        model.load(args.eval_only)
        print(f"Loaded model from {args.eval_only}")
        loss_history = []
        train_time = 0
    else:
        # ── Load & preprocess training data ──────────────────────
        split_path = SPLITS_DIR / args.split
        print(f"\nLoading circuits from {split_path}...")
        circuits = load_training_circuits(str(split_path), max_qubits=args.max_qubits)

        print(f"\nPreprocessing circuits...")
        circuits = preprocess_circuits(model, circuits)

        print(f"\nPreprocessing training backends ({len(TRAINING_BACKENDS)})...")
        backends = preprocess_backends(model, TRAINING_BACKENDS)

        # Build compatibility index
        compat_index = build_compatibility_index(circuits, backends)
        for b in backends:
            n_compat = len(compat_index[b['name']])
            if n_compat == 0:
                print(f"  [WARN] {b['name']} ({b['n_qubits']}Q): 0 compatible circuits")

        # Qubit distribution summary
        qubit_counts = [c['n_qubits'] for c in circuits]
        print(f"\n  Circuit qubit distribution: "
              f"min={min(qubit_counts)}, max={max(qubit_counts)}, "
              f"median={sorted(qubit_counts)[len(qubit_counts)//2]}")

        # ── Train ────────────────────────────────────────────────
        t0 = time.time()
        model, loss_history = train_model(
            model, circuits, backends, compat_index,
            epochs=args.epochs, lr=args.lr,
            circuits_per_epoch=args.circuits_per_epoch,
            backends_per_epoch=args.backends_per_epoch,
            seed=args.seed,
        )
        train_time = time.time() - t0

        # Save model
        model_path = run_dir / 'model.npz'
        model.save(str(model_path))
        print(f"Model saved to {model_path}")

        # Save loss history
        save_loss_csv(run_dir, loss_history)

    if args.skip_eval:
        print("\nSkipping evaluation (--skip-eval)")
        return

    # ── Evaluate ─────────────────────────────────────────────────
    print("\nLoading benchmark circuits for evaluation...")
    eval_circuits = load_benchmark_circuits()
    eval_circuit_names = [name for name, _ in eval_circuits]
    print(f"  {len(eval_circuits)} benchmark circuits")

    baselines = [
        ('SABRE', 'sabre', 'sabre'),
        ('QAP+NASSC', 'qap', 'nassc'),
    ]

    test_backends_info = [
        ('Toronto (27Q)', 'FakeTorontoV2'),
        ('Rochester (53Q)', 'FakeRochesterV2'),
        ('Washington (127Q)', 'FakeWashingtonV2'),
    ]

    results = {}
    for label, bname in test_backends_info:
        try:
            backend = get_backend_instance(bname)
            results[backend.name] = evaluate_backend(
                model, backend,
                [(name, qc) for name, qc in eval_circuits],
                baselines, label)
        except Exception as e:
            print(f"\n  [ERROR] {label} evaluation failed: {e}")
            print(f"  Continuing with remaining backends...")
            results[bname.replace('Fake', 'fake_').replace('V2', '').lower()] = {}

    # ── Summary ──────────────────────────────────────────────────
    method_labels = ['SABRE', 'QAP+NASSC', 'Ours+SABRE', 'Ours+NASSC']

    print(f"\n{'='*70}")
    print("SUMMARY (Average PST %)")
    print(f"{'='*70}")
    hdr = f"{'Backend':<20}"
    for lb in method_labels:
        hdr += f" | {lb:>12}"
    print(hdr)
    print('-' * len(hdr))
    for bname in ['fake_toronto', 'fake_rochester', 'fake_washington']:
        row = f"{bname:<20}"
        for lb in method_labels:
            vals = results.get(bname, {}).get(lb, [])
            if vals:
                row += f" | {np.mean(vals)*100:10.1f}%"
            else:
                row += f" | {'N/A':>12}"
        print(row)

    # 3-backend avg
    print('-' * len(hdr))
    row = f"{'AVERAGE':<20}"
    for lb in method_labels:
        backend_avgs = []
        for bname in ['fake_toronto', 'fake_rochester', 'fake_washington']:
            vals = results.get(bname, {}).get(lb, [])
            if vals:
                backend_avgs.append(np.mean(vals))
        if backend_avgs:
            row += f" | {np.mean(backend_avgs)*100:10.1f}%"
        else:
            row += f" | {'N/A':>12}"
    print(row)

    # ── Save ─────────────────────────────────────────────────────
    best_loss = min(loss_history) if loss_history else 0
    train_info = {
        'n_circuits': len(circuits) if not args.eval_only else 0,
        'n_backends': len(TRAINING_BACKENDS),
        'circuits_per_epoch': args.circuits_per_epoch,
        'backends_per_epoch': args.backends_per_epoch,
        'epochs': args.epochs,
        'best_loss': best_loss,
        'final_lam': model.lam,
        'train_time': train_time,
        'seed': args.seed,
        'd_0': args.d0, 'd_k': args.dk,
        'T': args.T, 'tau_0': args.tau0, 'beta': args.beta,
    }
    save_results_csv(run_dir, results, eval_circuit_names)
    save_summary(run_dir, results, method_labels, train_info)

    print(f"\nAll results saved to {run_dir}")


if __name__ == '__main__':
    main()
