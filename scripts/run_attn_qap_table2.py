"""
Reproduce Table 2 of the Attention_Qubit_Mapping paper.

Setup (per user spec):
- Backends: FakeTorontoV2 (27q), FakeRochesterV2 (53q), FakeWashingtonV2 (127q)
- Train 12 ORIG (RevLib) circuits on EACH backend from scratch (3 models: T, R, W)
- Evaluate on TWO circuit sets:
    (a) ORIG_CIRCUITS (12 train=eval, paper Table 2)
    (b) POZZI_CIRCUITS (13 unseen, extended setup from test_pozzi_table2.py)
- Circuits loaded from references/Attention_Qubit_Mapping/pozzi_benchmarks/ (Clifford+T raw)
  For ORIG_CIRCUITS we fall back to MQM/tests2/benchmarks/ (same filenames).
- Methods: SABRE, NA, NASSC, Ours_T, Ours_R, Ours_W
- PST via AerSimulator.from_backend(backend), 8192 shots, optimization_level=3
- 3 reps per (method, circuit, backend)
- Model HP: s=8, d_k=8, T=15, tau_0=1.0, beta=0.85, Adam lr=1e-2, 100 epochs

Outputs under runs/attn_qap/<TIMESTAMP>_reproduce_table2/
    config.json, models/{toronto,rochester,washington}.npz,
    train_logs/*.log, pst_raw.csv, pst_summary.csv, table2.md, stdout.log
"""
import sys
import os
import json
import csv
import time
import datetime as dt
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REF_ROOT = os.path.join(PROJECT_ROOT, 'references', 'Attention_Qubit_Mapping')
sys.path.insert(0, REF_ROOT)
sys.path.insert(0, os.path.join(REF_ROOT, 'MQM', 'tests2'))

from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager, passes
from qiskit_ibm_runtime import fake_provider
from qiskit_aer import AerSimulator

from butils import load_circuit, build_transpiler, PSTv2  # type: ignore
from attn_map.model import AttentionQAPModel  # type: ignore
from attn_map.layout_and_routing_pass import AttentionLayoutAndRouting  # type: ignore
from attn_map.test_full_comparison import build_opt3_post_routing, train  # type: ignore


SHOTS = 8192
REPS = 3

POZZI_DIR = os.path.join(REF_ROOT, 'pozzi_benchmarks')
MQM_BENCHMARK_DIR = os.path.join(REF_ROOT, 'MQM', 'tests2', 'benchmarks')

ORIG_CIRCUITS = [
    'bv_n3', 'bv_n4', 'peres_3', 'toffoli_3',
    'fredkin_3', 'xor5_254', '3_17_13', '4mod5-v1_22',
    'mod5mils_65', 'alu-v0_27', 'decod24-v2_43', '4gt13_92',
]

POZZI_CIRCUITS = [
    'ham3_102', 'miller_11', 'decod24-v0_38', 'rd32-v0_66',
    '4gt5_76', '4mod7-v0_94', 'alu-v2_32', 'hwb4_49',
    'ex1_226', 'decod24-bdd_294', 'ham7_104', 'rd53_138',
    'qft_10',
]

MODEL_HP = dict(d_0=10, d_k=8, T=15, tau_0=1.0, beta=0.85)
TRAIN_HP = dict(epochs=100, lr=1e-2)


def add_measurements(qc: QuantumCircuit) -> QuantumCircuit:
    if qc.count_ops().get('measure', 0) > 0:
        return qc
    used = set()
    for inst in qc.data:
        for q in inst.qubits:
            used.add(qc.find_bit(q).index)
    new_qc = qc.copy()
    if len(new_qc.clbits) == 0:
        new_qc.add_register(ClassicalRegister(len(used), 'meas'))
    for idx, q in enumerate(sorted(used)):
        if idx < len(new_qc.clbits):
            new_qc.measure(q, idx)
    return new_qc


def load_qasm(name: str) -> QuantumCircuit:
    """Prefer references/.../pozzi_benchmarks (Clifford+T raw).
    Fall back to MQM/tests2/benchmarks for the ORIG set if missing."""
    p1 = os.path.join(POZZI_DIR, f'{name}.qasm')
    if os.path.exists(p1):
        qc = QuantumCircuit.from_qasm_file(p1)
    else:
        p2 = os.path.join(MQM_BENCHMARK_DIR, f'{name}.qasm')
        if not os.path.exists(p2):
            raise FileNotFoundError(f'{name}.qasm not in {POZZI_DIR} or {MQM_BENCHMARK_DIR}')
        qc = QuantumCircuit.from_qasm_file(p2)
    return add_measurements(qc)


def transpile_baseline(qc, backend, layout_method: str):
    pm = build_transpiler(
        backend,
        layout_method=layout_method,
        routing_method='sabre',
        optimization_level=3,
    )
    return pm.run(qc)


def transpile_ours(qc, backend, model):
    pm1 = PassManager()
    pm1.append(passes.Unroll3qOrMore())
    pm1.append([passes.RemoveResetInZeroState(),
                passes.OptimizeSwapBeforeMeasure(),
                passes.RemoveDiagonalGatesBeforeMeasure()])
    pm1.append(AttentionLayoutAndRouting(backend, model=model, z_guided=True))
    tc = pm1.run(qc)
    pm2 = build_opt3_post_routing(backend)
    return pm2.run(tc)


def measure_pst(tc, noisy_sim, ideal_sim) -> float:
    ic = ideal_sim.run(tc, shots=SHOTS).result().get_counts()
    nc = noisy_sim.run(tc, shots=SHOTS).result().get_counts()
    pst = PSTv2(nc, ic)
    return float(sum(pst) / len(pst)) if isinstance(pst, list) else float(pst)


def eval_one(method: str, qc, backend, noisy_sim, ideal_sim, models) -> float:
    try:
        if method == 'SABRE':
            tc = transpile_baseline(qc, backend, 'sabre')
        elif method == 'NA':
            tc = transpile_baseline(qc, backend, 'noise_adaptive')
        elif method == 'NASSC':
            tc = transpile_baseline(qc, backend, 'dense')  # NASSC via dense layout + sabre route + opt3
        elif method.startswith('Ours_'):
            key = method.split('_')[1]
            tc = transpile_ours(qc, backend, models[key])
        else:
            return -1.0
        return measure_pst(tc, noisy_sim, ideal_sim)
    except Exception as e:
        print(f'    [WARN] {method} failed: {e}')
        return -1.0


def train_backend_model(backend, circuits_dag, log_path: str):
    model = AttentionQAPModel(**MODEL_HP)
    buf = StringIO()
    t0 = time.time()
    with redirect_stdout(buf):
        model = train(model, [backend], circuits_dag,
                      epochs=TRAIN_HP['epochs'], lr=TRAIN_HP['lr'])
    elapsed = time.time() - t0
    log = buf.getvalue()
    with open(log_path, 'w') as f:
        f.write(log)
        f.write(f'\n[elapsed] {elapsed:.1f}s\n')
    print(log)
    print(f'  [train] {backend.name}: {elapsed:.1f}s')
    return model


def main():
    ts = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(PROJECT_ROOT, 'runs', 'attn_qap',
                           f'{ts}_reproduce_table2')
    os.makedirs(os.path.join(run_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'train_logs'), exist_ok=True)

    # Tee stdout to stdout.log
    stdout_log = open(os.path.join(run_dir, 'stdout.log'), 'w')

    class Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, s):
            for st in self.streams: st.write(s); st.flush()
        def flush(self):
            for st in self.streams: st.flush()

    sys.stdout = Tee(sys.__stdout__, stdout_log)

    print(f'Run dir: {run_dir}')

    # --- Save config ---
    config = {
        'model_hp': MODEL_HP,
        'train_hp': TRAIN_HP,
        'shots': SHOTS,
        'reps': REPS,
        'optimization_level': 3,
        'backends': ['FakeTorontoV2', 'FakeRochesterV2', 'FakeWashingtonV2'],
        'train_circuits': ORIG_CIRCUITS,
        'eval_sets': {
            'orig12': ORIG_CIRCUITS,
            'pozzi13': POZZI_CIRCUITS,
        },
        'methods': ['SABRE', 'NA', 'NASSC', 'Ours_T', 'Ours_R', 'Ours_W'],
        'qasm_source': 'references/Attention_Qubit_Mapping/pozzi_benchmarks (fallback: MQM/tests2/benchmarks)',
    }
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # --- Backends ---
    toronto = fake_provider.FakeTorontoV2()
    rochester = fake_provider.FakeRochesterV2()
    washington = fake_provider.FakeWashingtonV2()
    backends = [(toronto, 'Toronto', 'T'),
                (rochester, 'Rochester', 'R'),
                (washington, 'Washington', 'W')]

    # --- Load 12 training circuits (as DAGs for training) ---
    print('\n=== Loading training circuits (12 ORIG) ===')
    circuits_dag = []
    for name in ORIG_CIRCUITS:
        qc = load_qasm(name).decompose()
        circuits_dag.append((name, circuit_to_dag(qc)))
        print(f'  {name}: {qc.num_qubits}q, {qc.size()} gates')

    # --- Train 3 models from scratch ---
    models = {}
    for backend, label, key in backends:
        print(f'\n=== Training {label} model (from scratch) ===')
        log_path = os.path.join(run_dir, 'train_logs', f'{label.lower()}.log')
        m = train_backend_model(backend, circuits_dag, log_path)
        m.save(os.path.join(run_dir, 'models', f'{label.lower()}.npz'))
        models[key] = m

    # --- Load evaluation circuits ---
    eval_sets = {
        'orig12': [(n, load_qasm(n)) for n in ORIG_CIRCUITS],
        'pozzi13': [(n, load_qasm(n)) for n in POZZI_CIRCUITS],
    }

    methods = ['SABRE', 'NA', 'NASSC', 'Ours_T', 'Ours_R', 'Ours_W']

    # --- Evaluate ---
    raw_rows = []  # eval_set, backend, circuit, method, rep, pst
    for backend, blabel, _ in backends:
        print(f'\n=== Evaluating on {blabel} ===')
        noisy_sim = AerSimulator.from_backend(backend)
        ideal_sim = AerSimulator.from_backend(backend, noise_model=None)
        for set_name, circs in eval_sets.items():
            print(f'\n-- set={set_name} --')
            for name, qc in circs:
                for rep in range(REPS):
                    for m in methods:
                        pst = eval_one(m, qc, backend, noisy_sim, ideal_sim, models)
                        raw_rows.append({
                            'eval_set': set_name,
                            'backend': blabel,
                            'circuit': name,
                            'method': m,
                            'rep': rep,
                            'pst': pst,
                        })
                    print(f'  {set_name}/{name} rep{rep}: ' +
                          ' '.join(f'{m}={raw_rows[-(len(methods)-i)]["pst"]:.3f}'
                                   for i, m in enumerate(methods)))

    # --- Save raw CSV ---
    raw_csv = os.path.join(run_dir, 'pst_raw.csv')
    with open(raw_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['eval_set', 'backend', 'circuit',
                                          'method', 'rep', 'pst'])
        w.writeheader()
        w.writerows(raw_rows)
    print(f'\nSaved raw: {raw_csv}')

    # --- Summary: mean/std over reps ---
    from collections import defaultdict
    agg = defaultdict(list)  # (eval_set, backend, circuit, method) -> [pst,...]
    for r in raw_rows:
        agg[(r['eval_set'], r['backend'], r['circuit'], r['method'])].append(r['pst'])

    summary_rows = []
    for (es, bk, ck, mt), vals in agg.items():
        vals_ok = [v for v in vals if v >= 0]
        summary_rows.append({
            'eval_set': es, 'backend': bk, 'circuit': ck, 'method': mt,
            'mean': float(np.mean(vals_ok)) if vals_ok else -1.0,
            'std': float(np.std(vals_ok)) if vals_ok else -1.0,
            'n': len(vals_ok),
        })
    sum_csv = os.path.join(run_dir, 'pst_summary.csv')
    with open(sum_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['eval_set', 'backend', 'circuit',
                                          'method', 'mean', 'std', 'n'])
        w.writeheader()
        w.writerows(summary_rows)
    print(f'Saved summary: {sum_csv}')

    # --- Table 2 style markdown (per eval_set) ---
    md_lines = ['# Attention QAP Reproduction — Table 2', '',
                f'Run: `{run_dir}`', '',
                f'Shots: {SHOTS}, Reps: {REPS}, opt_level: 3', '']
    for es_name, circ_list in [('orig12', ORIG_CIRCUITS),
                               ('pozzi13', POZZI_CIRCUITS)]:
        md_lines.append(f'## Eval set: {es_name} ({len(circ_list)} circuits)')
        md_lines.append('')
        for _, blabel, _ in backends:
            md_lines.append(f'### {blabel}')
            header = '| Circuit | ' + ' | '.join(methods) + ' |'
            sep = '|' + '---|' * (len(methods) + 1)
            md_lines += [header, sep]
            per_method_means = {m: [] for m in methods}
            for cname in circ_list:
                row = [cname]
                for m in methods:
                    key = (es_name, blabel, cname, m)
                    vals = agg.get(key, [])
                    vals_ok = [v for v in vals if v >= 0]
                    if vals_ok:
                        mean = float(np.mean(vals_ok))
                        per_method_means[m].append(mean)
                        row.append(f'{mean*100:.1f}%')
                    else:
                        row.append('-')
                md_lines.append('| ' + ' | '.join(row) + ' |')
            avg_row = ['**Average**']
            for m in methods:
                arr = per_method_means[m]
                avg_row.append(f'**{np.mean(arr)*100:.1f}%**' if arr else '-')
            md_lines.append('| ' + ' | '.join(avg_row) + ' |')
            md_lines.append('')
    md_path = os.path.join(run_dir, 'table2.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f'Saved table2: {md_path}')

    stdout_log.close()


if __name__ == '__main__':
    main()
