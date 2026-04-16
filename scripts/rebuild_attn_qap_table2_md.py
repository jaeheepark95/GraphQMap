"""Rebuild table2.md from pst_summary.csv fixing the ×100 display bug.

PSTv2 already returns 0-100 scale values, so we must not multiply by 100 again.
"""
import csv
import os
import sys
from collections import defaultdict
import numpy as np

RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    '/home/jaehee/workspace/projects/GraphQMap/runs/attn_qap/20260415_024009_reproduce_table2'

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
METHODS = ['SABRE', 'NA', 'NASSC', 'Ours_T', 'Ours_R', 'Ours_W']
BACKENDS = ['Toronto', 'Rochester', 'Washington']

with open(os.path.join(RUN_DIR, 'pst_summary.csv')) as f:
    rows = list(csv.DictReader(f))

agg = {}  # (eval_set, backend, circuit, method) -> mean (already in percent)
for r in rows:
    agg[(r['eval_set'], r['backend'], r['circuit'], r['method'])] = float(r['mean'])

md = ['# Attention QAP Reproduction — Table 2', '',
      f'Run: `{RUN_DIR}`', '',
      'Shots: 8192, Reps: 3, opt_level: 3',
      '', '_PST values are in %, averaged over 3 reps._', '']

for es_name, circ_list in [('orig12', ORIG_CIRCUITS), ('pozzi13', POZZI_CIRCUITS)]:
    md.append(f'## Eval set: {es_name} ({len(circ_list)} circuits)')
    md.append('')
    for backend in BACKENDS:
        md.append(f'### {backend}')
        md.append('| Circuit | ' + ' | '.join(METHODS) + ' |')
        md.append('|' + '---|' * (len(METHODS) + 1))
        per_method = {m: [] for m in METHODS}
        for cname in circ_list:
            row = [cname]
            for m in METHODS:
                v = agg.get((es_name, backend, cname, m))
                if v is not None and v >= 0:
                    per_method[m].append(v)
                    row.append(f'{v:.2f}')
                else:
                    row.append('-')
            md.append('| ' + ' | '.join(row) + ' |')
        avg_row = ['**Average**']
        for m in METHODS:
            arr = per_method[m]
            avg_row.append(f'**{np.mean(arr):.2f}**' if arr else '-')
        md.append('| ' + ' | '.join(avg_row) + ' |')
        md.append('')

# Cross-hardware generalization summary (paper-style)
md.append('## Cross-Hardware Generalization Summary (paper Table 2 format)')
md.append('')
for es_name, circ_list in [('orig12', ORIG_CIRCUITS), ('pozzi13', POZZI_CIRCUITS)]:
    md.append(f'### Eval set: {es_name} — average PST (%) across circuits')
    md.append('')
    md.append('| Train \\ Eval | Toronto | Rochester | Washington |')
    md.append('|---|---|---|---|')
    for train_key, train_label in [('T', 'Toronto'), ('R', 'Rochester'), ('W', 'Washington')]:
        cells = [f'Ours_{train_key} ({train_label}-trained)']
        for bk in BACKENDS:
            vals = [agg[(es_name, bk, c, f'Ours_{train_key}')]
                    for c in circ_list
                    if agg.get((es_name, bk, c, f'Ours_{train_key}'), -1) >= 0]
            cells.append(f'{np.mean(vals):.2f}' if vals else '-')
        md.append('| ' + ' | '.join(cells) + ' |')
    for baseline in ['SABRE', 'NA', 'NASSC']:
        cells = [baseline]
        for bk in BACKENDS:
            vals = [agg[(es_name, bk, c, baseline)]
                    for c in circ_list
                    if agg.get((es_name, bk, c, baseline), -1) >= 0]
            cells.append(f'{np.mean(vals):.2f}' if vals else '-')
        md.append('| ' + ' | '.join(cells) + ' |')
    md.append('')

path = os.path.join(RUN_DIR, 'table2.md')
with open(path, 'w') as f:
    f.write('\n'.join(md))
print(f'Wrote {path}')
