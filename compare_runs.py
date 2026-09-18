"""
Multi-task vs single-task comparison on the held-out test set.

Both model families are evaluated on exactly the same 120 official validation
exams, so the comparison is paired: every bootstrap resample draws the same
exams for both models, and the difference in AUC is computed inside the
resample. Writes runs/comparison.json.

Usage:  python compare_runs.py [--n_bootstrap 2000] [--seed 42]
"""
import argparse
import csv
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

TASKS = ['abnormal', 'acl', 'meniscus']
HERE = os.path.dirname(os.path.abspath(__file__))


def load_predictions(path: str, tasks):
    rows = list(csv.DictReader(open(path)))
    ids = np.array([int(r['id']) for r in rows])
    out = {t: (np.array([int(float(r[f'{t}_label'])) for r in rows]),
               np.array([float(r[f'{t}_prob']) for r in rows])) for t in tasks}
    return ids, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--runs_dir', default=os.path.join(HERE, 'runs'))
    p.add_argument('--multitask', default='multitask', help='multi-task run directory name')
    p.add_argument('--single_suffix', default='', help="suffix of the single-task run dirs: single_<task><suffix>")
    p.add_argument('--out', default=None, help='output json (default runs/comparison[_<suffix>].json)')
    a = p.parse_args()

    ids, mt = load_predictions(os.path.join(a.runs_dir, a.multitask, 'test_predictions.csv'), TASKS)
    st = {}
    for t in TASKS:
        i, d = load_predictions(os.path.join(a.runs_dir, f'single_{t}{a.single_suffix}', 'test_predictions.csv'), [t])
        if not (i == ids).all():
            raise SystemExit(f'single_{t} was evaluated on different exams than the multi-task model')
        st[t] = d[t]

    auc = lambda pred, t: roc_auc_score(pred[t][0], pred[t][1])            # noqa: E731
    point = {
        'multitask': {t: auc(mt, t) for t in TASKS},
        'single': {t: auc(st, t) for t in TASKS},
    }
    for k in ('multitask', 'single'):
        point[k]['mean'] = float(np.mean([point[k][t] for t in TASKS]))

    n = len(ids)
    rng = np.random.default_rng(a.seed)
    boot = {k: {t: [] for t in TASKS + ['mean']} for k in ('multitask', 'single', 'difference')}
    n_valid = 0
    for _ in range(a.n_bootstrap):
        idx = rng.integers(0, n, n)
        try:
            m = {t: roc_auc_score(mt[t][0][idx], mt[t][1][idx]) for t in TASKS}
            s = {t: roc_auc_score(st[t][0][idx], st[t][1][idx]) for t in TASKS}
        except ValueError:      # a resample without both classes for some task
            continue
        n_valid += 1
        for t in TASKS:
            boot['multitask'][t].append(m[t]); boot['single'][t].append(s[t])
            boot['difference'][t].append(m[t] - s[t])
        mm, ss = float(np.mean(list(m.values()))), float(np.mean(list(s.values())))
        boot['multitask']['mean'].append(mm); boot['single']['mean'].append(ss)
        boot['difference']['mean'].append(mm - ss)

    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    results = {
        'test_set': {'n': n, 'ids': [int(ids.min()), int(ids.max())],
                     'positives': {t: int(mt[t][0].sum()) for t in TASKS}},
        'bootstrap': {'n_resamples': a.n_bootstrap, 'n_valid': n_valid, 'seed': a.seed,
                      'method': 'paired percentile bootstrap over exams (same resample for both models)'},
        'multitask': {t: {'auc': point['multitask'][t], 'ci95': ci(boot['multitask'][t])} for t in TASKS + ['mean']},
        'single_task': {t: {'auc': point['single'][t], 'ci95': ci(boot['single'][t])} for t in TASKS + ['mean']},
        'difference_multi_minus_single': {
            t: {'delta': point['multitask'][t] - point['single'][t], 'ci95': ci(boot['difference'][t]),
                'excludes_zero': bool(ci(boot['difference'][t])[0] > 0 or ci(boot['difference'][t])[1] < 0)}
            for t in TASKS + ['mean']},
    }
    results['runs'] = {'multitask': a.multitask, 'single': [f'single_{t}{a.single_suffix}' for t in TASKS]}
    out = a.out or os.path.join(a.runs_dir, f'comparison{a.single_suffix or ""}.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'test set: n={n}, positives {results["test_set"]["positives"]}')
    for t in TASKS + ['mean']:
        m, s, d = results['multitask'][t], results['single_task'][t], results['difference_multi_minus_single'][t]
        print(f'{t:9s} multi {m["auc"]:.4f} [{m["ci95"][0]:.3f},{m["ci95"][1]:.3f}] | '
              f'single {s["auc"]:.4f} [{s["ci95"][0]:.3f},{s["ci95"][1]:.3f}] | '
              f'diff {d["delta"]:+.4f} [{d["ci95"][0]:+.3f},{d["ci95"][1]:+.3f}]'
              f'{"  *" if d["excludes_zero"] else ""}')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
