"""
Generates the paper figures into figures_v2/ (no "Figure N." text is baked
into any image; numbering belongs in the manuscript captions).

  architecture.png         shared-backbone multi-view multi-task architecture
  samples.png              one exam per class, highest-variance slice per plane
                           (block-averaged variance, central half of the slices)
  training_loss.png        from runs/multitask/metrics.csv          (if present)
  training_auc.png         from runs/multitask/metrics.csv          (if present)
  learning_rate.png        from runs/multitask/metrics.csv          (if present)
  test_auc_comparison.png  multitask vs single-task test AUC + 95% CI
                           from runs/*/results.json                 (if present)

Usage:  python generate_figures.py [--only architecture,samples,curves,comparison]
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures_v2')
RUNS = os.path.join(HERE, 'runs')
# run directories of the FINAL protocol (gradient accumulation, effective batch 8, lr 8e-5);
# override with MRNET_MULTITASK_RUN / MRNET_SINGLE_SUFFIX
MULTITASK_RUN = os.environ.get('MRNET_MULTITASK_RUN', 'multitask_accum8_lr8e-5')
SINGLE_SUFFIX = os.environ.get('MRNET_SINGLE_SUFFIX', '_b8')
TASKS = ['abnormal', 'acl', 'meniscus']
TASK_LABEL = {'abnormal': 'Abnormality', 'acl': 'ACL tear', 'meniscus': 'Meniscal tear', 'mean': 'Mean'}
VIEWS = ['axial', 'coronal', 'sagittal']

# categorical slots (fixed order): blue, orange, aqua; text stays in neutral ink
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
TASK_COLOR = {'abnormal': C1, 'acl': C2, 'meniscus': C3}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
    'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'legend.frameon': False, 'lines.linewidth': 2,
})


def save(fig, name, **kw):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches='tight', **kw)
    plt.close(fig)
    print(f'  saved {os.path.relpath(path, HERE)}')


# ═══════════════════════════════════════════════════════════════ architecture
def make_architecture():
    fig, ax = plt.subplots(figsize=(15, 6.6))
    ax.set_xlim(0, 15.4); ax.set_ylim(0, 6.8); ax.axis('off')

    col = {'input': '#d6e6f8', 'backbone': '#cdeede', 'proj': '#e6f4ec', 'pool': '#fbe9c4',
           'fuse': '#f9d9cb', 'head': '#e4def5'}

    def box(x, y, w, h, fc, text, sub=None, fs=9, lw=1.1, bold=True):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.03,rounding_size=0.08',
                                    fc=fc, ec='#3a3a38', lw=lw))
        ty = y + h / 2 + (0.16 if sub else 0)
        ax.text(x + w / 2, ty, text, ha='center', va='center', fontsize=fs,
                fontweight='bold' if bold else 'normal', color=INK)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.2, sub, ha='center', va='center', fontsize=7.5,
                    color=INK2, style='italic')

    def arrow(x1, y1, x2, y2, **kw):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='-|>', color='#3a3a38', lw=1.3, shrinkA=0, shrinkB=0, **kw))

    ys = {'axial': 4.9, 'coronal': 3.3, 'sagittal': 1.7}   # centre line of each plane stream
    h = 0.9

    # inputs
    for v, y in ys.items():
        box(0.1, y - h / 2, 1.75, h, col['input'], v.capitalize(), 'S × 256 × 256', fs=9)

    # ONE shared backbone spanning all three streams
    bx, bw = 2.45, 2.75
    by0, by1 = ys['sagittal'] - 0.75, ys['axial'] + 0.75
    ax.add_patch(FancyBboxPatch((bx, by0), bw, by1 - by0, boxstyle='round,pad=0.03,rounding_size=0.12',
                                fc=col['backbone'], ec='#1c5b3f', lw=2.0))
    ax.text(bx + bw / 2, by1 + 0.55, 'ONE shared ResNet50', ha='center', va='center',
            fontsize=11, fontweight='bold', color=INK)
    ax.text(bx + bw / 2, by1 + 0.24, '(ImageNet weights; same weights for all planes)',
            ha='center', va='center', fontsize=7.5, color=INK2)
    for v, y in ys.items():
        # each plane passes through the same network separately (slice-wise)
        arrow(1.85, y, bx + 0.02, y)
        ax.plot([bx + 0.12, bx + bw - 0.12], [y, y], color='#1c5b3f', lw=1.0, ls=(0, (4, 3)), alpha=0.7)
        ax.text(bx + bw / 2, y + 0.2, f'{v} slices', ha='center', va='bottom', fontsize=7.5, color=INK2)
    ax.text(bx + bw / 2, ys['coronal'] - 0.6,
            'conv1 · layer1–2: frozen (BN eval)\nlayer3–4: fine-tuned\nglobal avg-pool → 2048-d per slice',
            ha='center', va='center', fontsize=7.3, color=INK2, linespacing=1.35)

    # per-plane projection + max-pool
    px, pw = 5.8, 1.55
    mx, mw = 7.75, 1.45
    for v, y in ys.items():
        arrow(bx + bw, y, px, y)
        box(px, y - 0.36, pw, 0.72, col['proj'], 'Linear 2048→256', f'ReLU · {v} proj.', fs=8)
        arrow(px + pw, y, mx, y)
        box(mx, y - 0.36, mw, 0.72, col['pool'], 'Max-pool', 'over slices → 256', fs=8)

    # concat
    cx, cw = 9.75, 0.95
    cy0, cy1 = ys['sagittal'] - 0.45, ys['axial'] + 0.45
    box(cx, cy0, cw, cy1 - cy0, col['fuse'], 'Concat', '768', fs=9)
    for y in ys.values():
        arrow(mx + mw, y, cx, y)

    # shared FC
    fx, fw = 11.05, 1.5
    fy = ys['coronal']
    box(fx, fy - 0.55, fw, 1.1, col['fuse'], 'FC 768→256', 'ReLU · Dropout 0.5', fs=8.5)
    arrow(cx + cw, fy, fx, fy)

    # heads
    hx, hw = 13.0, 2.2
    head_y = {'abnormal': 4.5, 'acl': 3.3, 'meniscus': 2.1}
    for t, y in head_y.items():
        box(hx, y - 0.34, hw, 0.68, col['head'], TASK_LABEL[t], 'Linear 256→1 → logit', fs=8.5)
        ax.annotate('', xy=(hx, y), xytext=(fx + fw, fy),
                    arrowprops=dict(arrowstyle='-|>', color='#3a3a38', lw=1.2, shrinkA=0, shrinkB=0,
                                    connectionstyle='arc3,rad=0'))

    ax.text(hx + hw / 2, 1.3, 'Single-task baselines:\nidentical network with one head',
            ha='center', va='top', fontsize=7.5, color=INK2, style='italic')
    ax.text(7.5, 0.25, 'Loss: mean over tasks of per-task BCE with pos_weight = neg/pos (training split). '
            'Batch = 1 exam; each plane has a variable number of slices S.',
            ha='center', va='center', fontsize=8, color=INK2)
    save(fig, 'architecture.png', facecolor='white')


# ════════════════════════════════════════════════════════════════════ samples
def _labels():
    lab = {}
    for t in TASKS:
        with open(os.path.join(HERE, f'train-{t}.csv')) as f:
            for r in csv.reader(f):
                lab.setdefault(int(r[0]), {})[t] = int(r[1])
    return lab


# Exam 0003 (first meniscal-tear exam) has a mis-positioned sagittal series: the
# knee is almost entirely outside the field of view and most slices are noise, so
# it is not used as an illustrative example.
EXCLUDE_SAMPLE_IDS = {3}
SAMPLE_IDS = None     # or e.g. [11, 46, 5] to fix the exams shown


def representative_slice(vol: np.ndarray, block: int = 4, central_frac: float = 0.5) -> int:
    """Index of the slice with the highest intensity variance, computed on a
    block-averaged (block x block) copy so that pixel noise does not dominate,
    and restricted to the central `central_frac` of the slices (edge slices
    are mostly subcutaneous fat or empty field of view)."""
    S, H, W = vol.shape
    lo = int(round(S * (1 - central_frac) / 2)); hi = max(lo + 1, S - lo)
    v = vol[lo:hi, :H // block * block, :W // block * block].astype(np.float32)
    v = v.reshape(hi - lo, H // block, block, W // block, block).mean(axis=(2, 4))
    return lo + int(np.argmax(v.reshape(hi - lo, -1).var(axis=1)))


def make_samples():
    labels = _labels()

    def has_all(c):
        return all(os.path.isfile(os.path.join(HERE, 'train', v, f'{c:04d}.npy')) for v in VIEWS)

    def pick(abn, acl, men):
        cands = [c for c, l in sorted(labels.items())
                 if (l['abnormal'], l['acl'], l['meniscus']) == (abn, acl, men) and c not in EXCLUDE_SAMPLE_IDS]
        full = [c for c in cands if has_all(c)]
        return (full or cands)[0]

    ids = SAMPLE_IDS or [pick(0, 0, 0), pick(1, 1, 0), pick(1, 0, 1)]
    names = ['Normal', 'ACL tear', 'Meniscal tear']
    cases = list(zip(ids, names))

    fig, axes = plt.subplots(3, 3, figsize=(9.2, 9.8))
    fig.patch.set_facecolor('#111111')
    for ci, (cid, name) in enumerate(cases):
        l = labels[cid]
        for ri, view in enumerate(VIEWS):
            ax = axes[ri][ci]
            ax.grid(False)
            path = os.path.join(HERE, 'train', view, f'{cid:04d}.npy')
            substituted = not os.path.isfile(path)
            if substituted:   # only when the sagittal file truly is missing
                path = os.path.join(HERE, 'train', 'coronal', f'{cid:04d}.npy')
            vol = np.load(path).astype(np.float32)
            k = representative_slice(vol)
            ax.imshow(vol[k], cmap='gray', vmin=vol[k].min(), vmax=vol[k].max())
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True); s.set_edgecolor('#444444')
            ax.text(0.03, 0.03, f'slice {k + 1}/{len(vol)}', transform=ax.transAxes, color='white',
                    fontsize=7.5, ha='left', va='bottom',
                    bbox=dict(fc='black', ec='none', alpha=0.55, pad=1.5))
            if substituted:
                ax.text(0.97, 0.97, 'coronal substituted', transform=ax.transAxes, color='#ffd166',
                        fontsize=8, ha='right', va='top', fontweight='bold',
                        bbox=dict(fc='black', ec='none', alpha=0.7, pad=2))
            if ri == 0:
                ax.set_title(f'{name}  (exam {cid:04d})\nabnormal={l["abnormal"]}, ACL={l["acl"]}, '
                             f'meniscus={l["meniscus"]}', color='white', fontsize=9.5, pad=6)
            if ci == 0:
                ax.set_ylabel(view.capitalize(), color='white', fontsize=11, fontweight='bold', labelpad=8)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    save(fig, 'samples.png', facecolor=fig.get_facecolor())


# ═════════════════════════════════════════════════════════════ training curves
def _read_metrics(run):
    path = os.path.join(RUNS, run, 'metrics.csv')
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return {k: np.array([float(r[k]) if r[k] not in ('', 'nan') else np.nan for r in rows])
            for k in rows[0]}


def _read_results(run):
    path = os.path.join(RUNS, run, 'results.json')
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def make_curves():
    m = _read_metrics(MULTITASK_RUN)
    if m is None:
        print('  skip training curves: metrics.csv of the multi-task run not found')
        return
    res = _read_results(MULTITASK_RUN)
    ep = m['epoch']
    best = res['best_epoch'] if res else int(ep[np.nanargmax(m['val_auc_mean'])])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(ep, m['train_loss'], color=C1, label='Training')
    ax.plot(ep, m['val_loss'], color=C2, label='Validation')
    ax.axvline(best, color=INK2, lw=1.2, ls='--', label=f'Selected epoch ({best})')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (mean weighted BCE over tasks)')
    ax.set_title('Training and validation loss', color=INK, loc='left')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    save(fig, 'training_loss.png', facecolor='white')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    ax = axes[0]
    for t in TASKS:
        ax.plot(ep, m[f'val_auc_{t}'], color=TASK_COLOR[t], label=TASK_LABEL[t])
    ax.plot(ep, m['val_auc_mean'], color=INK, lw=2.4, label='Mean')
    ax.axvline(best, color=INK2, lw=1.2, ls='--', label=f'Selected epoch ({best})')
    ax.set_xlabel('Epoch'); ax.set_ylabel('ROC AUC')
    lo = np.nanmin([np.nanmin(m[f'val_auc_{t}']) for t in TASKS] + [np.nanmin(m['train_auc_mean'])])
    ax.set_ylim(min(0.5, np.floor(lo * 10) / 10), 1.01)
    ax.set_title('Validation AUC per task', color=INK, loc='left'); ax.legend(fontsize=8.5)
    ax = axes[1]
    ax.plot(ep, m['train_auc_mean'], color=C1, label='Training (augmented, dropout on)')
    ax.plot(ep, m['val_auc_mean'], color=C2, label='Validation')
    ax.axvline(best, color=INK2, lw=1.2, ls='--', label=f'Selected epoch ({best})')
    ax.set_xlabel('Epoch'); ax.set_title('Mean AUC: training vs validation', color=INK, loc='left')
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    for a_ in axes:
        a_.xaxis.set_major_locator(MaxNLocator(integer=True))
    save(fig, 'training_auc.png', facecolor='white')

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(ep, m['lr'], color=C1)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Learning rate')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.set_title('Cosine-annealing learning rate', color=INK, loc='left')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    save(fig, 'learning_rate.png', facecolor='white')


# ═════════════════════════════════════════════════════════ test AUC comparison
def _single_task_mean_ci(n_boot=2000, seed=42):
    """Mean of the three single-task test AUCs with a bootstrap CI obtained by
    resampling the same exams for all three single-task prediction files."""
    from sklearn.metrics import roc_auc_score
    preds = {}
    for t in TASKS:
        path = os.path.join(RUNS, f'single_{t}{SINGLE_SUFFIX}', 'test_predictions.csv')
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            preds[t] = {r['id']: (int(r[f'{t}_label']), float(r[f'{t}_prob'])) for r in csv.DictReader(f)}
    ids = sorted(set.intersection(*(set(p) for p in preds.values())))
    y = np.array([[preds[t][i][0] for t in TASKS] for i in ids])
    p = np.array([[preds[t][i][1] for t in TASKS] for i in ids])
    point = float(np.mean([roc_auc_score(y[:, k], p[:, k]) for k in range(3)]))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(ids), len(ids))
        if all(len(np.unique(y[idx, k])) == 2 for k in range(3)):
            vals.append(np.mean([roc_auc_score(y[idx, k], p[idx, k]) for k in range(3)]))
    return {'auc': point, 'ci95': [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}


def make_comparison():
    multi = _read_results(MULTITASK_RUN)
    singles = {t: _read_results(f'single_{t}{SINGLE_SUFFIX}') for t in TASKS}
    if multi is None and not any(singles.values()):
        print('  skip test AUC comparison: no runs/*/results.json found')
        return
    groups = TASKS + ['mean']
    series = []
    if multi is not None:
        series.append(('Multi-task (one model)', C1,
                       {g: multi['test']['auc'].get(g) for g in groups}))
    if any(singles.values()):
        sd = {t: (singles[t]['test']['auc'][t] if singles[t] else None) for t in TASKS}
        sd['mean'] = _single_task_mean_ci()
        series.append(('Single-task (one model per task)', C2, sd))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(groups))
    step = 0.3
    lows = [0.5]
    for si, (name, color, d) in enumerate(series):
        off = (si - (len(series) - 1) / 2) * step
        first = True
        for gi, g in enumerate(groups):
            e = d.get(g)
            if not e or e.get('auc') is None:
                continue
            a = e['auc']
            lo, hi = e.get('ci95') or [None, None]
            if lo is not None and hi is not None:
                ax.errorbar(x[gi] + off, a, yerr=[[a - lo], [hi - a]], fmt='none', ecolor=color,
                            elinewidth=2, capsize=4)
                lows.append(lo)
            ax.plot(x[gi] + off, a, 'o', ms=8, color=color, mec='white', mew=1.5,
                    label=name if first else None, zorder=3)
            first = False
            ax.annotate(f'{a:.3f}', (x[gi] + off, a), xytext=(7, 0), textcoords='offset points',
                        ha='left', va='center', fontsize=8, color=INK)
    ax.axhline(0.5, color=INK2, lw=1, ls=':', zorder=1)
    ax.text(len(groups) - 0.5, 0.5, 'chance', ha='right', va='bottom', fontsize=8, color=INK2)
    ax.set_xticks(x, [TASK_LABEL[g] for g in groups])
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(max(0.0, np.floor(min(lows) * 20) / 20 - 0.02), 1.02)
    ax.set_ylabel('Test ROC AUC (95% bootstrap CI)')
    ax.grid(axis='x', visible=False)
    ref = multi or next(r for r in singles.values() if r)
    ax.set_title(f'Held-out test set ({ref["split_mode"]} split, n = {ref["split_sizes"]["test"]})',
                 color=INK, loc='left', pad=28)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=len(series), fontsize=9,
              handletextpad=0.3, borderaxespad=0.2)
    save(fig, 'test_auc_comparison.png', facecolor='white')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='architecture,samples,curves,comparison')
    ap.add_argument('--runs_dir', default=RUNS)
    ap.add_argument('--out_dir', default=OUT)
    a = ap.parse_args()
    RUNS, OUT = a.runs_dir, a.out_dir
    todo = set(a.only.split(','))
    print(f'Generating figures into {OUT}')
    for key, fn in [('architecture', make_architecture), ('samples', make_samples),
                    ('curves', make_curves), ('comparison', make_comparison)]:
        if key in todo:
            fn()
