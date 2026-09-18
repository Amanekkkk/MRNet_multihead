"""
Training / evaluation script for the shared-backbone multi-view MRNet model.

  * multi-task:  python train.py --tasks abnormal,acl,meniscus --out_dir runs/multitask
  * single-task: python train.py --tasks acl --out_dir runs/single_acl

Protocol
  - train split: optimisation (augmentation on by default)
  - val split:   model selection (best val mean AUC), early stopping, Youden thresholds
  - test split:  evaluated ONCE with the best checkpoint at the end of training
  (see dataset.py for how the splits are built in 'official' vs 'fallback' mode)

Loss
  The loss is the MEAN over tasks of the per-task binary cross-entropy with
  logits, each task using pos_weight = n_neg / n_pos computed on the TRAIN split
  (torch.nn.BCEWithLogitsLoss(pos_weight=..., reduction='mean') on a (1, T)
  logit vector). It is NOT a sum over tasks; for T=3 the two differ by a
  constant factor of 3, which only rescales the effective learning rate.
"""
import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from dataset import TASKS, MRNetDataset, VolumeAugment, collate_single, make_splits  # noqa: E402
from model import MultiViewMRNet  # noqa: E402

LOSS_DEFINITION = ('mean over tasks of BCEWithLogits(logit_t, y_t; pos_weight_t), '
                   'pos_weight_t = n_neg_t / n_pos_t on the train split (1.0 if --no_pos_weight); '
                   'one exam per forward pass, gradients averaged over `accum_steps` exams '
                   'per optimiser step (effective batch size)')
EARLY_STOPPING_RULE = ('select the epoch with the highest validation mean AUC (mean over the '
                       'trained tasks); stop when it has not improved for `patience` consecutive epochs '
                       '(only from epoch `min_epochs` onward) or at max_epochs')
BN_HANDLING = ('conv1/bn1/layer1/layer2 frozen: no gradient and BatchNorm kept in eval mode '
               '(ImageNet running statistics). BatchNorm in layer3/layer4 trains normally, with '
               'batch statistics over the slices of one plane of one exam; eval mode uses running '
               'statistics.')


# ═════════════════════════════════════════════════════════════════ utilities
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger('mrnet')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter('%(asctime)s  %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def safe_auc(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float('nan')


def fast_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties (== sklearn's ROC AUC)."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    r = rankdata(p)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def nanmean_or_nan(values):
    vals = [v for v in values if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float('nan')


def bootstrap_auc_ci(labels: np.ndarray, probs: np.ndarray, tasks, n_boot: int = 2000,
                     seed: int = 42, alpha: float = 0.05) -> dict:
    """Percentile bootstrap over exams (the same resample for every task, so the
    mean-AUC CI accounts for between-task correlation). Resamples in which a
    task has only one class are skipped for that task (and for the mean)."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    per_task = {t: [] for t in tasks}
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        aucs = [fast_auc(labels[idx, i], probs[idx, i]) for i in range(len(tasks))]
        for t, a in zip(tasks, aucs):
            if not math.isnan(a):
                per_task[t].append(a)
        if not any(math.isnan(a) for a in aucs):
            means.append(float(np.mean(aucs)))

    def ci(vals):
        if len(vals) < 10:
            return [float('nan'), float('nan')]
        return [float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))]

    out = {t: {'ci95': ci(v), 'n_valid_resamples': len(v)} for t, v in per_task.items()}
    out['mean'] = {'ci95': ci(means), 'n_valid_resamples': len(means)}
    return out


def youden_threshold(y, p) -> float:
    if len(np.unique(y)) < 2:
        return float('nan')
    fpr, tpr, thr = roc_curve(y, p)
    finite = np.isfinite(thr)
    j = np.where(finite, tpr - fpr, -np.inf)
    return float(thr[int(np.argmax(j))])


def sens_spec(y, p, thr) -> dict:
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return {
        'threshold': float(thr),
        'sensitivity': tp / (tp + fn) if tp + fn else float('nan'),
        'specificity': tn / (tn + fp) if tn + fp else float('nan'),
        'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp,
    }


def json_clean(o):
    """NaN -> None so results.json is strict JSON."""
    if isinstance(o, dict):
        return {k: json_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, (np.floating,)):
        return json_clean(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


# ══════════════════════════════════════════════════════════════ train / eval
def run_epoch(model, loader, criterion, device, amp: bool, optimizer=None, scaler=None,
              accum_steps: int = 1) -> dict:
    """One pass over `loader`. When training, gradients are averaged over
    `accum_steps` consecutive examinations before each optimiser step, so the
    effective batch is `accum_steps` examinations even though the slice count
    forces one examination per forward pass."""
    training = optimizer is not None
    model.train(training)
    total_loss, logits_all, labels_all, ids_all = 0.0, [], [], []
    n_batches = len(loader)
    if training:
        optimizer.zero_grad(set_to_none=True)

    with torch.set_grad_enabled(training):
        for step, (axial, coronal, sagittal, label, cid) in enumerate(loader):
            axial = axial.to(device, non_blocking=True)
            coronal = coronal.to(device, non_blocking=True)
            sagittal = sagittal.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(axial, coronal, sagittal)                      # (T,)
            # loss in float32; (1,T) -> mean over tasks (see module docstring)
            loss = criterion(logits.float().unsqueeze(0), label.unsqueeze(0))

            if training:
                # scale so the accumulated gradient is the MEAN over the group
                scaler.scale(loss / accum_steps).backward()
                last = (step + 1) == n_batches
                if (step + 1) % accum_steps == 0 or last:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            logits_all.append(logits.detach().float().cpu().numpy())
            labels_all.append(label.cpu().numpy())
            ids_all.append(int(cid))

    logits_all = np.stack(logits_all)
    labels_all = np.stack(labels_all).astype(int)
    probs = sigmoid(logits_all)
    tasks = model.tasks
    aucs = {t: safe_auc(labels_all[:, i], probs[:, i]) for i, t in enumerate(tasks)}
    return {
        'loss': total_loss / max(len(loader), 1),
        'aucs': aucs,
        'mean_auc': nanmean_or_nan(aucs.values()),
        'probs': probs,
        'labels': labels_all,
        'ids': ids_all,
    }


# ══════════════════════════════════════════════════════════════════ plotting
TASK_COLORS = {'abnormal': '#1f77b4', 'acl': '#d62728', 'meniscus': '#2ca02c'}


def save_plots(history: dict, tasks, best_epoch: int, out_dir: str):
    ep = history['epoch']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ep, history['train_loss'], label='Train', lw=2)
    ax.plot(ep, history['val_loss'], label='Validation', lw=2)
    ax.axvline(best_epoch, color='gray', ls='--', lw=1.2, label=f'Best epoch ({best_epoch})')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (mean weighted BCE over tasks)')
    ax.set_title('Training and validation loss'); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'loss.png'), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for t in tasks:
        ax.plot(ep, history[f'val_auc_{t}'], color=TASK_COLORS[t], lw=1.8, label=f'Val {t}')
        ax.plot(ep, history[f'train_auc_{t}'], color=TASK_COLORS[t], lw=1, ls='--', alpha=0.5, label=f'Train {t}')
    if len(tasks) > 1:
        ax.plot(ep, history['val_auc_mean'], color='black', lw=2.2, label='Val mean')
    ax.axvline(best_epoch, color='gray', ls='--', lw=1.2, label=f'Best epoch ({best_epoch})')
    ax.set_xlabel('Epoch'); ax.set_ylabel('ROC AUC'); ax.set_ylim(0.3, 1.02)
    ax.set_title('Training and validation AUC'); ax.grid(alpha=0.3); ax.legend(ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'auc.png'), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ep, history['lr'], lw=2, color='#7b3294')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Learning rate'); ax.set_title('Cosine-annealing learning rate')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0)); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'lr.png'), dpi=150); plt.close(fig)


def write_predictions(path, res, tasks):
    # probabilities are written at full precision: rounding them creates ties
    # between exams, which changes the AUC computed from this file
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id'] + [c for t in tasks for c in (f'{t}_label', f'{t}_prob')])
        for k, cid in enumerate(res['ids']):
            w.writerow([f'{cid:04d}'] + [v for i in range(len(tasks))
                                         for v in (int(res['labels'][k, i]), repr(float(res['probs'][k, i])))])


# ══════════════════════════════════════════════════════════════════════ main
def evaluate_test_split(model, test_ds, va, criterion, device, amp, tasks, loader_kw, args, log) -> dict:
    """Evaluate the (already loaded) best checkpoint ONCE on the test split: AUCs with
    bootstrap CIs, sensitivity/specificity at 0.5 and at the Youden threshold chosen on
    the validation predictions `va`. Writes test_predictions.csv, returns the test block."""
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)
    te = run_epoch(model, test_loader, criterion, device, amp)
    write_predictions(os.path.join(args.out_dir, 'test_predictions.csv'), te, tasks)

    boot = bootstrap_auc_ci(te['labels'], te['probs'], tasks, n_boot=args.n_bootstrap, seed=args.seed)
    test_block = {'loss': te['loss'], 'n': len(te['ids']), 'auc': {}, 'thresholds': {}}
    for i, t in enumerate(tasks):
        y, p = te['labels'][:, i], te['probs'][:, i]
        thr_y = youden_threshold(va['labels'][:, i], va['probs'][:, i])
        test_block['auc'][t] = {'auc': te['aucs'][t], **boot[t]}
        test_block['thresholds'][t] = {
            'at_0.5': sens_spec(y, p, 0.5),
            'at_val_youden': sens_spec(y, p, thr_y) if not math.isnan(thr_y) else None,
            'val_youden_threshold': thr_y,
        }
    test_block['auc']['mean'] = {'auc': te['mean_auc'], **boot['mean']}
    test_block['bootstrap'] = {'n_resamples': args.n_bootstrap, 'seed': args.seed,
                               'method': 'percentile, resampling exams with replacement, 95%'}

    log('TEST results (best checkpoint, evaluated once):')
    for t in tasks + ['mean']:
        a = test_block['auc'][t]
        log(f'  AUC {t:9s} {a["auc"]:.4f}  95% CI [{a["ci95"][0]:.4f}, {a["ci95"][1]:.4f}]')
    for t in tasks:
        th = test_block['thresholds'][t]
        s5 = th['at_0.5']
        msg = f'  {t:9s} @0.5: sens {s5["sensitivity"]:.3f} spec {s5["specificity"]:.3f}'
        if th['at_val_youden']:
            sy = th['at_val_youden']
            msg += f' | @Youden(val)={sy["threshold"]:.3f}: sens {sy["sensitivity"]:.3f} spec {sy["specificity"]:.3f}'
        log(msg)
    return test_block


def main(args):
    t_start = time.time()
    set_seed(args.seed)
    tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.out_dir, 'training.log'))
    log = logger.info
    log(f'Command: {" ".join(sys.argv)}')
    log(f'Args: {vars(args)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = (not args.no_amp) and device.type == 'cuda'
    gpu_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else None
    log(f'Device: {device} ({gpu_name}) | AMP: {amp} | torch {torch.__version__}')
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    # ── data ────────────────────────────────────────────────────────────────
    split = make_splits(args.data_dir, seed=args.seed, limit=args.limit, log=log)
    augmenter = VolumeAugment()
    ds = {
        'train': MRNetDataset(args.data_dir, split['entries']['train'], split['labels'], tasks,
                              augment=args.augment, augmenter=augmenter),
        'val': MRNetDataset(args.data_dir, split['entries']['val'], split['labels'], tasks),
        'test': MRNetDataset(args.data_dir, split['entries']['test'], split['labels'], tasks),
    }
    log(f'Augmentation: {augmenter.config() if args.augment else "off"}')

    g = torch.Generator(); g.manual_seed(args.seed)
    loader_kw = dict(batch_size=1, collate_fn=collate_single, num_workers=args.workers,
                     pin_memory=device.type == 'cuda')
    if args.workers > 0:
        loader_kw.update(persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(ds['train'], shuffle=True, generator=g, **loader_kw)
    val_loader = DataLoader(ds['val'], shuffle=False, **loader_kw)

    # ── pos_weight from TRAIN split ─────────────────────────────────────────
    train_labels = np.array([[split['labels'][c][TASKS.index(t)] for t in tasks]
                             for c, _ in split['entries']['train']])
    n_pos = train_labels.sum(0); n_neg = len(train_labels) - n_pos
    if args.pos_weight:
        pw = [float(n_neg[i] / n_pos[i]) if n_pos[i] > 0 else 1.0 for i in range(len(tasks))]
    else:
        pw = [1.0] * len(tasks)
    pos_weight = dict(zip(tasks, pw))
    log(f'pos_weight (neg/pos on train): {pos_weight}')
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device), reduction='mean')

    # ── model / optimisation ───────────────────────────────────────────────
    model = MultiViewMRNet(tasks=tasks, pretrained=True).to(device)
    params = model.count_parameters()
    log(f'Model: shared ResNet50, tasks={tasks} | params total={params["total"]:,} '
        f'trainable={params["trainable"]:,} frozen={params["frozen"]:,}')
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.min_lr)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    csv_cols = (['epoch', 'lr', 'epoch_time_s', 'train_time_s', 'val_time_s', 'peak_gpu_mem_mb',
                 'peak_gpu_reserved_mb', 'train_loss', 'train_auc_mean']
                + [f'train_auc_{t}' for t in tasks] + ['val_loss', 'val_auc_mean']
                + [f'val_auc_{t}' for t in tasks])
    history = {c: [] for c in csv_cols}
    csv_f = open(os.path.join(args.out_dir, 'metrics.csv'), 'w', newline='')
    csv_w = csv.writer(csv_f); csv_w.writerow(csv_cols)

    ckpt_path = os.path.join(args.out_dir, 'best_model.pt')
    best_auc, best_epoch, best_val = -math.inf, None, None
    stopped_early = False

    for epoch in range(1, args.max_epochs + 1):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        lr_now = optimizer.param_groups[0]['lr']
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, device, amp, optimizer, scaler,
                       accum_steps=args.accum_steps)
        t1 = time.time()
        va = run_epoch(model, val_loader, criterion, device, amp)
        t2 = time.time()
        scheduler.step()
        peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else float('nan')
        peak_res = torch.cuda.max_memory_reserved() / 2**20 if device.type == 'cuda' else float('nan')

        row = {'epoch': epoch, 'lr': lr_now, 'epoch_time_s': t2 - t0, 'train_time_s': t1 - t0,
               'val_time_s': t2 - t1, 'peak_gpu_mem_mb': peak, 'peak_gpu_reserved_mb': peak_res,
               'train_loss': tr['loss'], 'train_auc_mean': tr['mean_auc'],
               'val_loss': va['loss'], 'val_auc_mean': va['mean_auc']}
        for t in tasks:
            row[f'train_auc_{t}'] = tr['aucs'][t]
            row[f'val_auc_{t}'] = va['aucs'][t]
        for c in csv_cols:
            history[c].append(row[c])
        csv_w.writerow([row['epoch']] + [f'{row[c]:.6g}' for c in csv_cols[1:]]); csv_f.flush()

        fmt = lambda d: ' '.join(f'{t[:3]}={d[t]:.3f}' for t in tasks)  # noqa: E731
        log(f'Epoch {epoch:03d}/{args.max_epochs} | lr {lr_now:.2e} | '
            f'train loss {tr["loss"]:.4f} AUC {tr["mean_auc"]:.4f} [{fmt(tr["aucs"])}] | '
            f'val loss {va["loss"]:.4f} AUC {va["mean_auc"]:.4f} [{fmt(va["aucs"])}] | '
            f'{t2 - t0:.0f}s (train {t1 - t0:.0f}s) | peak mem {peak:.0f} MB')

        score = va['mean_auc'] if not math.isnan(va['mean_auc']) else -math.inf
        if best_epoch is None or score > best_auc:
            best_auc, best_epoch = score, epoch
            best_val = {'loss': va['loss'], 'aucs': va['aucs'], 'mean_auc': va['mean_auc']}
            torch.save({'epoch': epoch, 'tasks': tasks, 'model_state': model.state_dict(),
                        'val_auc_mean': va['mean_auc'], 'val_aucs': va['aucs'], 'args': vars(args)},
                       ckpt_path)
            log(f'  -> new best val mean AUC {va["mean_auc"]:.4f} (epoch {epoch}), checkpoint saved')
        elif epoch >= args.min_epochs and epoch - best_epoch >= args.patience:
            stopped_early = True
            log(f'Early stopping: no val mean AUC improvement for {args.patience} epochs '
                f'(best epoch {best_epoch}, AUC {best_auc:.4f})')
            break
    csv_f.close()
    epochs_run = len(history['epoch'])
    save_plots(history, tasks, best_epoch, args.out_dir)

    # ── final evaluation with the best checkpoint ──────────────────────────
    del train_loader  # release persistent workers
    log(f'Loading best checkpoint (epoch {best_epoch}) for final evaluation')
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model_state'])
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    va = run_epoch(model, val_loader, criterion, device, amp)
    write_predictions(os.path.join(args.out_dir, 'val_predictions.csv'), va, tasks)
    if len(ds['test']) > 0:
        test_block = evaluate_test_split(model, ds['test'], va, criterion, device, amp, tasks,
                                         loader_kw, args, log)
    else:
        test_block = None
        log('TEST evaluation deferred: official validation labels not on disk. '
            'Run `python evaluate_test.py runs/<name>` once valid-*.csv are added.')
    eval_peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else float('nan')

    summ = split['summary']
    results = {
        'split_mode': split['mode'],
        'stratification': split['stratification'],
        'split_sizes': {k: v['n'] for k, v in summ.items()},
        'positives': {k: v['positives'] for k, v in summ.items()},
        'sagittal_substituted': {k: v['sagittal_substituted'] for k, v in summ.items()},
        'tasks': tasks,
        'params': params,
        'pos_weight': pos_weight,
        'loss_definition': LOSS_DEFINITION,
        'effective_batch_size': args.accum_steps,
        'early_stopping_rule': EARLY_STOPPING_RULE,
        'bn_handling': BN_HANDLING,
        'augmentation': augmenter.config() if args.augment else None,
        'input_preprocessing': 'per-volume min-max to [0,1], grey->3 channels, ImageNet mean/std normalisation, 256x256',
        'best_epoch': best_epoch,
        'epochs_run': epochs_run,
        'stopped_early': stopped_early,
        'val_at_best_epoch': best_val,
        'val_best_checkpoint_reeval': {'loss': va['loss'], 'aucs': va['aucs'], 'mean_auc': va['mean_auc']},
        'test': test_block,
        'test_pending': test_block is None,
        'timing': {
            'mean_epoch_time_s': float(np.mean(history['epoch_time_s'])),
            'mean_epoch_time_s_excl_first': (float(np.mean(history['epoch_time_s'][1:]))
                                             if epochs_run > 1 else None),
            'mean_train_time_s': float(np.mean(history['train_time_s'])),
            'mean_val_time_s': float(np.mean(history['val_time_s'])),
            'mean_train_s_per_exam': float(np.mean(history['train_time_s']) / max(len(ds['train']), 1)),
            'total_wall_time_s': time.time() - t_start,
        },
        'memory': {
            'peak_gpu_mem_mb_train_epochs_max': float(np.nanmax(history['peak_gpu_mem_mb'])) if device.type == 'cuda' else None,
            'peak_gpu_mem_mb_train_epochs_mean': float(np.nanmean(history['peak_gpu_mem_mb'])) if device.type == 'cuda' else None,
            'peak_gpu_reserved_mb_max': float(np.nanmax(history['peak_gpu_reserved_mb'])) if device.type == 'cuda' else None,
            'peak_gpu_mem_mb_final_eval': eval_peak,
            'measure': 'torch.cuda.max_memory_allocated, reset at the start of every epoch (train+val)',
        },
        'amp': amp,
        'environment': {'torch': torch.__version__, 'gpu': gpu_name, 'cudnn_benchmark': args.cudnn_benchmark},
        'args': vars(args),
    }
    with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
        json.dump(json_clean(results), f, indent=2)
    log(f'Wrote {os.path.join(args.out_dir, "results.json")} | total time {(time.time() - t_start) / 60:.1f} min')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Shared-backbone multi-view MRNet (multi-/single-task)')
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument('--data_dir', default=here)
    p.add_argument('--tasks', default='abnormal,acl,meniscus', help='comma list, subset of abnormal,acl,meniscus')
    p.add_argument('--out_dir', default=os.path.join(here, 'runs', 'multitask'))
    p.add_argument('--max_epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=20, help='early stopping patience on val mean AUC')
    p.add_argument('--min_epochs', type=int, default=0,
                   help='early stopping cannot trigger before this epoch (slow-starting runs)')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--min_lr', type=float, default=1e-6, help='cosine annealing eta_min')
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--augment', dest='augment', action='store_true', default=True)
    p.add_argument('--no_augment', dest='augment', action='store_false')
    p.add_argument('--pos_weight', dest='pos_weight', action='store_true', default=True)
    p.add_argument('--no_pos_weight', dest='pos_weight', action='store_false')
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--cudnn_benchmark', dest='cudnn_benchmark', action='store_true', default=False,
                   help='off by default: measured no speed-up with variable slice counts')
    p.add_argument('--no_cudnn_benchmark', dest='cudnn_benchmark', action='store_false')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--limit', type=int, default=None, help='use only the first N exams of each split (smoke tests)')
    p.add_argument('--accum_steps', type=int, default=1,
                   help='gradient accumulation: number of examinations averaged per optimiser step '
                        '(effective batch size; 1 = update after every examination)')
    p.add_argument('--n_bootstrap', type=int, default=2000)
    args = p.parse_args(argv)
    bad = [t for t in args.tasks.split(',') if t.strip() not in TASKS]
    if bad:
        p.error(f'unknown task(s) {bad}; choose from {TASKS}')
    return args


if __name__ == '__main__':
    main(parse_args())
