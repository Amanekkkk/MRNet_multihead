"""
Deferred TEST evaluation for runs trained in `official_test_pending` mode
(official validation images on disk, label csvs added later).

Loads `best_model.pt` of each run, rebuilds the identical train/val split,
re-evaluates the validation split (for the Youden thresholds) and evaluates the
official validation set (= TEST) once. Updates results.json in place.

Usage:
    python evaluate_test.py runs/multitask runs/single_abnormal ...
    python evaluate_test.py --all          # every runs/*/ with test_pending
"""
import argparse
import glob
import json
import os
import sys

import torch
import torch.nn as nn

from dataset import MRNetDataset, collate_single, make_splits
from model import MultiViewMRNet
from train import evaluate_test_split, json_clean, run_epoch, set_seed, setup_logger, write_predictions


def evaluate_run(run_dir: str, workers: int, force: bool) -> bool:
    res_path = os.path.join(run_dir, 'results.json')
    ckpt_path = os.path.join(run_dir, 'best_model.pt')
    if not os.path.isfile(res_path) or not os.path.isfile(ckpt_path):
        print(f'[skip] {run_dir}: results.json or best_model.pt missing')
        return False
    with open(res_path) as f:
        results = json.load(f)
    if results.get('test') and not force:
        print(f'[skip] {run_dir}: test results already present (use --force)')
        return False

    args = argparse.Namespace(**results['args'])
    args.out_dir = run_dir
    args.workers = workers
    log = setup_logger(os.path.join(run_dir, 'training.log')).info
    log(f'evaluate_test.py: deferred TEST evaluation of {run_dir}')
    set_seed(args.seed)

    split = make_splits(args.data_dir, seed=args.seed, limit=args.limit, log=log)
    if split['mode'] != 'official':
        raise SystemExit(f'official validation labels not found (mode={split["mode"]}); '
                         'add valid-*.csv (or val-*.csv) first')
    # the train/val split must be exactly the one the model was trained with
    for k in ('train', 'val'):
        if split['summary'][k]['n'] != results['split_sizes'][k] or \
                split['summary'][k]['positives'] != results['positives'][k]:
            raise SystemExit(f'{k} split differs from the one used in training -- aborting')

    tasks = results['tasks']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = bool(results['amp']) and device.type == 'cuda'
    pw = [results['pos_weight'][t] for t in tasks]
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device), reduction='mean')

    model = MultiViewMRNet(tasks=tasks, pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model_state'])
    log(f'Loaded best checkpoint (epoch {state["epoch"]})')

    loader_kw = dict(batch_size=1, collate_fn=collate_single, num_workers=workers,
                     pin_memory=device.type == 'cuda')
    val_ds = MRNetDataset(args.data_dir, split['entries']['val'], split['labels'], tasks)
    test_ds = MRNetDataset(args.data_dir, split['entries']['test'], split['labels'], tasks)
    va = run_epoch(model, torch.utils.data.DataLoader(val_ds, shuffle=False, **loader_kw),
                   criterion, device, amp)
    write_predictions(os.path.join(run_dir, 'val_predictions.csv'), va, tasks)
    test_block = evaluate_test_split(model, test_ds, va, criterion, device, amp, tasks,
                                     loader_kw, args, log)

    summ = split['summary']
    results.update({
        'split_mode': 'official',
        'split_sizes': {k: v['n'] for k, v in summ.items()},
        'positives': {k: v['positives'] for k, v in summ.items()},
        'sagittal_substituted': {k: v['sagittal_substituted'] for k, v in summ.items()},
        'test': test_block,
        'test_pending': False,
        'test_evaluated_by': 'evaluate_test.py (deferred; checkpoint and train/val split unchanged)',
    })
    with open(res_path, 'w') as f:
        json.dump(json_clean(results), f, indent=2)
    log(f'Updated {res_path}')
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dirs', nargs='*')
    p.add_argument('--all', action='store_true', help='all runs/*/ with a checkpoint')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    dirs = a.run_dirs or []
    if a.all:
        dirs += sorted(d for d in glob.glob(os.path.join(here, 'runs', '*')) if os.path.isdir(d))
    if not dirs:
        p.error('give run directories or --all')
    for d in dirs:
        evaluate_run(d, a.workers, a.force)


if __name__ == '__main__':
    sys.exit(main())
