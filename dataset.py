"""
MRNet-v1.0 data loading, train/val/test splitting and volume augmentation.

Split modes
-----------
* official : `valid/` (or `val/`) `{axial,coronal,sagittal}` and
             `valid-{abnormal,acl,meniscus}.csv` (or `val-*.csv`) are present. The 1130 official training exams are split 85/15 into
             train / val (val is used ONLY for model selection, early stopping and
             threshold choice) and the 120 official validation exams
             (ids 1130-1249) are the held-out TEST set.
* official_test_pending : validation images present but label csvs missing.
             Same train/val split as `official`; no test split, the test
             evaluation is run later with evaluate_test.py.
* fallback : no official validation data on disk. The 1130 training exams are
             split 70/15/15 into train / val / test.

Splits are stratified on the joint label combination (abnormal, acl, meniscus);
if a stratum is too small for sklearn to stratify, we fall back to stratifying
on `abnormal` only, and finally to an unstratified split. The seed is fixed.

Missing sagittal volumes (incomplete download) are replaced by the coronal
volume of the same exam as a last-resort fallback; the number of affected exams
per split is reported.
"""
import csv
import math
import os
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

VIEWS = ('axial', 'coronal', 'sagittal')
TASKS = ('abnormal', 'acl', 'meniscus')

Entry = Tuple[int, str]          # (exam id, sub-directory: 'train' | 'valid')


# ════════════════════════════════════════════════════════════════════ labels
def load_labels(data_dir: str, prefix: str = 'train') -> Dict[int, List[int]]:
    """Returns {exam_id: [abnormal, acl, meniscus]} from `<prefix>-<task>.csv`
    (no header, rows `id,label`)."""
    per_task: Dict[str, Dict[int, int]] = {}
    for task in TASKS:
        with open(os.path.join(data_dir, f'{prefix}-{task}.csv')) as f:
            per_task[task] = {int(r[0]): int(r[1]) for r in csv.reader(f) if r}
    ids = set(per_task[TASKS[0]])
    for task in TASKS[1:]:
        if set(per_task[task]) != ids:
            raise ValueError(f'{prefix}-*.csv files do not list the same exam ids')
    return {cid: [per_task[t][cid] for t in TASKS] for cid in sorted(ids)}


VALID_DIR_NAMES = ('valid', 'val')          # Redivis export uses `val/`
VALID_CSV_PREFIXES = ('valid', 'val')


def validation_dir(data_dir: str) -> Optional[str]:
    """Name of the official validation image directory, or None."""
    for name in VALID_DIR_NAMES:
        if all(os.path.isdir(os.path.join(data_dir, name, v)) for v in VIEWS):
            return name
    return None


def validation_csv_prefix(data_dir: str) -> Optional[str]:
    """Prefix of the official validation label csvs (`<prefix>-<task>.csv`), or None."""
    for prefix in VALID_CSV_PREFIXES:
        if all(os.path.isfile(os.path.join(data_dir, f'{prefix}-{t}.csv')) for t in TASKS):
            return prefix
    return None


def official_validation_available(data_dir: str) -> bool:
    return validation_dir(data_dir) is not None and validation_csv_prefix(data_dir) is not None


def official_images_without_labels(data_dir: str) -> bool:
    """Official validation images are present but their label csvs are not (yet)."""
    return validation_dir(data_dir) is not None and validation_csv_prefix(data_dir) is None


def volume_path(data_dir: str, subdir: str, view: str, cid: int) -> str:
    return os.path.join(data_dir, subdir, view, f'{cid:04d}.npy')


# ════════════════════════════════════════════════════════════════════ splits
def _stratified_split(ids: Sequence[int], labels: Dict[int, List[int]],
                      test_size: float, seed: int) -> Tuple[List[int], List[int], str]:
    """Split `ids` into (keep, held_out). Tries joint-label stratification,
    then abnormal-only, then none. Returns the strategy actually used."""
    ids = list(ids)
    strategies = [
        ('joint(abnormal,acl,meniscus)', ['{}{}{}'.format(*labels[c]) for c in ids]),
        ('abnormal', [labels[c][0] for c in ids]),
        ('none', None),
    ]
    for name, strat in strategies:
        if strat is not None:
            counts = Counter(strat)
            n_test = math.ceil(test_size * len(ids))
            # each stratum must be able to contribute to both sides
            if min(counts.values()) < 2 or n_test < len(counts) or len(ids) - n_test < len(counts):
                continue
        try:
            keep, held = train_test_split(ids, test_size=test_size, stratify=strat,
                                          random_state=seed, shuffle=True)
            return sorted(keep), sorted(held), name
        except ValueError:
            continue
    raise RuntimeError('could not split data')  # unreachable in practice


def make_splits(data_dir: str, seed: int = 42, limit: Optional[int] = None,
                log: Callable[[str], None] = print) -> dict:
    """
    Returns a dict:
      mode            'official' | 'fallback'
      entries         {'train'|'val'|'test': [(id, subdir), ...]}
      labels          {id: [abnormal, acl, meniscus]} for all exams involved
      stratification  {step: strategy}
      limit           limit applied (first N ids per split) or None
    """
    train_labels = load_labels(data_dir, 'train')
    all_ids = sorted(train_labels)
    labels = dict(train_labels)
    strat = {}

    if official_validation_available(data_dir) or official_images_without_labels(data_dir):
        vdir = validation_dir(data_dir)
        # train/val split depends only on the training labels, so it is identical
        # whether or not the validation (= test) labels are already on disk
        tr, va, strat['train_vs_val'] = _stratified_split(all_ids, labels, 0.15, seed)
        if official_validation_available(data_dir):
            mode = 'official'
            valid_labels = load_labels(data_dir, validation_csv_prefix(data_dir))
            overlap = set(valid_labels) & set(train_labels)
            if overlap:
                raise ValueError(f'train/valid exam ids overlap: {sorted(overlap)[:5]}...')
            labels.update(valid_labels)
            test_entries = [(c, vdir) for c in sorted(valid_labels)]
        else:
            mode = 'official_test_pending'
            test_entries = []
        entries = {
            'train': [(c, 'train') for c in tr],
            'val':   [(c, 'train') for c in va],
            'test':  test_entries,
        }
    else:
        mode = 'fallback'
        rest, te, strat['test_vs_rest'] = _stratified_split(all_ids, labels, 0.15, seed)
        # 15% of the total = 0.15/0.85 of the remainder
        tr, va, strat['train_vs_val'] = _stratified_split(rest, labels, 0.15 / 0.85, seed)
        entries = {
            'train': [(c, 'train') for c in tr],
            'val':   [(c, 'train') for c in va],
            'test':  [(c, 'train') for c in te],
        }

    if limit is not None and limit > 0:
        entries = {k: v[:limit] for k, v in entries.items()}

    info = {'mode': mode, 'entries': entries, 'labels': labels,
            'stratification': strat, 'limit': limit}
    info['summary'] = split_summary(data_dir, info)

    if mode == 'official_test_pending':
        log('Split mode: OFFICIAL (TEST PENDING) -- 1130 train exams split 85/15 into train/val; '
            'official validation images found but no label csvs, so the test evaluation is '
            'deferred to evaluate_test.py.')
    elif mode == 'official':
        log('Split mode: OFFICIAL -- 1130 train exams split 85/15 into train/val; '
            'official 120-exam validation set used as held-out TEST set.')
    else:
        log('Split mode: FALLBACK -- official valid/ data not found; 1130 train exams '
            'split 70/15/15 into train/val/test. Results are NOT comparable to the '
            'official MRNet benchmark.')
    log(f'Stratification: {strat}' + (f' | limit={limit} per split' if limit else ''))
    for split, s in info['summary'].items():
        pos = ', '.join(f'{t}={s["positives"][t]}' for t in TASKS)
        log(f'  {split:5s}: n={s["n"]:4d} | positives: {pos} | '
            f'sagittal->coronal substitutions: {s["sagittal_substituted"]}')
    return info


def split_summary(data_dir: str, info: dict) -> dict:
    out = {}
    for split, ents in info['entries'].items():
        missing_sag = [c for c, sd in ents if not os.path.isfile(volume_path(data_dir, sd, 'sagittal', c))]
        out[split] = {
            'n': len(ents),
            'positives': {t: int(sum(info['labels'][c][i] for c, _ in ents)) for i, t in enumerate(TASKS)},
            'sagittal_substituted': len(missing_sag),
            'sagittal_substituted_ids': missing_sag,
        }
    return out


# ═══════════════════════════════════════════════════════════════ augmentation
class VolumeAugment:
    """
    Training-time augmentation for one MRI volume (S, 1, H, W) in [0, 1].
    Parameters are sampled ONCE per volume (per plane) and applied identically
    to every slice, so the 3-D consistency across slices is preserved.

      rotation     U(-10, +10) degrees
      translation  U(-5%, +5%) of the image size, independently in x and y
      scale        U(0.95, 1.05)
      contrast     factor U(0.9, 1.1) around the volume mean
      brightness   additive offset U(-0.1, +0.1); result clamped to [0, 1]
      h-flip       p = 0.5, decided ONCE per exam (see `MRNetDataset`)

    Implemented with a single affine_grid/grid_sample call per volume
    (bilinear, zero padding), which costs a few ms on CPU.
    """

    def __init__(self, max_rotate_deg: float = 10.0, max_translate: float = 0.05,
                 scale_range: Tuple[float, float] = (0.95, 1.05),
                 contrast: float = 0.1, brightness: float = 0.1, hflip_p: float = 0.5):
        self.max_rotate_deg = max_rotate_deg
        self.max_translate = max_translate
        self.scale_range = scale_range
        self.contrast = contrast
        self.brightness = brightness
        self.hflip_p = hflip_p

    def config(self) -> dict:
        return dict(max_rotate_deg=self.max_rotate_deg, max_translate_frac=self.max_translate,
                    scale_range=list(self.scale_range), contrast_factor_range=[1 - self.contrast, 1 + self.contrast],
                    brightness_offset_range=[-self.brightness, self.brightness], hflip_p=self.hflip_p,
                    hflip_rule='one decision per exam; axial & coronal flipped along the '
                               'left-right image axis; sagittal not flipped in-plane '
                               '(its left-right axis is the slice axis, and slice order '
                               'is irrelevant under max-pooling)')

    @staticmethod
    def _uniform(lo: float, hi: float) -> float:
        return lo + (hi - lo) * torch.rand(()).item()

    def __call__(self, vol: torch.Tensor, hflip: bool) -> torch.Tensor:
        S, C, H, W = vol.shape
        angle = math.radians(self._uniform(-self.max_rotate_deg, self.max_rotate_deg))
        scale = self._uniform(*self.scale_range)
        # affine_grid works in normalised coords [-1, 1] -> image width == 2 units
        tx = 2 * self._uniform(-self.max_translate, self.max_translate)
        ty = 2 * self._uniform(-self.max_translate, self.max_translate)
        cos, sin = math.cos(angle) / scale, math.sin(angle) / scale
        theta = torch.tensor([[cos, -sin, tx], [sin, cos, ty]], dtype=vol.dtype)
        grid = F.affine_grid(theta.unsqueeze(0).expand(S, 2, 3), [S, C, H, W], align_corners=False)
        vol = F.grid_sample(vol, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

        if hflip:
            vol = torch.flip(vol, dims=[-1])

        c = self._uniform(1 - self.contrast, 1 + self.contrast)
        b = self._uniform(-self.brightness, self.brightness)
        mean = vol.mean()
        vol = ((vol - mean) * c + mean + b).clamp_(0.0, 1.0)
        return vol


# ═══════════════════════════════════════════════════════════════════ dataset
class MRNetDataset(Dataset):
    """
    Returns, per exam: (axial, coronal, sagittal, label, exam_id)
      each view: float32 tensor (S_view, 1, 256, 256) in [0, 1]
                 (per-volume min-max normalised; grey->3-channel expansion and
                 ImageNet normalisation happen on the GPU inside the model)
      label:     float32 tensor (len(tasks),) in the order of `tasks`
    """

    def __init__(self, data_dir: str, entries: Sequence[Entry], labels: Dict[int, List[int]],
                 tasks: Sequence[str] = TASKS, augment: bool = False,
                 augmenter: Optional[VolumeAugment] = None):
        self.data_dir = data_dir
        self.entries = list(entries)
        self.task_idx = [TASKS.index(t) for t in tasks]
        self.labels = labels
        self.augment = augment
        self.augmenter = augmenter or VolumeAugment()

        self.sag_substituted = set()
        for cid, sd in self.entries:
            for v in ('axial', 'coronal'):
                if not os.path.isfile(volume_path(data_dir, sd, v, cid)):
                    raise FileNotFoundError(volume_path(data_dir, sd, v, cid))
            if not os.path.isfile(volume_path(data_dir, sd, 'sagittal', cid)):
                self.sag_substituted.add(cid)

    def __len__(self):
        return len(self.entries)

    def _load(self, cid: int, subdir: str, view: str) -> torch.Tensor:
        vol = torch.from_numpy(np.load(volume_path(self.data_dir, subdir, view, cid))).float()
        vmin, vmax = vol.min(), vol.max()
        vol = (vol - vmin) / (vmax - vmin + 1e-8)       # per-volume min-max
        return vol.unsqueeze(1)                          # (S, 1, H, W)

    def __getitem__(self, idx):
        cid, sd = self.entries[idx]
        # Horizontal flip: left and right knees are mirror images of each other,
        # so a left-right mirror produces an anatomically plausible exam. One
        # decision per exam keeps the planes consistent. In axial and coronal
        # images the horizontal image axis is the medial-lateral (left-right)
        # axis, so those are flipped; in sagittal images left-right is the
        # through-plane (slice) axis, and reversing slice order does not change
        # a max-pooled representation, so sagittal is not flipped in-plane.
        hflip = self.augment and torch.rand(()).item() < self.augmenter.hflip_p

        views = []
        for view in VIEWS:
            src = 'coronal' if (view == 'sagittal' and cid in self.sag_substituted) else view
            vol = self._load(cid, sd, src)
            if self.augment:
                vol = self.augmenter(vol, hflip=hflip and src != 'sagittal')
            views.append(vol)

        label = torch.tensor([self.labels[cid][i] for i in self.task_idx], dtype=torch.float32)
        return views[0], views[1], views[2], label, cid


def collate_single(batch):
    """Batch size is always 1 exam (slice counts differ per exam/plane)."""
    assert len(batch) == 1
    return batch[0]
