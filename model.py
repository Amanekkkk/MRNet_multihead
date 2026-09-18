"""
Multi-view (axial / coronal / sagittal) knee-MRI classifier with a single
ImageNet-pretrained ResNet50 backbone SHARED by all three planes.

Pipeline for one exam (batch size 1):

    for each plane p in (axial, coronal, sagittal):
        slices (S_p, 1, 256, 256)
          -> grey->3ch, ImageNet mean/std normalisation
          -> shared ResNet50 (fc removed, global avg-pool)   (S_p, 2048)
          -> plane-specific projection Linear(2048,256)+ReLU (S_p, 256)
          -> max-pool over slices                            (256,)
    concat three plane vectors                               (768,)
      -> shared FC Linear(768,256) + ReLU + Dropout(0.5)     (256,)
      -> one Linear(256,1) head per requested task           (T,) logits

Design note (per-plane projection vs. shared projection + view embedding):
the backbone weights are shared, but the three planes are acquired with
different sequences (sagittal/coronal/axial differ in contrast and in which
structures are visible), so a light plane-specific projection lets the model
re-weight the shared 2048-d features per plane. It costs only ~0.5M parameters
per plane (1.6M, ~6% of the model, in total) and keeps the concatenation order-aware, so a
separate view embedding is not needed.

The same class is used for the single-task baselines by passing e.g.
tasks=['acl']; everything except the number of heads is identical.
"""
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torchvision.models as models

VIEWS = ('axial', 'coronal', 'sagittal')
ALL_TASKS = ('abnormal', 'acl', 'meniscus')

# ImageNet statistics expected by the pretrained torchvision weights.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class MultiViewMRNet(nn.Module):
    """
    One shared ResNet50 backbone for all planes, plane-specific 2048->256
    projections, slice-wise max-pooling, late fusion by concatenation, a shared
    FC layer and one binary head per task.

    Frozen part: conv1, bn1, layer1, layer2 of the backbone (parameters do not
    receive gradients AND their BatchNorm layers are kept in eval mode, so the
    ImageNet running statistics are used and never updated; see `train()`).
    layer3, layer4, projections, shared FC and heads are trained; the BatchNorm
    layers inside layer3/layer4 use batch statistics over the slices of the
    plane being processed.
    """

    FROZEN_STAGES = ('conv1', 'bn1', 'layer1', 'layer2')

    def __init__(self, tasks: Sequence[str] = ALL_TASKS, feature_dim: int = 256,
                 pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        tasks = list(tasks)
        if not tasks or any(t not in ALL_TASKS for t in tasks) or len(set(tasks)) != len(tasks):
            raise ValueError(f'tasks must be a non-empty subset of {ALL_TASKS}, got {tasks}')
        self.tasks: List[str] = tasks

        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)
        resnet.fc = nn.Identity()          # backbone output: (S, 2048) after avgpool+flatten
        self.backbone = resnet

        for name in self.FROZEN_STAGES:
            for p in getattr(self.backbone, name).parameters():
                p.requires_grad = False

        self.register_buffer('img_mean', torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('img_std', torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        # plane-specific light projections (see module docstring)
        self.proj = nn.ModuleDict({
            v: nn.Sequential(nn.Linear(2048, feature_dim), nn.ReLU(inplace=True))
            for v in VIEWS
        })

        self.shared_fc = nn.Sequential(
            nn.Linear(feature_dim * len(VIEWS), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict({t: nn.Linear(256, 1) for t in tasks})

    # ------------------------------------------------------------------ utils
    def train(self, mode: bool = True):
        """Standard train/eval switch, but frozen stages always stay in eval
        mode so their BatchNorm running statistics are neither used as batch
        statistics nor updated."""
        super().train(mode)
        for name in self.FROZEN_STAGES:
            getattr(self.backbone, name).eval()
        return self

    def count_parameters(self) -> Dict[str, int]:
        return count_parameters(self)

    # ---------------------------------------------------------------- forward
    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        """(S,1,H,W) or (S,3,H,W) in [0,1] -> ImageNet-normalised (S,3,H,W)."""
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        return (x - self.img_mean) / self.img_std

    def encode_view(self, x: torch.Tensor, view: str) -> torch.Tensor:
        """x: (S, C, H, W) slices of one plane -> (feature_dim,)"""
        feats = self.backbone(self._prep(x))       # (S, 2048)
        feats = self.proj[view](feats)             # (S, feature_dim)
        return feats.max(dim=0).values             # max over slices

    def forward(self, axial: torch.Tensor, coronal: torch.Tensor,
                sagittal: torch.Tensor) -> torch.Tensor:
        """Single exam. Each input (S_view, C, H, W). Returns (T,) logits in
        the order of `self.tasks`."""
        fused = torch.cat([
            self.encode_view(axial, 'axial'),
            self.encode_view(coronal, 'coronal'),
            self.encode_view(sagittal, 'sagittal'),
        ], dim=0)                                  # (768,)
        shared = self.shared_fc(fused)             # (256,)
        return torch.cat([self.heads[t](shared) for t in self.tasks], dim=0)  # (T,)


# Backwards-compatible name used by the original scripts.
MultiTaskMRNet = MultiViewMRNet


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable, 'frozen': total - trainable}


if __name__ == '__main__':
    for tasks in (list(ALL_TASKS), ['acl']):
        m = MultiViewMRNet(tasks=tasks, pretrained=False)
        print(tasks, m.count_parameters())
