# MVMT-MRNet — multi-view, multi-task knee MRI classification

Training and evaluation code for a single shared-backbone model that classifies three knee
pathologies (general abnormality, ACL tear, meniscus tear) from all three MRI planes in one
forward pass, together with the single-task baselines it is compared against.

One ImageNet-pretrained ResNet50 encodes the slices of every plane; per-plane linear
projections are max-pooled over slices, concatenated, and passed through a shared fully
connected layer to one classification head per task (25.3 M parameters, 23.8 M trainable,
against 75.8 M for three separate single-task models).

## Results

Held-out test set = the official 120-examination MRNet validation set (ids 1130–1249;
95 abnormality, 54 ACL, 52 meniscus positives). Each model is evaluated on it exactly once,
after the architecture, hyperparameters, checkpoint and decision thresholds have been fixed on
a validation subset. Confidence intervals: percentile bootstrap over examinations, 2,000
resamples, seed 42. The difference column is paired (same resample for both models).

| Task | Multi-task (one model) | Single-task (one model per task) | Difference |
|---|---|---|---|
| Abnormality | 0.844 [0.752, 0.922] | 0.825 [0.724, 0.914] | +0.019 [−0.108, +0.150] |
| ACL tear | 0.914 [0.857, 0.959] | 0.944 [0.896, 0.978] | −0.030 [−0.095, +0.032] |
| Meniscus tear | 0.664 [0.567, 0.757] | 0.790 [0.703, 0.868] | −0.126 [−0.231, −0.020] |
| **Mean** | **0.807 [0.749, 0.860]** | 0.853 [0.802, 0.897] | −0.046 [−0.115, +0.015] |

On the summary metric the two designs are statistically indistinguishable, at one third of the
parameters and one forward pass instead of three; the shared model is significantly worse on
meniscus tear. Parameter sharing redistributes accuracy across the tasks rather than adding it.

### Effect of the effective batch size and the learning rate

The variable slice count forces one examination per forward pass, so gradients are averaged over
several examinations per optimiser step instead (`--accum_steps`). Padding volumes to a common
slice count was rejected: padded slices would enter the max-pooling over slices. All four runs
below are the multi-task model with everything else identical; the configuration was chosen on
the validation column alone.

| Examinations per update | Learning rate | Validation mean AUC | Test mean AUC | Run directory |
|---|---|---|---|---|
| 1 | 1e-5 | 0.768 | 0.778 [0.713, 0.836] | `runs/multitask` |
| 8 | 1e-5 | 0.751 | 0.685 [0.618, 0.745] | `runs/multitask_accum8` |
| **8** | **8e-5** | **0.838** | **0.807 [0.749, 0.860]** | `runs/multitask_accum8_lr8e-5` |
| 1 | 8e-5 | 0.628 | 0.584 [0.509, 0.659] | `runs/multitask_b1_lr8e-5` |

A larger effective batch helps only when the learning rate is scaled with it, and that same rate
at one examination per update diverges (ACL tear falls to 0.435, below chance). At one
examination per update, a rate of 1e-4 with the class-weighted loss collapses to a constant
predictor (`runs/_collapsed_multitask_lr1e-4`, kept for reference).

## Protocol

- **Splits** (seed 42, stratified on the joint label combination): 960 training / 170 validation
  examinations from the 1,130 public training examinations; the 120 public validation
  examinations are the held-out test set. The validation subset is used only for early stopping,
  checkpoint selection and the Youden thresholds.
- **Preprocessing:** per-volume min–max scaling to [0, 1], grayscale replicated to 3 channels,
  ImageNet mean/std normalisation, native 256 × 256.
- **Augmentation** (training only, sampled once per volume and applied to every slice): rotation
  ±10°, translation ±5%, scale 0.95–1.05, contrast ×0.9–1.1, brightness ±0.1, horizontal flip
  p = 0.5 on axial and coronal only (in sagittal images the left–right axis is the slice axis).
- **Loss:** mean over tasks of BCE-with-logits weighted by `pos_weight = neg/pos` on the training
  split (0.237 abnormality, 4.424 ACL, 1.849 meniscus).
- **Optimisation:** Adam, lr 8e-5, weight decay 1e-5, cosine annealing to 8e-7 over at most 100
  epochs, mixed precision, gradients averaged over 8 examinations per step.
- **Early stopping:** best validation mean AUC, patience 20, not active before epoch 30.
- **Frozen layers:** conv1, bn1, layer1, layer2, with their BatchNorm kept in eval mode.

Mixed precision was measured at 2,597 MB peak vs 4,609 MB without it (−43.7%) and 1.76× faster
per examination.

## Data

The MRNet dataset is distributed by Stanford under a research-use agreement and is **not**
included here (neither the volumes nor the label CSVs). Request access at
<https://stanfordmlgroup.github.io/competitions/mrnet/> and place the download so that the
project directory contains `train/{axial,coronal,sagittal}/*.npy`, `train-{abnormal,acl,meniscus}.csv`,
and the official validation set as `valid/` (or `val/`) plus `valid-*.csv` (or `val-*.csv`).
Redivis exports the label tables with the first data row used as the header and with underscore
names — convert them to headerless `id,label` rows before use.

Without the official validation data the code falls back to a 70/15/15 split of the training
examinations and says so in the log; results from that mode are not comparable to published
MRNet numbers.

## Usage

```bash
pip install -r requirements.txt

# full experiment queue (benchmarks, multi-task, single-task baselines)
bash run_experiments.sh          # first round: one examination per optimiser step
bash run_experiments2.sh         # final protocol: effective batch 8, lr 8e-5

# a single run
python train.py --out_dir runs/my_run --tasks abnormal,acl,meniscus \
    --max_epochs 100 --patience 20 --min_epochs 30 --lr 8e-5 --min_lr 8e-7 --accum_steps 8

# test evaluation of an existing run (e.g. labels arrived after training)
python evaluate_test.py runs/my_run

# paired multi-task vs single-task comparison
python compare_runs.py --multitask multitask_accum8_lr8e-5 --single_suffix _b8

# figures
python generate_figures.py
```

Each run writes `results.json` (splits, positives, parameter counts, loss definition, best epoch,
test AUCs with CIs, operating points, timing, memory, full argument list), `metrics.csv` (per
epoch), `training.log`, `test_predictions.csv`, `val_predictions.csv` and its curves.

## Repository layout

| Path | Contents |
|---|---|
| `model.py` | shared-backbone multi-view model, any subset of the three tasks |
| `dataset.py` | split logic, volume loading, augmentation |
| `train.py` | training loop, early stopping, bootstrap CIs, single test evaluation |
| `evaluate_test.py` | deferred test evaluation from a saved checkpoint |
| `compare_runs.py` | paired bootstrap comparison, writes `runs/comparison*.json` |
| `generate_figures.py` | all figures in `figures_v2/` |
| `run_experiments.sh`, `run_experiments2.sh` | the experiment queues |
| `runs/` | per-run metrics, logs, predictions and `results.json`, including superseded configurations (model checkpoints are not included) |
| `figures_v2/` | figures used in the manuscript |

Hardware used: a single NVIDIA GeForce RTX 5060 Ti (8 GB); about 70 minutes per run at 139 s per
epoch.

## Funding

Supported by Grant BR24992820 from the Science Committee of the Ministry of Science and Higher
Education of the Republic of Kazakhstan.

## License

Apache License 2.0 — see `LICENSE`.
