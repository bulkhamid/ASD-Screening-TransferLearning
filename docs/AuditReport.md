# ASD Screening Project Audit Report

Date: 2026-06-02
Baseline: local trusted branch `Zhamshidbek-updates` at `8493dd3`

## Summary

This pass fixed code paths that could produce misleading results or fragile
runtime behavior, rebuilt the cache with aspect-ratio-preserving gray
letterbox preprocessing, reran hold-out training, and added Grad-CAM/layer
inspection tooling. No CV experiments were run.

The local branch was checked against GitHub and found to be behind
`origin/Zhamshidbek-updates`, but the newer remote code was not merged because
it was not yet trusted/reviewed.

## Fixes Implemented

- Disabled `--cached --cv` in `train.py` because the current cache contains one
  fixed `train.npz` / `val.npz` / `test.npz` split, not fold-specific cached
  splits.
- Changed `--cv` to opt-in. Cached training now defaults to hold-out behavior.
- Applied the validation-selected threshold to test accuracy, precision,
  recall, and F1. AUC remains threshold-free.
- Made Weights & Biases logging opt-in with `--wandb` and lazy-loaded to avoid
  import-time console/runtime side effects.
- Replaced deprecated `r3d_18(weights=True)` usage with the explicit
  `R3D_18_Weights.KINETICS400_V1` enum.
- Rebuilt `data/save_numpy_dataset.py` as a CLI tool with full-file hashing,
  decoded-array hashing, unreadable-video exclusion by default, all-zero
  exclusion by default, and split/skipped manifests.
- Fixed double normalization in `experiments/debug_shuffle_labels.py`.
- Added missing dependencies to `requirements.txt`: `scikit-learn`, `wandb`.
- Updated `README.md` so cached historical logs are described as repeated
  hold-out runs, not validated 5-fold CV.

## Data Integrity Check

Initial cached split check before rebuild:

| Split | Shape | Pos | Neg | Within-split duplicates | All-zero videos |
| --- | --- | ---: | ---: | ---: | ---: |
| train | `(73, 60, 112, 112, 3)` | 37 | 36 | 0 | 1 |
| val | `(11, 60, 112, 112, 3)` | 5 | 6 | 0 | 0 |
| test | `(22, 60, 112, 112, 3)` | 11 | 11 | 0 | 0 |

Cross-split decoded-array duplicate counts:

| Pair | Duplicates |
| --- | ---: |
| train-val | 0 |
| train-test | 0 |
| val-test | 0 |

Finding: the old cache had no exact decoded-array duplicates across splits, but
it contained one all-zero training video.

## Clean Cache Rebuild

The cache was rebuilt from `C:\Users\zhams\HWs\Autsim\dataset` with aspect-ratio-preserving letterbox resize:

```powershell
asd\Scripts\python.exe data\save_numpy_dataset.py --root C:\Users\zhams\HWs\Autsim\dataset --output_dir data --max_frames 60 --target_size 112 112 --seed 42 --unreadable_policy exclude --pad_value 128
```

Raw folder counts were 54 positive videos and 54 negative videos. One positive
video was excluded as unreadable:

`C:\Users\zhams\HWs\Autsim\dataset\positive\20240129050823.webm`

Rebuilt cached split check:

| Split | Shape | Pos | Neg | Within-split duplicates | All-zero videos |
| --- | --- | ---: | ---: | ---: | ---: |
| train | `(74, 60, 112, 112, 3)` | 37 | 37 | 0 | 0 |
| val | `(11, 60, 112, 112, 3)` | 5 | 6 | 0 | 0 |
| test | `(22, 60, 112, 112, 3)` | 11 | 11 | 0 | 0 |

Rebuilt cross-split decoded-array duplicate counts:

| Pair | Duplicates |
| --- | ---: |
| train-val | 0 |
| train-test | 0 |
| val-test | 0 |

The rebuilt cache now has 107 usable videos and no all-zero decoded arrays.
Because the cache changed, model checkpoints and plots produced before this
rebuild correspond to the old cache, not the rebuilt cache.

## Corrected Label-Shuffle Checks

The corrected hold-out label-shuffle sanity checks were run for 10 epochs.

| Model | Train AUC | Val AUC | Test AUC | Test loss | Test accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Simple3DLSTM | 0.740 | 0.567 | 0.612 | 0.685 | 0.591 |
| R3D | 0.835 | 0.667 | 0.521 | 0.691 | 0.545 |

Both test AUCs remain below the script's leakage warning threshold of `0.65`.
The elevated train AUC on shuffled labels shows the models can fit noise in the
training split, which is expected for small datasets and reinforces the need to
rely on held-out test metrics.

## Gray-Letterbox Hold-Out Training

After the final gray-padding cache rebuild, both main models were trained on
the corrected cached hold-out split for up to 100 epochs with early stopping.

| Model | Training setup | Temperature | Threshold | Test AUC | Test Acc | Test F1 | Test loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Simple3DLSTM | `--cached --temp --epochs 100 --patience 20` | 0.382 | 0.05 | 1.000 | 1.000 | 1.000 | 0.0026 |
| R3D | `--cached --finetune --full_finetune --full_ft_epochs 8 --full_ft_lr 1e-5 --temp --epochs 100 --patience 20` | 0.693 | 0.05 | 1.000 | 1.000 | 1.000 | 0.0013 |

The `simple2dcnn` frame-level ablation was also trained with gray padding
jitter:

| Model | Training setup | Temperature | Threshold | Test AUC | Test Acc | Test F1 | Test loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Simple2DCNN | `--cached --temp --epochs 100 --patience 20 --pad_jitter 16` | 0.439 | 0.05 | 1.000 | 1.000 | 1.000 | 0.0003 |

The Conv2D result is a scientific warning: the current hold-out split can be
solved from frame-level/static cues. It does not prove the 3D models learned
temporal ASD behavior.

## Source-Aware Hold-Out Cache

A separate source-aware cache was created in `data_source_aware/`:

```powershell
asd\Scripts\python.exe data\save_numpy_dataset.py --root C:\Users\zhams\HWs\Autsim\dataset --output_dir data_source_aware --max_frames 60 --target_size 112 112 --seed 42 --unreadable_policy exclude --pad_value 128 --split_strategy source_aware
```

The cache kept the original class balance and integrity guarantees:

| Split | Shape | Pos | Neg | Cross-split decoded duplicates | All-zero videos |
| --- | --- | ---: | ---: | ---: | ---: |
| train | `(74, 60, 112, 112, 3)` | 37 | 37 | 0 | 0 |
| val | `(11, 60, 112, 112, 3)` | 5 | 6 | 0 | 0 |
| test | `(22, 60, 112, 112, 3)` | 11 | 11 | 0 | 0 |

`data_source_aware/split_diagnostics.json` reports that source-aware splitting
cannot fully solve the raw data bias:

- `2024-08` and `2024-09` are all positive.
- `.mp4` files are all positive.
- July/October are mostly negative.

So this split is a better audit split than a single random split, but it is not
equivalent to subject/source-independent validation.

## Source-Aware 50-Epoch Training

All three models were trained on `data_source_aware/` for 50 epochs with early
stopping and validation-only threshold/temperature selection.

| Model | Training setup | Early stop | Temperature | Threshold | Test AUC | Test Acc | Test Precision | Test Recall | Test F1 | Test loss |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Simple2DCNN | `--cached --data_path data_source_aware --temp --epochs 50 --patience 10 --pad_jitter 16` | No | 0.488 | 0.55 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.0053 |
| Simple3DLSTM | `--cached --data_path data_source_aware --temp --epochs 50 --patience 10 --pad_jitter 16` | No | 0.002 | 0.05 | 0.955 | 0.955 | 1.000 | 0.909 | 0.952 | 5.4924 |
| R3D | `--cached --data_path data_source_aware --finetune --temp --epochs 50 --patience 10 --pad_jitter 16` | Yes, during fine-tune stage at epoch 17/25 | 0.546 | 0.05 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.0594 |

The near-zero Simple3DLSTM temperature is a calibration warning. Its ranking
remained strong, but its calibrated probabilities should not be interpreted as
reliable.

## Shortcut Baseline Audit

To probe why the deep models are so high, a non-neural shortcut audit was added
in `experiments/shortcut_baselines.py`. The script trains only fixed, simple
baselines on the current train split and reports validation/test metrics.

| Baseline | Val AUC | Test AUC | Test Acc | Test F1 | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Majority prior | 0.500 | 0.500 | 0.500 | 0.667 | Expected chance-style baseline |
| Global color/brightness/padding stats | 0.800 | 0.967 | 0.955 | 0.952 | Low-level visual statistics are highly predictive |
| Frame-difference motion stats | 0.967 | 1.000 | 0.955 | 0.952 | Coarse motion/recording differences are highly predictive |
| Tiny static thumbnails | 0.967 | 1.000 | 0.955 | 0.952 | Static frame/background/framing cues are highly predictive |
| Path suffix/date tokens | 0.500 | 0.545 | 0.545 | 0.167 | Near chance after excluding label-folder names |

This result is a major blind spot. The current split can be classified almost
perfectly without a video deep-learning model. The next validity step should be
subject/source/session-aware splitting or an external test set.

On the source-aware cache, shortcut baselines remained strong:

| Baseline | Val AUC | Test AUC | Test Acc | Test F1 |
| --- | ---: | ---: | ---: | ---: |
| Majority prior | 0.500 | 0.500 | 0.500 | 0.667 |
| Global color/brightness/padding stats | 0.867 | 0.901 | 0.864 | 0.870 |
| Frame-difference motion stats | 1.000 | 1.000 | 0.955 | 0.952 |
| Tiny static thumbnails | 1.000 | 0.967 | 0.955 | 0.957 |
| Path suffix/date tokens | 0.500 | 0.591 | 0.591 | 0.308 |

This means the source-aware split did not remove shortcut predictability.

## Feature Distribution Audit

`experiments/feature_distribution_audit.py` was added to identify which simple
signals differ by class.

Top class differences:

- Positive videos have much larger frame-difference values. On the test split,
  `diff_p99` has Cohen's d `3.449` and `diff_std` has Cohen's d `2.213`.
- Negative videos are brighter and have higher green/blue channel means. On the
  test split, `mean_g` has Cohen's d `-1.644`, `mean_b` has Cohen's d `-1.548`,
  and `brightness_mean` has Cohen's d `-1.289`.
- Capture month is confounded with label. In the current manifest,
  August/September videos are almost entirely positive, while July/October
  videos are mostly negative.
- All `.mp4` files in the current manifest are positive.

These findings are consistent with visual/source shortcuts, not necessarily
behavioral ASD features.

## Best-Practice Tuning Strategy

- Keep an untouched final test set. Do not use it for model choice,
  threshold choice, temperature scaling, epoch count, or architecture choice.
- Use validation loss for early stopping and validation metrics for threshold
  selection. Report AUC as the primary threshold-free metric.
- For hyperparameter tuning, use a small predefined grid and choose only on the
  validation set, or use nested/grouped CV when the dataset is large enough.
- Prefer source/subject/session grouped splits over random file-level splits.
- Run shortcut baselines before trusting deep models. If a simple baseline is
  strong, treat high neural scores as suspect until the split is strengthened.
- Report calibration cautiously on this dataset because the validation set is
  small.
- See `docs/EvaluationProtocol.md` for the full project tuning and reporting
  standard.

## Remaining Limitations

- Current verified claims should be limited to the fixed hold-out split.
- True GroupKFold CV still requires raw subject metadata or fold-specific cached
  splits and should be implemented/rerun before claiming group CV results.
- Subject-level independence cannot be proven from the current `.npz` files
  alone because the cache does not include subject IDs.
- Aspect-ratio-preserving resize is now implemented in both raw-video loading
  and cache generation. Gray padding is now preferred over black padding
  because black borders appeared as salient Grad-CAM regions in some
  Simple3DLSTM examples.
- R3D now supports a third transfer-learning stage with
  `--finetune --full_finetune`, which unfreezes the full backbone at a lower
  learning rate.
- Grad-CAM dashboard generation is available through
  `python -m tools.gradcam_dashboard`; use it after retraining on the current
  preprocessing to interpret final checkpoints.
- A `simple2dcnn` frame-level ablation was added. It also reached perfect
  hold-out performance on the current split, which means static frame,
  recording, background, or subject/source cues may be sufficient for this
  dataset. Treat this as a warning against claiming the models necessarily
  learned temporal ASD behavior without subject-level or external validation.

## Verification Commands

All verification used `asd\Scripts\python.exe`.

- AST parse check for all project Python files: passed.
- `python train.py --help`: passed.
- `python data\save_numpy_dataset.py --help`: passed.
- `python -m experiments.debug_shuffle_labels --help`: passed.
- `python train.py --model r3d --cached --cv`: failed fast as intended.
- `python -m experiments.shortcut_baselines`: passed and wrote
  `experiments/shortcut_baselines_results.json`.
- `python -m experiments.feature_distribution_audit`: passed and wrote
  `experiments/feature_distribution_audit_results.json`.
- `python -m experiments.shortcut_baselines --data_dir data_source_aware`: passed
  and wrote `experiments/shortcut_baselines_source_aware_results.json`.
- `python -m experiments.feature_distribution_audit --data_dir data_source_aware`:
  passed and wrote `experiments/feature_distribution_source_aware_results.json`.
- `python train.py --model simple2dcnn --cached --data_path data_source_aware ...`:
  passed for 50 epochs.
- `python train.py --model simple3dlstm --cached --data_path data_source_aware ...`:
  passed for 50 epochs.
- `python train.py --model r3d --cached --data_path data_source_aware --finetune ...`:
  passed with early stopping in the fine-tuning stage.
- Cached split duplicate/all-zero check after rebuild: passed with no all-zero
  videos and no exact decoded duplicates across train/val/test.
- `python -m unittest discover -s tests -v`: passed, 23 tests.
