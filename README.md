# ASD Screening With Video Models

Latest review: [September 11 follow-up audit and rerun](docs/FollowupAudit_2026-09-11.md).

Current data preparation: [owner-confirmed participant mapping and reviewed benchmark](docs/ReviewedSubjectBenchmark.md).

Research code for evaluating video models for Autism Spectrum Disorder (ASD)
screening. This repository is an internal research prototype, not a clinical
device.

## Current Scientific Status

The previously reported near-perfect results are **not trustworthy estimates of
subject-independent generalization**. The audit found:

- 11/11 legacy validation videos and 21/22 legacy test videos share a capture
  date with training videos.
- All 29 observed capture-date groups are single-class groups.
- A motion-statistics-only classifier still reaches capture-date-grouped OOF
  ROC AUC `0.957` and balanced accuracy `0.887`.
- The legacy raw `--cv` option used filename stems, not verified subject IDs.
- The old cache sampled 60 frames across long videos and treated those sparse
  frames as a contiguous sequence.
- Legacy R3D input normalization, frame-wise augmentation, BatchNorm handling,
  label permutation, and calibration contained validity problems.

The old `data/` and `data_source_aware/` NPZ files are retained only as legacy
audit artifacts. Their test sets have been inspected repeatedly and are now
development data. The owner has since confirmed one main participant per video;
the private mapping now passes validation. A frame-content audit found one pair
of repeated recordings despite different file hashes. After decoding exclusions
and removing one copy from the experiment manifest, 102 videos remain for the
corrected benchmark. Shared capture dates and recording conditions remain risks.

## Corrected Protocol

The subject-independent path is separate from legacy `train.py`:

1. Map every included video to one pseudonymous `subject_id` and `session_id`.
2. Validate that subjects have one label, sessions belong to one subject, and at
   least three subjects exist in each class.
3. Build a split-independent, per-video cache with five deterministic clips.
4. Create stratified outer folds from unique subjects, then select validation
   subjects only inside each outer training fold.
5. Train with seeds `42`, `43`, and `44`; aggregate clip predictions to video,
   session, and subject.
6. Select checkpoints by subject-level validation NLL. Evaluate outer test
   subjects only after selection.
7. Report subject ROC AUC from raw logits as the primary metric. Use fixed
   probability threshold `0.5`; primary experiments do not tune thresholds or
   apply temperature scaling.

Without new external subjects, results remain internal subject-grouped
validation and cannot establish external or clinical generalization.

## Professor Feedback Implemented

- **Aspect ratio:** frames are letterboxed to `112x112`; the long side is scaled
  to 112 and the short side is centered in neutral gray padding (`128`).
- **Loss:** model heads output one raw logit. `nn.BCEWithLogitsLoss` combines the
  sigmoid operation and binary cross-entropy in one numerically stable loss.
- **R3D stage 3:** the grouped benchmark uses 15 head-only epochs at `1e-3`, 20
  `layer4 + fc` epochs at `1e-4`, and 8 full-backbone epochs at `1e-5`.
- **Evaluation:** unverifiable cached/raw CV is disabled. Subject folds are the
  unit of evaluation; the outer test fold is inaccessible to early stopping.
- **2D ablation:** `Simple2DCNN` provides a frame-level Conv2D baseline with
  temporal average pooling.

## Setup

Use the existing project environment:

```powershell
.\asd\Scripts\python.exe -m pip check
.\asd\Scripts\python.exe -m unittest discover -s tests -v
```

Dependencies are listed in `requirements.txt`. WandB is optional at runtime and
defaults to disabled.

## Private Metadata

Generate the private template:

```powershell
.\asd\Scripts\python.exe -m data.prepare_dataset template `
  --root "<PRIVATE_DATASET_ROOT>" `
  --output private\subject_sessions.csv
```

The generated file contains SHA-256 `video_id` values and relative paths. Fill
in pseudonymous `subject_id` and `session_id`; do not use names, medical record
numbers, or other direct identifiers. `private/` is ignored by Git.

Validate the completed mapping:

```powershell
.\asd\Scripts\python.exe -m data.prepare_dataset validate `
  --root "<PRIVATE_DATASET_ROOT>" `
  --metadata private\subject_sessions.csv
```

The current local mapping has 108 rows: 107 owner-confirmed participant IDs and
one excluded unreadable file. IDs and participant-specific session IDs were
assigned from the owner's confirmation, with a private backup and provenance
record. Four more videos failed strict clip decoding; one repeated recording is
excluded in `private/reviewed_cache_manifest.csv`. Use that reviewed manifest for
the current benchmark and its matching baselines.

## Clip Cache

After metadata validation passes:

```powershell
.\asd\Scripts\python.exe -m data.prepare_dataset cache `
  --root "<PRIVATE_DATASET_ROOT>" `
  --metadata private\subject_sessions.csv `
  --output_dir data_grouped
```

Each video receives five clips of 16 frames sampled at 15 FPS and positioned
uniformly over its duration. The cache records frame indices, dimensions, FPS,
duration, SHA-256 hashes, content boxes, read failures, and packed content masks.
Padding is set to zero after normalization.

## Subject Benchmark

Validate the cache and fold assignment without training:

```powershell
.\asd\Scripts\python.exe -m experiments.run_subject_benchmark --validate_only
```

Run all primary models only after the mapping and cache pass validation:

```powershell
.\asd\Scripts\python.exe -m experiments.run_subject_benchmark `
  --models simple2dcnn simple3dlstm_v2 r3d
```

Configuration lives in `configs/subject_cv.json`. Each run is stored under a
configuration hash in ignored `runs/` storage with environment, Git state,
private fold assignments, histories, predictions, metrics, checkpoints, and
checkpoint hashes.

### Models

- `simple2dcnn`: Conv2D per frame plus temporal mean pooling, up to 50 epochs.
- `simple3dlstm_v2`: compact GroupNorm Conv3D network, adaptive spatial pooling,
  128-to-64 LSTM, dropout `0.3`, and gradient clipping at `1.0`.
- `r3d`: torchvision R3D-18 with Kinetics-400 normalization, frozen BatchNorm,
  and three-stage transfer learning.
- `simple3dlstm_legacy`: checkpoint-compatible old 6.7M-parameter model; excluded
  from the primary benchmark.

Training draws four samples per subject per epoch. Each draw chooses a session,
video, and cached clip; this prevents subjects with more recordings from
dominating the objective. A single spatial/color augmentation is shared across
all frames in a clip.

## Validity Experiments

The completed aggregate legacy shortcut audit is committed as
`experiments/grouped_shortcut_date_results.json`. Its path-bearing source
manifest was removed afterward. To reproduce it, provide an equivalent private
manifest explicitly:

```powershell
.\asd\Scripts\python.exe -m experiments.grouped_shortcut_baselines `
  --manifest "<PRIVATE_LEGACY_MANIFEST>"
```

After the grouped cache exists:

```powershell
.\asd\Scripts\python.exe -m experiments.subject_shortcut_baselines
.\asd\Scripts\python.exe -m experiments.perceptual_duplicate_audit
.\asd\Scripts\python.exe -m experiments.run_permutation_suite
```

The benchmark also evaluates normal, reversed, shuffled, and repeated-middle
frame order. If temporal models retain performance under shuffled or static
frames, their score is probably driven by appearance or recording shortcuts
rather than temporal behavior.

## Grad-CAM

Grouped Grad-CAM uses the exact cached clip and checkpoint normalization:

```powershell
.\asd\Scripts\python.exe -m tools.gradcam_dashboard `
  --model r3d `
  --checkpoint "<RUN_DIR>\best_checkpoint.pth" `
  --fold_assignments "<RUN_ROOT>\fold_assignments.csv" `
  --fold 1 --role test --target predicted
```

The dashboard shows original and overlay frames and identifies the model, layer,
video hash, subject, session, clip, frame, and target class. Grad-CAM visualizes
local sensitivity; it does not prove that highlighted regions are causal or
clinically meaningful.

## Legacy Guard

`train.py --cv` always fails because verified subjects are unavailable there.
Legacy holdout runs require `--legacy_holdout_ack`. Threshold tuning requires
`--tune_threshold`; experimental temperature scaling additionally requires
`--experimental_calibration`. These options do not make legacy results valid.

See `docs/AuditReport.md`, `docs/EvaluationProtocol.md`, and
`docs/PrivacyAndMetadata.md` for the complete audit and reporting rules.
