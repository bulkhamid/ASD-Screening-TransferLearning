# ASD Video Project Audit Report

Historical audit snapshot. The subsequent owner-confirmed mapping and footage
review supersede the missing-metadata status below; see
[Reviewed Subject Benchmark](ReviewedSubjectBenchmark.md).

## Executive Finding

The repository is technically runnable, but its previously reported
near-perfect performance is not a trustworthy estimate of generalization to new
subjects. The strongest current conclusion is that recording and collection
shortcuts are highly predictive in the available videos.

No new deep benchmark was run in this audit. That is intentional: the private
subject/session template is incomplete, and training before identity validation
would repeat the central scientific error.

## Evidence Against the Legacy Claim

### Split leakage risk

- The rebuilt legacy cache contains 107 readable videos from 108 raw files.
- All 11 validation videos and 21 of 22 test videos share a capture date with at
  least one training video.
- Capture date is not proven to be a subject, but overlap makes session,
  background, device, and recording-condition leakage plausible.
- The old raw `--cv` code used unique filename stems as groups. Because each
  filename was effectively unique, this did not create subject-independent CV.
- The old cached `--cv` path loaded the same fixed train/validation/test NPZ
  files repeatedly. Those logs were repeated holdout runs, not five-fold CV.

### Source confounding

Every one of the 29 observed capture-date groups contains only one class. Month
and extension are also class-confounded. A source-aware random holdout can make
source distributions look more similar, but it cannot remove this structural
confounding.

### Shortcut baselines

The new audit held complete capture dates out with grouped OOF prediction across
all 107 legacy videos:

| Feature family | ROC AUC | PR AUC | Balanced accuracy | F1 |
|---|---:|---:|---:|---:|
| Duration, resolution, extension, capture time | 0.705 | 0.704 | 0.683 | 0.696 |
| Color, brightness, padding | 0.691 | 0.577 | 0.692 | 0.713 |
| Frame-difference motion | **0.957** | **0.962** | **0.887** | **0.880** |
| Tiny thumbnails | 0.733 | 0.635 | 0.767 | 0.786 |

Some grouped folds contain one test class because capture-date groups are pure
and unevenly sized. The table therefore uses aggregate OOF metrics, not a mean
of per-fold AUCs. It is a shortcut diagnostic, not a valid subject result.

The motion result is especially important: a four-feature logistic classifier
nearly reproduces the deep-model ranking without learning social behavior. A
deep score of 1.0 on the old holdout is therefore fully compatible with shortcut
learning.

### Temporal construction problem

Legacy preprocessing randomly selected 60 frames across videos lasting roughly
three minutes, sorted the selected indices, and passed them to Conv3D/R3D as
adjacent frames. The resulting sequence does not represent continuous motion.
Frame-wise random crops, colors, rotations, flips, and erasing also introduced
synthetic motion because each frame received a different transform.

### R3D transfer mismatch

Legacy R3D used ImageNet normalization. The torchvision Kinetics-400 R3D-18
weights expect mean `(0.43216, 0.394666, 0.37645)` and standard deviation
`(0.22803, 0.22145, 0.216989)`. In addition, frozen R3D BatchNorm buffers were
allowed to update during head-only training.

### Calibration failure

The legacy Simple3DLSTM produced temperature approximately `0.00184`. On its old
test set, raw NLL was approximately `0.378`; calibrated NLL worsened to
approximately `5.51`. Probability saturation also changed numerically computed
AUC from `1.00` to approximately `0.955`, even though positive scaling should
not change ranking. This exposed two problems:

1. AUC was computed from saturated probabilities instead of raw logits.
2. Calibration used too little, repeatedly examined validation data and reached
   a pathological boundary-like solution.

Primary experiments now disable calibration and use logit AUC. The repaired
experimental scaler is bounded, optimizes `BCEWithLogitsLoss`, records pre/post
NLL, and refuses to retain a solution that worsens calibration NLL. It still
requires independent calibration subjects.

### Invalid old permutation control

The old script independently shuffled training, validation, and test labels.
That destroys the ground truth used to evaluate the null and makes a near-chance
test result uninformative. It also hardcoded the default cache.

The legacy diagnostic now permutes training labels only and leaves validation
and test labels fixed. The subject benchmark permutes labels at the training
subject level only. Planned controls are five complete Simple2DCNN permutations
and one complete grouped permutation for each larger model.

### Grad-CAM warning

Legacy Simple3DLSTM visualizations showed strong responses at black letterbox
borders. Gray padding reduced this particular response, but a changed heatmap is
not evidence that the model learned ASD-related behavior. The corrected pipeline
stores content masks, sets padding to zero after normalization, masks overlays,
and traces each Grad-CAM to an exact cached clip, frame, checkpoint, model,
layer, normalization, and target class.

## Implemented Corrections

### Privacy and metadata

- Added `data.prepare_dataset template|validate|cache`.
- Generated a private 108-row SHA-256 template locally: 107 readable, one
  excluded unreadable file.
- Revalidated all 108 full-file hashes and metadata coverage; no duplicate raw
  SHA-256 identifiers were found.
- Added consistency checks for subject labels, session ownership, coverage,
  duplicates, relative paths, hashes, and minimum subjects per class.
- Removed tracked manifests containing absolute paths and timestamps.
- Ignored private metadata, grouped caches, predictions, fold assignments,
  checkpoints, and runs.
- Kept an empty committed schema in `docs/subject_metadata_schema.csv`.

### Clip preprocessing

- Five deterministic clips per video.
- Sixteen frames per clip at 15 FPS.
- Clip locations distributed uniformly across video duration.
- Aspect-ratio-preserving `112x112` letterbox with gray value `128`.
- Packed content masks and normalized-zero padding.
- Shared crop, flip, and color jitter for every frame in one clip.
- Cache metadata includes frame indices, hashes, FPS, duration, original
  dimensions, content boxes, failures, and cache version.

### Models and training

- Preserved the old model as `simple3dlstm_legacy`.
- Added compact `simple3dlstm_v2` with GroupNorm, adaptive spatial pooling,
  128-to-64 LSTM, dropout `0.3`, and gradient clipping `1.0`.
- Kept Simple2DCNN as a temporal-order-insensitive ablation.
- Added explicit R3D Kinetics normalization.
- Froze R3D BatchNorm statistics and affine parameters.
- Added exact R3D stage budgets: 15 head, 20 layer4, 8 full backbone.
- Selects the globally best validation-NLL checkpoint across all R3D stages.
- Uses `BCEWithLogitsLoss`; sigmoid is applied only to convert logits to
  probabilities for probability metrics.
- WandB defaults to disabled and is optional/offline-safe.

### Subject evaluation

- Outer stratified folds operate on unique subjects and abort below three
  subjects per class.
- Inner validation subjects come only from the outer training fold.
- Training samples exactly four examples per subject per epoch.
- Predictions aggregate clips to video, video to session, and session to
  subject.
- Primary ROC AUC is computed directly from subject logits.
- The decision threshold is fixed at `0.5`; no primary threshold tuning.
- Reports PR AUC, balanced accuracy, sensitivity, specificity, precision, F1,
  NLL, Brier, confusion matrices, and stratified subject-bootstrap intervals.
- Temporal evaluation includes normal, reversed, shuffled, and static frames.

### Reproducibility

Each grouped run records a configuration SHA-256, configuration, environment,
Git state, private fold assignment, normalization, history, predictions,
metrics, checkpoints, and checkpoint SHA-256. Seeds are `42`, `43`, and `44`
with one fixed subject-fold assignment.

## Professor Feedback Status

| Suggestion | Status |
|---|---|
| Preserve aspect ratio and pad to 112 | Implemented with gray letterboxing and content masks |
| Explain BCEWithLogitsLoss includes sigmoid + BCE | Implemented in code documentation and manuscript guidance |
| Add full-backbone R3D stage | Implemented as 8 epochs at `1e-5` |
| Correct questionable CV | Legacy CV disabled; true subject folds implemented, awaiting metadata |
| Compare Conv2D | Simple2DCNN retained as an explicit ablation |

The PDF manuscript itself has not been regenerated in this pass. Any revision
must replace the old CV and near-perfect-result claims with the findings in this
report.

## Current Execution Gate

The private template intentionally has blank `subject_id` and `session_id`
fields. Strict validation therefore fails, as it should. No grouped clip cache,
perceptual cross-subject audit, Grad-CAM from grouped checkpoints, permutation
suite, or deep subject benchmark may run until the mapping is manually completed
and validated.

## Verification Completed

- The earlier suite ran `63` tests: `62` passed and one legacy-manifest test was intentionally
  skipped because that sensitive file was removed.
- All `61` project Python files passed AST parsing and import checks.
- One-batch backward/optimizer smoke tests passed for Simple2DCNN,
  Simple3DLSTM v2, and R3D.
- A synthetic end-to-end subject-grouped run produced normalization, history,
  checkpoint, checkpoint hash, predictions, and metrics artifacts.
- All CLI smoke tests passed.
- `pip check` reported no broken requirements.
- `git diff --check` reported no whitespace errors.
- The private template passed coverage and full SHA-256 validation in
  incomplete mode, then correctly failed the strict identity gate.

Once the gate passes, run in this order:

1. Build `data_grouped/`.
2. Run cache integrity and perceptual near-duplicate review.
3. Run `run_subject_benchmark` for all folds, models, and three seeds.
4. Run grouped permutation controls.
5. Review temporal ablations and target-specific Grad-CAM.
6. Report internal subject-grouped estimates with confidence intervals.

## Claim Boundary

The current evidence supports this statement:

> Legacy holdout performance is confounded by collection artifacts, with simple
> motion features achieving capture-date-grouped OOF ROC AUC 0.957. A corrected
> subject-grouped pipeline is implemented, but valid deep estimates await manual
> subject/session mapping.

It does **not** support claims of perfect ASD classification, validated five-fold
CV, external generalization, clinical screening accuracy, or learned behavioral
biomarkers.
