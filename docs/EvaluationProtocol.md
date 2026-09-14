# Subject-Grouped Evaluation Protocol

## Objective

Estimate how models rank and classify previously unseen subjects. Videos,
sessions, clips, and frames are repeated observations, not independent samples.

## Eligibility Gate

An inferential run must abort unless:

- every included video has one pseudonymous subject and session;
- each subject has one consistent label;
- each session belongs to exactly one subject;
- no subject or session crosses train, validation, and test roles;
- each class contains at least three subjects;
- the clip cache passes shape, mask, padding, zero-video, hash, and exact
  cross-subject duplicate checks.

## Nested Subject Splitting

- Build up to five stratified outer folds from unique subjects using split seed
  `42`. Use fewer folds when the minority class has fewer than five subjects.
- Expand subject assignments to all sessions and videos.
- Select validation subjects only from the outer training subjects with a fixed
  split derived from the outer fold number.
- Keep outer test subjects inaccessible until all stages have finished and the
  globally best validation-NLL checkpoint has been restored.
- Reuse one fixed fold assignment for training seeds `42`, `43`, and `44`.

This is internal nested grouped validation. It is not an external test because
all subjects come from the same collected dataset.

## Preprocessing

- Cache five deterministic clips per video.
- Each clip contains 16 frames sampled at 15 FPS.
- Place clip starts uniformly from the beginning to the latest valid start.
- Letterbox to `112x112` with gray `128` and preserve the content mask.
- Apply crop scale `0.9-1.0`, aspect ratio `0.95-1.05`, horizontal flip `0.5`,
  and color jitter `0.1` consistently to the full clip.
- Remove per-frame rotation and random erasing.
- Use Kinetics-400 V1 mean/std for R3D.
- Compute simple-model RGB mean/std from content pixels of inner-training
  subjects only.
- Force padding to normalized value zero.

## Training

- Loss: `BCEWithLogitsLoss`, which combines sigmoid and binary cross-entropy in
  a stable operation. Heads return raw logits.
- Sampling: four draws per subject per epoch, with random session, video, clip.
- Simple models: at most 50 epochs, patience 8 on subject-level validation NLL.
- Simple3DLSTM v2: gradient clipping at 1.0.
- R3D: head 15 epochs at `1e-3`; `layer4 + fc` 20 at `1e-4`; full backbone 8
  at `1e-5`; patience 8 in each stage.
- Freeze all R3D BatchNorm statistics and affine parameters during transfer.
- Compare validation NLL globally across every stage, not only within a stage.

## Prediction Unit

Average raw logits hierarchically:

1. clips to video;
2. videos to session;
3. sessions to subject.

This gives sessions and subjects equal weight even when they contain different
numbers of recordings.

## Metrics

Primary metric: subject-level ROC AUC computed directly from raw logits.

Also report:

- PR AUC;
- balanced accuracy;
- sensitivity and specificity;
- precision and F1;
- confusion matrix;
- NLL and Brier score;
- stratified subject-bootstrap 95% confidence intervals.

Use the fixed probability threshold `0.5` (equivalently logit `0`). Do not tune
thresholds. Temperature scaling is disabled in primary runs. Any calibration
experiment must use subjects separate from training, checkpoint validation, and
outer testing, report pre/post NLL, and disclose boundary solutions.

## Negative Controls

- Capture-date-grouped shortcut models for duration, resolution, extension,
  capture time, color, padding, motion, and thumbnails.
- Perceptual embedding nearest-neighbor review across different subjects.
- Training-subject label permutations; validation and test labels never move.
- Five complete Simple2DCNN permutations and one complete grouped permutation
  for Simple3DLSTM v2 and R3D.
- Temporal tests: normal, reversed, shuffled, and repeated-middle frames.
- Grad-CAM targeted separately to positive and negative evidence.

## Reporting

Never describe legacy NPZ results as cross-validation or independent test
performance. Report the number of subjects, sessions, videos, class counts,
folds, seeds, exclusions, duplicate findings, and all uncertainty intervals.
State explicitly that this dataset cannot establish clinical validity or
generalization to new sites, devices, demographics, or collection protocols.
