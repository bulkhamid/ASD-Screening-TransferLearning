# Evaluation and Tuning Protocol

This protocol is the standard to use before reporting new ASD screening model
results.

## Current Risk

The current hold-out split is not strong enough to support a generalization
claim. Simple non-neural baselines can almost perfectly classify the test split:

| Baseline | Test AUC | Test accuracy | Test F1 |
| --- | ---: | ---: | ---: |
| Global color/brightness/padding stats | 0.967 | 0.955 | 0.952 |
| Frame-difference motion stats | 1.000 | 0.955 | 0.952 |
| Tiny static thumbnails | 1.000 | 0.955 | 0.952 |

A source-aware cache reduced some split randomness but did not remove shortcut
predictability:

| Baseline on source-aware cache | Test AUC | Test accuracy | Test F1 |
| --- | ---: | ---: | ---: |
| Global color/brightness/padding stats | 0.901 | 0.864 | 0.870 |
| Frame-difference motion stats | 1.000 | 0.955 | 0.952 |
| Tiny static thumbnails | 0.967 | 0.955 | 0.957 |

The feature distribution audit also shows class-correlated source artifacts:

- Positive videos have much higher frame-difference statistics.
- Negative videos are brighter and have higher green/blue channel means.
- Capture month is confounded with class: August/September videos are almost
  entirely positive, while July/October videos are mostly negative.
- Positive videos include all `.mp4` files in the current manifest.

These findings mean deep model performance may be driven by source, recording,
background, lighting, or framing cues.

The current best stricter split is `data_source_aware/`, but it is still not a
valid substitute for subject/source-independent external testing.

## Required Reporting Standard

- Report current results as fixed hold-out results only.
- Do not call cached results 5-fold CV unless fold-specific cached splits are
  created and each fold trains from scratch.
- Use AUC as the primary metric because it is threshold-free.
- Report thresholded metrics only with the threshold selection rule clearly
  stated.
- Report temperature scaling separately and cautiously because the validation
  set is small.
- Always include the shortcut baseline audit next to deep model results.

## Hyperparameter Tuning

- Keep the test set untouched until final reporting.
- Choose hyperparameters using only the training and validation sets.
- Use validation loss for early stopping.
- Use validation metrics for threshold selection.
- Prefer a small predefined grid over exploratory tuning after seeing test
  performance.
- If enough grouped data exists, use nested grouped CV:
  - inner loop: hyperparameter and threshold selection;
  - outer loop: unbiased performance estimation.
- If the dataset remains small, prefer one conservative configuration and
  repeated grouped hold-outs over a large tuning search.

## Split Requirements

Preferred split priority:

1. External test set from a separate source/site/session.
2. Subject-level split where the same child never appears in more than one
   split.
3. Source/session-level split where capture date, device, room/background, and
   collection batch are balanced across labels.
4. Random file-level split only for debugging, not for final claims.

Minimum manifest fields for trustworthy splitting:

- subject ID;
- session/source ID;
- capture date or batch;
- camera/device/site when available;
- raw video path;
- label;
- split/fold;
- raw file hash;
- decoded frame-array hash.

## Required Sanity Checks

Run these before claiming a result:

```powershell
.\asd\Scripts\python.exe -m experiments.shortcut_baselines
.\asd\Scripts\python.exe -m experiments.feature_distribution_audit
.\asd\Scripts\python.exe -m experiments.debug_shuffle_labels --model simple3dlstm --epochs 20 --batch_size 8
.\asd\Scripts\python.exe -m experiments.debug_shuffle_labels --model r3d --epochs 20 --batch_size 8
.\asd\Scripts\python.exe -m unittest discover -s tests -v
```

## External Guidance

- scikit-learn documents cross-validation, model selection, and nested CV as
  tools for separating model selection from performance evaluation:
  https://scikit-learn.org/stable/model_selection.html
- TRIPOD+AI emphasizes transparent reporting of prediction model development
  and validation:
  https://www.tripod-statement.org/scope/
- CLAIM 2024 recommends clear reporting of AI evaluation and uses external
  testing terminology for evaluation on data from another source:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11304031/
