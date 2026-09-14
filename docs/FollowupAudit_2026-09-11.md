# Follow-up Code Audit and Experiment Rerun

This snapshot predates the owner's participant confirmation. See
[Reviewed Subject Benchmark](ReviewedSubjectBenchmark.md) for the subsequent
validated mapping, repeated-footage finding, and experiment status.

Scope: the local subject benchmark, splitting, metrics, checkpoint selection,
experiment configuration, and legacy shortcut rerun. Existing local changes were
preserved. No remote code was merged and no changes were published.

## Confirmed Errors and Corrections

| Issue | Evidence / impact | Correction |
| --- | --- | --- |
| Inner validation can lose the minority class | With 2 negative and 20 positive outer-training subjects at seed 42, the old splitter selected 5 positives and no negatives for validation. | Constrained class allocation reserves at least one subject of each class in both partitions; tested over multiple seeds and imbalances. |
| Invalid validation loss can overwrite the best model | NaN compares false to the previous best, allowing it into the checkpoint. | Reject non-finite loss before changing checkpoint state; reject non-finite training loss before an optimizer update. |
| Single-class AUC fabricated a chance score | Subject metrics returned 0.5 when ROC AUC was undefined. | Return null for single-class ROC/PR AUC; reject empty, invalid-label, and non-finite inputs. |
| Video metadata could be silently overwritten | Aggregation kept the last subject/session/label for a reused video ID. | Reject inconsistent video identities before aggregation. |
| Training helper relied on its caller for isolation | Direct calls could bypass split checks. | Validate isolation, subject existence, and both classes before creating run artifacts. |
| Run settings could disagree with execution | Temperature scaling could be requested but ignored; omitting normal evaluation failed after training; duplicate seeds collided; epoch limits were absent from the run hash. | Validate options before cache access; include epoch limits and manifest SHA-256 in the run specification. |
| Legacy audit could trust the wrong manifest order | Counts and labels cannot detect reordering within a class. | Verify every raw and decoded hash; close NPZ files explicitly. Read historical manifests from Git in memory. |
| Unknown dates became one group per video | This could silently make the date audit an ordinary video split. | Fail on missing dates instead of inventing independent groups. |
| Interrupted training lost completed histories | Histories were written only after training. | Write history atomically after each validation epoch. |

## Experiment Rerun

Reran four shortcut baselines on the existing source-aware cache, holding out
capture-date groups. All 107 raw and decoded hashes matched the historical
manifest before fitting. No private manifest was restored to a public location.

| Features | Out-of-fold ROC AUC | Balanced accuracy |
| --- | ---: | ---: |
| Duration, resolution, extension, capture time | 0.704752 | 0.682739 |
| Color, brightness, padding | 0.690776 | 0.692348 |
| Frame-difference motion statistics | 0.956674 | 0.887317 |
| Tiny thumbnails | 0.733403 | 0.767296 |

These match the previous audit results. All 29 capture-date groups are
single-class. One held-out fold has only positive videos, so the table reports
pooled out-of-fold AUC, not a mean of per-fold AUCs. Date groups do not establish
subject independence. The motion baseline demonstrates a simple predictive
signal; it does not establish whether that signal is behavior, recording
conditions, or a combination. It cannot validate clinical generalization.

Local aggregate output: `runs/audit_rerun_20260911/date_shortcuts.json`.

Reproduce while the historical manifest exists at HEAD:

```powershell
.\asd\Scripts\python.exe -m experiments.grouped_shortcut_baselines --data_dir data_source_aware --manifest_git_ref HEAD:data_source_aware/split_manifest.csv --output runs/audit_rerun_20260911/date_shortcuts.json
```

## Deep Training Status

Strict validation was rerun: all 107 included videos still lack subject and
session IDs; the remaining raw file is excluded. The corrected subject cache
and real-data deep benchmark were not run. No identities were invented.

Complete the pseudonymous person and recording-session mapping in
`private/subject_sessions.csv`, then run strict validation before caching or deep
training. This protocol requires at least three distinct subjects in each class.
Repeated recordings of one person must keep the same subject ID.

Synthetic grouped training and one-batch learning checks are software tests;
they are not estimates of ASD screening performance.

## Verification

- Final full suite: 71 tests run, 70 passed, 1 skipped for the absent legacy manifest.
- All 62 project Python files passed AST parsing.
- The suite includes one-batch learning for all three primary models and a
  synthetic grouped training/evaluation run.
- `pip check` found no broken requirements; `git diff --check` passed.
- The rerun result JSON is exactly equal to the previous aggregate audit JSON.
- Strict metadata validation failed as expected for missing identities; the
  benchmark's validation-only command also stopped because no grouped cache exists.
- An initial new test used the Windows temporary directory and hit a permission
  error. It now follows the existing project-local test workspace convention;
  the focused test and final full suite pass.
