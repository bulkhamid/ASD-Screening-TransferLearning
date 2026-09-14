# Private Metadata Guide

## Why Manual Mapping Is Required

The filename is not a subject identifier. Capture dates can approximate sessions
for a shortcut audit, but they cannot prove whether two videos belong to the
same person. Subject-independent evaluation therefore requires a manually
verified or dataset-owner-confirmed private mapping. When the owner confirms
that each participant appears in exactly one video, stable pseudonyms may be
assigned automatically. That confirmation must be recorded as provenance, not
described as independent identity verification.

The current mapping uses that owner-confirmed route. The subsequent footage
review excluded one near-identical recording copy from the experiment manifest.
See [Reviewed Subject Benchmark](ReviewedSubjectBenchmark.md) for the current
102-video analysis set and completed primary results.

## Fields

- `video_id`: SHA-256 of the full raw file; generated automatically.
- `relative_path`: path below the private dataset root; generated automatically.
- `label`: `0` or `1`, derived from the class folder.
- `subject_id`: stable pseudonym shared by every video of one person.
- `session_id`: stable pseudonym shared by videos from one recording session.
- `capture_datetime`: filename-derived hint; verify when possible.
- `site_id`: optional pseudonymous collection-site code.
- `device_id`: optional pseudonymous camera/device code.
- `include`: `1` to include, `0` to exclude.
- `notes`: non-identifying rationale for exclusions or uncertainty.

Do not enter names, dates of birth, addresses, medical record numbers, contact
details, or other direct identifiers.

## Mapping Rules

- Every included row needs exactly one subject and session.
- A subject may have many sessions and videos.
- A session belongs to exactly one subject.
- Here, session means one participant's recording session, not a collection day
  shared by several participants. Retain date, site, and device as separate
  potential source-confounding fields.
- A subject must never have both labels.
- Use `include=0` when identity is uncertain; explain the reason in `notes`.
- Do not infer identities only from similar filenames, dates, backgrounds, or
  model embeddings.

## Storage

`private/`, `runs/`, cached clips and masks, private exclusion records, cache
manifests, predictions, and fold assignments are ignored by Git. Cache settings,
the empty schema, and aggregate diagnostics
belong in the repository. Existing Git history is not rewritten, so do not
reuse path-bearing legacy manifests in new commits.

## Workflow

```powershell
.\asd\Scripts\python.exe -m data.prepare_dataset template `
  --root "<PRIVATE_DATASET_ROOT>"

.\asd\Scripts\python.exe -m data.prepare_dataset validate `
  --root "<PRIVATE_DATASET_ROOT>"

.\asd\Scripts\python.exe -m data.prepare_dataset cache `
  --root "<PRIVATE_DATASET_ROOT>"
```

Do not execute deep benchmarks until the strict validation command succeeds.

For a dataset with explicit owner confirmation of one video per participant,
the alternative to entering pseudonyms by hand is:

```powershell
.\asd\Scripts\python.exe -m data.prepare_dataset assign_unique_subjects `
  --root "<PRIVATE_DATASET_ROOT>" --confirmed_one_video_per_subject
```

This command checks raw hashes, backs up the mapping, and records the owner
confirmation. It does not identify people from video frames. Strict validation
and footage-duplicate review still apply before training.
