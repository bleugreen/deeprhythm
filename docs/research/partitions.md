# Canonical dataset partitions

The manifests in `data/splits` are the canonical v0.8 train, validation, and test selection. Generate them with
`python -m deeprhythm.partitions --datasets-root ~/mir_datasets`; use `--validate-only` to check committed files
without access to audio. Rows contain the dataset-relative `audio_path`, audio MD5, reference `tempo`, `group_key`,
fold, source fold, duplicate fingerprint, and annotation provenance. `SHA256SUMS` identifies every committed JSONL
manifest and is verified by `--validate-only`.

## GiantSteps

The source is `giantsteps-tempo-dataset` commit `0b7d47b`, using its canonical audio MD5 files and crowdsourced v2
tempo annotations. Official folds 1–7 are training, fold 8 is validation, and folds 9–10 are test. Three official
entries with non-positive v2 tempo (`1327052`, `3041381`, and `3041383`) are omitted, leaving 661 of 664 entries. GiantSteps supplies no
artist identity in the checked metadata, so its group key is the Beatport track identifier. This preserves the
official split but **does not establish artist disjointness**; no artist identities are inferred.

## GTZAN

Tempo comes from the checked-out `gtzan_tempo_beat` annotations. Selection starts from the fault-filtered
Kereliuk–Sturm–Larsen partition at GTZAN metadata commit `fedc781`: 443 training, 197 validation, and 290 test
entries before the historical DeepRhythm exclusions. `jazz.00054` is removed explicitly; `reggae.00086` is already
absent from the fault-filtered source. Those files still cross some artist keys in the current index, so every known
artist is moved wholly to its most protected source fold (test before validation before training). Artist keys are
parsed verbatim from Sturm's `index.txt`. Rows whose index artist is unavailable receive a per-track key rather than
an invented identity. The final split is 422 training, 194 validation, and 313 test tracks. Twenty-four tracks move
from their source fold to coalesce artists, and 14 entries use per-track keys.

## Ballroom

Audio and tempo annotations are Ballroom B_1.0. The CPJKU annotation README reports four exact replicas and nine
recording replicas; the second item of every published pair is excluded. Remaining tracks are grouped by the only
identity evidence encoded locally: the `Albums-<album>-<track>` album token or the first four digits of the
six-digit `Media-` identifier. These are album/media groups, **not claimed artist identities**. Groups are assigned
deterministically while balancing dance style and 20-BPM tempo bins toward 70/10/20 percent. The 53 indivisible
groups yield 502 training, 49 validation, and 134 test tracks after 13 published-replica exclusions.

Chromaprint is not installed in the reproducible environment and the source does not ship Extended Ballroom's
fingerprints. The `fingerprint` field therefore detects byte-identical audio through MD5, while the published
recording-replica list is enforced by exclusion. This does not detect unknown perceptual duplicates. External
corpora must add a real acoustic fingerprint comparison against all test audio before training.

## Historical v0.7 archive and comparability

`data/splits/archive-v0.7` records the obtainable full-set selections: GiantSteps v2 with usable annotations,
Ballroom's complete 698 tracks, and GTZAN minus the two historical exclusions. They are immutable selection records,
not leakage-safe training folds. The v0.8 held-out test results are not directly comparable to the published v0.7
full-set headlines. Reports must rescore v0.7 predictions on the new test manifests for an apples-to-apples change.

## Validation guarantees

`validate_partitions` rejects duplicate paths, group keys spanning folds, and fingerprint values spanning folds.
Tests exercise each failure mode and validate every committed canonical manifest. MD5 is retained because the
upstream GiantSteps distribution itself uses MD5 and because these checksums identify the exact local audio bytes;
it is not used as a cryptographic trust primitive.