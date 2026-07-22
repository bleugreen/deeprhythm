# Metrical-level annotation

`deeprhythm-annotate` turns existing per-track tempo predictions into a resumable, keyboard-driven local annotation queue. Preparation injects each predicted BPM into PhaseFinder to obtain an aligned beat grid, chooses a 15-second onset-active excerpt, and writes one audio FLAC plus half/current/double click FLACs. Source audio is never copied into the repository.

Prepare assets from prediction JSONL containing `filename` plus either `bpm` or `deeprhythm_v07`:

```bash
deeprhythm-annotate prepare \
  --predictions predictions.jsonl \
  --output ~/mir_datasets/v08_annotations \
  --phasefinder-path ~/projects/phasefinder
```

Preparation is incremental: rows already present in `manifest.jsonl` are skipped. Each manifest row records the source SHA-256, model BPM, excerpt offset, and generated assets.

Start the local UI:

```bash
deeprhythm-annotate serve --output ~/mir_datasets/v08_annotations
```

Open `http://127.0.0.1:8765`. Preview the half/current/double click grids with `H`, `C`, and `D`. Press Space to play or pause. Labels describe how the model prediction must change: `1` needs 2× (model half), `2` correct, `3` needs ½× (model double), `4` other, and `S` skip. Every response is synchronously appended to `annotations.jsonl`; skipped tracks remain explicitly identifiable for later review.

Treat the source manifest and annotations as a pair. Split them by artist or album group before training, and never allow the same audio fingerprint to cross training and evaluation folds.
