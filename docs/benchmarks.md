# Tempo benchmarks

The canonical benchmark entry point is `deeprhythm-benchmark`. It evaluates the current model and Librosa's beat
tracker against the same JSONL manifest so that dataset selection and scoring cannot drift between methods.

Each manifest row contains an absolute or user-relative audio path and its reference tempo:

```json
{"audio_path": "~/mir_datasets/ballroom/B_1.0/audio/Waltz/Media-105901.wav", "tempo": 84.0}
```

Run a benchmark with:

```bash
deeprhythm-benchmark ballroom.jsonl --device mps --output-dir benchmark-results/ballroom
```

Acc1 accepts predictions within 2% of the reference. Acc2 additionally accepts half, double, one-third, and triple
metrical levels. The output also includes 4% variants because that tolerance is common in tempo-estimation papers;
the 2% figures remain directly comparable with this project's historical README benchmark.

## DeepRhythm 0.7 results

These results were measured on 2026-07-20 on an Apple M4 Max with 128 GiB RAM, PyTorch 2.13.0, Librosa 0.11.0,
and the MPS backend. The evaluated weights have SHA-256
`c7cc8cc0425929cd2bf695474d7ec1fd63ed0d0a4a68f361d4e4b57bd9b3d9c4`.

| Dataset | Evaluated | Method | Acc1, 2% | Acc2, 2% | Acc1, 4% | Acc2, 4% | Batch total | Batch ms/audio |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ballroom | 698 | DeepRhythm 0.7 | 56.02% | 79.23% | 61.17% | 86.96% | 9.35 s | 13.40 |
| Ballroom | 698 | Librosa | 49.57% | 68.77% | 62.61% | 86.39% | 3.61 s | 5.18 |
| GTZAN | 998 | DeepRhythm 0.7 | 63.73% | 85.37% | 68.44% | 91.78% | 12.10 s | 12.11 |
| GTZAN | 998 | Librosa | 56.61% | 72.04% | 68.44% | 87.68% | 4.91 s | 4.92 |
| GiantSteps v2 | 661 | DeepRhythm 0.7 | 66.72% | 92.59% | 71.26% | 98.49% | 36.09 s | 54.35 |
| GiantSteps v2 | 661 | Librosa | 22.09% | 37.22% | 36.46% | 52.19% | 15.79 s | 23.77 |

The dataset totals measure warm-cache, parallel batch throughput, not single-file latency. DeepRhythm processes up to
128 eight-second clips per accelerator batch, while Librosa uses eight CPU threads. DeepRhythm's total includes model
and feature-kernel initialization once per dataset. The per-audio values are total wall time divided by files processed;
they must not be interpreted as the latency of calling `predict` for one file.

### Single-file latency

Single-file latency was measured separately on GTZAN. The warm measurement uses a genre-balanced sample of 20
30-second tracks, processes them serially, and follows one unmeasured warm-up call. The cold measurement uses a fresh
Python process and one hip-hop track. Times start after imports.

| Method | Fresh-process first call | Warm serial mean | Warm serial median | Warm serial p95 |
| --- | ---: | ---: | ---: | ---: |
| DeepRhythm 0.7 | 2.790 s | 32.62 ms | 32.64 ms | 33.44 ms |
| Librosa | 5.093 s | 17.17 ms | 17.12 ms | 17.59 ms |

The first call includes lazy backend compilation. The warm measurement is the appropriate latency comparison for a
long-running application; the batch table is the appropriate throughput comparison for dataset processing.

### GTZAN by genre

The aggregate GTZAN score hides the model's intended-domain performance. Genre rows use the same predictions as the
full GTZAN run; timing is intentionally reported only at the full-dataset level because model initialization and batch
occupancy cannot be assigned meaningfully to individual genres.

| Genre | Tracks | DeepRhythm Acc1 / Acc2 | Librosa Acc1 / Acc2 |
| --- | ---: | ---: | ---: |
| Blues | 100 | 51% / 73% | 51% / 69% |
| Classical | 100 | 35% / 51% | 30% / 40% |
| Country | 100 | 51% / 89% | 57% / 86% |
| Disco | 100 | **96% / 98%** | 81% / 84% |
| Hip-hop | 100 | **92% / 95%** | 68% / 77% |
| Jazz | 99 | 42.42% / 79.80% | 45.45% / 67.68% |
| Metal | 100 | 48% / 81% | 50% / 71% |
| Pop | 100 | **80% / 95%** | 63% / 75% |
| Reggae | 99 | 64.65% / 97.98% | 52.53% / 72.73% |
| Rock | 100 | 77% / 94% | 68% / 78% |

## Dataset provenance and exclusions

Ballroom audio and annotations were downloaded and checksum-validated with `mirdata` 1.0.0; all 698 indexed tracks
were evaluated. GTZAN annotations came from `mirdata`, and audio came from the CC BY 4.0 `m-a-p/GTZAN` mirror on
Hugging Face because the original Marsyas URL no longer responds. `jazz.00054` was excluded because its checksum is
invalid and it cannot be decoded; `reggae.00086` was excluded because the tempo annotation is absent.

GiantSteps annotations and MD5 manifests came from the canonical dataset repository. Its original Beatport preview
endpoint now returns 404, but the canonical JKU backup still serves all 664 audio files. Every download was verified
against its published MD5 checksum. Three tracks do not have usable v2 annotations, leaving 661 evaluated tracks. Timing covers inference over all 664
downloaded files, so the reported per-audio time uses 664 as its denominator. The highest-confidence v2 reference
tempo was used for each included track.

The README's historical 953-track benchmark describes its genre mix but does not publish an obtainable manifest or
reference annotations. It remains useful historical context, but the results above are the public, independently
reproducible benchmark.
