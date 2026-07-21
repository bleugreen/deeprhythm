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

Acc1 accepts predictions within 4% of the reference, the conventional tolerance in tempo-estimation literature. Acc2
additionally accepts half, double, one-third, and triple metrical levels. The tolerance is configurable with
`--tolerance`; use `--tolerance 0.02` only when comparing with this project's historical README benchmark.

## DeepRhythm 0.7 results

These results were measured on 2026-07-20 on an Apple M4 Max with 128 GiB RAM, PyTorch 2.13.0, Librosa 0.11.0,
and the MPS backend. The evaluated weights have SHA-256
`c7cc8cc0425929cd2bf695474d7ec1fd63ed0d0a4a68f361d4e4b57bd9b3d9c4`.

| Dataset | Evaluated | Method | Acc1 | Acc2 | Batch total | Batch ms/audio |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Ballroom | 698 | DeepRhythm 0.7 | 61.17% | 86.96% | 9.35 s | 13.40 |
| Ballroom | 698 | Librosa | 62.61% | 86.39% | 3.61 s | 5.18 |
| GTZAN | 998 | DeepRhythm 0.7 | 68.44% | 91.78% | 12.10 s | 12.11 |
| GTZAN | 998 | Librosa | 68.44% | 87.68% | 4.91 s | 4.92 |
| GiantSteps v2 | 661 | DeepRhythm 0.7 | 71.26% | 98.49% | 36.09 s | 54.35 |
| GiantSteps v2 | 661 | Librosa | 36.46% | 52.19% | 15.79 s | 23.77 |

The dataset totals measure warm-cache, parallel batch throughput, not single-file latency. This benchmark uses
128 eight-second clips per accelerator batch and eight audio-loader workers; Librosa uses eight CPU threads.
DeepRhythm's total includes model and feature-kernel initialization once per dataset. The per-audio values are total
wall time divided by files processed; they must not be interpreted as the latency of calling `predict` for one file.

### Batch-size selection

The 128-clip setting is not a memory limit. Historical scripts used 256 clips for training and Librosa preprocessing,
and 1,024 clips with 16 workers for offline CUDA HCQM generation. A two-run sweep on the full 999-file GTZAN audio
workload produced:

| Clips per batch | Mean wall time |
| ---: | ---: |
| 128 | 12.45 s |
| 256 | 12.43 s |
| 512 | 12.37 s |
| 1,024 | 14.10 s |

All batch sizes produced identical BPM predictions. Larger batches fit in the M4 Max's unified memory but do not improve
MPS throughput; 1,024 is measurably slower. A separate loader sweep found eight workers faster and more stable than
four or sixteen. The benchmark therefore keeps 128 as the conservative default and exposes `--batch-size` for
device-specific tuning.

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

For a device comparison, DeepRhythm processed the full 999-file GTZAN audio workload in 12.45 seconds on MPS and
33.39 seconds on CPU with the same 128-clip batch and eight loaders. MPS is 2.68 times faster for throughput. Warm
serial latency is closer: 32.62 ms on MPS versus 36.94 ms on CPU. CPU and MPS produced identical BPM predictions.

### GTZAN by genre

The aggregate GTZAN score hides the model's intended-domain performance. Genre rows use the same predictions as the
full GTZAN run; timing is intentionally reported only at the full-dataset level because model initialization and batch
occupancy cannot be assigned meaningfully to individual genres.

| Genre | Tracks | DeepRhythm Acc1 / Acc2 | Librosa Acc1 / Acc2 |
| --- | ---: | ---: | ---: |
| Blues | 100 | 63% / 90% | 65% / 85% |
| Classical | 100 | 46% / 64% | 44% / 62% |
| Country | 100 | 54% / 93% | 69% / 99% |
| Disco | 100 | **96% / 98%** | 93% / 97% |
| Hip-hop | 100 | **97% / 100%** | 76% / 88% |
| Jazz | 99 | 45.45% / 87.88% | 54.55% / 81.82% |
| Metal | 100 | 56% / 91% | 63% / 89% |
| Pop | 100 | **80% / 96%** | 73% / 89% |
| Reggae | 99 | 64.65% / 98.99% | 70.71% / 96.97% |
| Rock | 100 | 82% / 99% | 76% / 89% |

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
