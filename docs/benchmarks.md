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

| Dataset | Evaluated | Method | Acc1, 2% | Acc2, 2% | Acc1, 4% | Acc2, 4% | Time |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Ballroom | 698 | DeepRhythm 0.7 | 56.02% | 79.23% | 61.17% | 86.96% | 9.35 s |
| Ballroom | 698 | Librosa | 49.57% | 68.77% | 62.61% | 86.39% | 3.61 s |
| GTZAN | 998 | DeepRhythm 0.7 | 63.73% | 85.37% | 68.44% | 91.78% | 12.10 s |
| GTZAN | 998 | Librosa | 56.61% | 72.04% | 68.44% | 87.68% | 4.91 s |
| GiantSteps v2 subset | 386 | DeepRhythm 0.7 | 66.32% | 92.23% | 70.98% | 98.45% | 20.51 s |
| GiantSteps v2 subset | 386 | Librosa | 20.73% | 36.53% | 35.23% | 51.55% | 4.64 s |

Timing covers audio loading, feature extraction, and inference after imports and model setup. Dataset audio was warm
in the operating-system cache for both methods, so these timings compare compute paths rather than cold disk access.

## Dataset provenance and exclusions

Ballroom audio and annotations were downloaded and checksum-validated with `mirdata` 1.0.0; all 698 indexed tracks
were evaluated. GTZAN annotations came from `mirdata`, and audio came from the CC BY 4.0 `m-a-p/GTZAN` mirror on
Hugging Face because the original Marsyas URL no longer responds. `jazz.00054` was excluded because its checksum is
invalid and it cannot be decoded; `reggae.00086` was excluded because the tempo annotation is absent.

The official GiantSteps distribution no longer provides audio. The evaluated audio came from revision
`3f1992d1c8e3d01e06217c9e20421a1dd7d874a1` of `nicolaus625/cmi` on Hugging Face, which contains 388 of the 664
indexed tracks. Two of those tracks do not have usable v2 annotations, leaving 386. This is explicitly a subset
result and must not be presented as a full-dataset GiantSteps score. The highest-confidence v2 reference tempo was
used for each included track.