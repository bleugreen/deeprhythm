# DeepRhythm: High-Speed Tempo Prediction

DeepRhythm is a convolutional neural network designed for rapid, precise tempo prediction for modern music. It runs on anything that supports Pytorch (I've tested Ubunbu, MacOS, Windows, Raspbian).

Audio is batch-processed using a vectorized Harmonic Constant-Q Modulation (HCQM), drastically reducing computation time by avoiding the usual bottlenecks encountered in feature extraction.

[more details here](https://bleu.green/deeprhythm)

## Classification Process

1. Split input audio into 8 second clips `[len_batch, len_audio]`
2. Compute the HCQM of each clip
   1. Compute STFT `[len_batch, stft_bands, len_audio/hop]`
   2. Sum STFT bins into 8 log-spaced bands using filter matrix `[len_batch, 8, len_audio/hop]`
   3. Flatten bands for parallel CQT processing `[len_batch*8, len_audio/hop]`
   4. For each of the six harmonics, compute the CQT `[6, len_batch*8, num_cqt_bins]`
   5. Reshape `[len_batch, num_cqt_bins, 8, 6]`
3. Feed HCQM through CNN `[len_batch, num_classes (256)]`
4. Softmax the outputs to get probabilities
5. Choose the class with the highest probability and convert to bpm (bpms = `[len_batch]`)

## Benchmarks

| Method                  | Acc1 (%)  | Acc2 (%)  | Avg. Time (s) | Total Time (s) |
| ----------------------- | --------- | --------- | ------------- | -------------- |
| DeepRhythm (cuda)       | **95.91** | 96.54     | **0.021**     | 20.11          |
| DeepRhythm (cpu)        | **95.91** | 96.54     | 0.12          | 115.02         |
| TempoCNN (cnn)          | 84.78     | **97.69** | 1.21          | 1150.43        |
| TempoCNN (fcn)          | 83.53     | 96.54     | 1.19          | 1131.51        |
| Essentia (multifeature) | 87.93     | 97.48     | 2.72          | 2595.64        |
| Essentia (percival)     | 85.83     | 95.07     | 1.35          | 1289.62        |
| Essentia (degara)       | 86.46     | 97.17     | 1.38          | 1310.69        |
| Librosa                 | 66.84     | 75.13     | 0.48          | 460.52         |

- Test done on 953 songs, mostly Electronic, Hip Hop, Pop, and Rock
- Acc1 = Prediction within +/- 2% of actual bpm
- Acc2 = Prediction within +/- 2% of actual bpm or a multiple (e.g. 120 ~= 60)
- Timed from filepath in to bpm out (audio loading, feature extraction, model inference)
- I could only get TempoCNN to run on cpu (it requires Cuda 10)

## Installation

To install DeepRhythm, ensure you have Python and pip installed. Then run:

```bash
pip install deeprhythm
```

## Usage

### CLI Inference

#### Single

```bash
python -m deeprhythm.infer /path/to/song.wav -cq
> ([bpm], [confidence])
```

Flags:

- `-c`, `--conf` - include confidence scores
- `-d`, `--device [cuda/cpu/mps]` - specify model device
- `-q`, `--quiet` - prints only bpm/conf

#### Batch

To predict the tempo of all songs in a directory, run

```bash
python -m deeprhythm.batch_infer /path/to/dir
```

This will create in a jsonl file mapping filepath to predicted BPM.

Flags:

- `-o output_path.jsonl` - provide a custom output path (default 'batch_results.jsonl`)
- `-c`, `--conf` - include confidence scores
- `-d`, `--device [cuda/cpu/mps]` - specify model device
- `-q`, `--quiet` - doesn't print status / logs

### Python Inference

To predict the tempo of a song:

```python
from deeprhythm import DeepRhythmPredictor

model = DeepRhythmPredictor()

tempo = model.predict('path/to/song.mp3')

# to include confidence
tempo, confidence = model.predict('path/to/song', include_confidence=True)

print(f"Predicted Tempo: {tempo} BPM")
```

Audio is loaded with librosa, which supports most audio formats

## References

[1] Hadrien Foroughmand and Geoffroy Peeters, “Deep-Rhythm for Global Tempo Estimation in Music”, in Proceedings of the 20th International Society for Music Information Retrieval Conference, Delft, The Netherlands, Nov. 2019, pp. 636–643. doi: 10.5281/zenodo.3527890.

[2] K. W. Cheuk, H. Anderson, K. Agres and D. Herremans, "nnAudio: An on-the-Fly GPU Audio to Spectrogram Conversion Toolbox Using 1D Convolutional Neural Networks," in IEEE Access, vol. 8, pp. 161981-162003, 2020, doi: 10.1109/ACCESS.2020.3019084.
