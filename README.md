# DeepRhythm: High-Speed Tempo Prediction

## Introduction
DeepRhythm is a Convolutional Neural Network (CNN) designed for rapid, precise tempo prediction, specifically on modern music.

The implementation is heavily inspired by [1].

Audio is batch-processed using a vectorized HCQM, drastically reducing computation time by avoiding the common bottlenecks encountered in feature extraction.

## Benchmarks

| Method                | Acc1 (%) | Acc2 (%) | Avg. Time (s) | Total Time (s) |
|-----------------------|------|------|-----------|------------|
| DeepRhythm (cuda)     | **95.91** | 96.54 | **0.021** | 20.11 |
| DeepRhythm (cpu)      | **95.91** | 96.54 | 0.12 | 115.02 |
| TempoCNN (cnn)        | 84.78 | **97.69** | 1.21 | 1150.43 |
| TempoCNN (fcn)        | 83.53 | 96.54 | 1.19 | 1131.51 |
| Essentia (multifeature) | 87.93 | 97.48 | 2.72 | 2595.64 |
| Essentia (percival)   | 85.83 | 95.07 | 1.35 | 1289.62 |
| Essentia (degara)     | 86.46 | 97.17 | 1.38 | 1310.69 |
| Librosa               | 66.84 | 75.13 | 0.48 | 460.52 |

- Test done on 953 songs, mostly Electronic, Hip Hop, Pop, and Rock
- Acc1 = Prediction within +/- 2% of actual bpm
- Acc2 = Prediction within +/- 2% of actual bpm or a multiple (e.g. 120 ~= 60)
- Timed from filepath in to bpm out (audio loading, feature extraction, model inference)
- I could only get TempoCNN to run on cpu (it requires Cuda 10 and I'm not downgrading my Cuda install for curiosity's sake)

## Installation
To install DeepRhythm, ensure you have Python and pip installed. Then run:
```bash
pip install deeprhythm
```

## Usage
To predict the tempo of a song with DeepRhythm:
```python
from deeprhythm import DeepRhythmPredictor

model = DeepRhythmPredictor()
tempo = model.predict('path/to/song.mp3')
print(f"Predicted Tempo: {tempo} BPM")
```

## References
[1] Hadrien Foroughmand and Geoffroy Peeters, “Deep-Rhythm for Global Tempo Estimation in Music”, in Proceedings of the 20th International Society for Music Information Retrieval Conference, Delft, The Netherlands, Nov. 2019, pp. 636–643. doi: 10.5281/zenodo.3527890.

[2] K. W. Cheuk, H. Anderson, K. Agres and D. Herremans, "nnAudio: An on-the-Fly GPU Audio to Spectrogram Conversion Toolbox Using 1D Convolutional Neural Networks," in IEEE Access, vol. 8, pp. 161981-162003, 2020, doi: 10.1109/ACCESS.2020.3019084.
