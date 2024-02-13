# DeepRhythm: High-Speed Tempo Prediction

## Introduction
DeepRhythm is a Convolutional Neural Network (CNN) designed for rapid, precise tempo prediction, specifically on modern music.

The implementation is heavily inspired by [1].

Audio is batch-processed using a vectorized HCQM, drastically reducing computation time by avoiding the common bottlenecks encountered in feature extraction.

## Benchmarks

| Method                | Acc1 (%) | Acc2 (%) | Avg. Time (s) | Total Time (s) |
|-----------------------|------|------|-----------|------------|
| Essentia (multifeature) | 79.15 | 94.19 | 2.78 | 1635.48 |
| Essentia (Percival)   | 80.51 | 94.87 | 1.46 | 851.91 |
| Essentia (degara)     | 77.26 | 91.97 | 1.40 | 820.85 |
| Librosa               | 52.82 | 63.93 | 0.51 | 299.68 |
| DeepRhythm (cpu)      | 90.77 | 96.75 | 0.127 | 74.43 |
| DeepRhythm (cuda)     | 90.77 | 96.75 | 0.0235 | 13.74 |

- Test done on 586 songs, mostly Hip Hop, Electronic, Pop, and Rock
- Acc1 = Prediction within +/- 2% of actual bpm
- Acc2 = Prediction within +/- 2% of actual bpm or a multiple (e.g. 120 ~= 60)

## Installation
To install DeepRhythm, ensure you have Python and pip installed. Then run:
```bash
pip install git+https://github.com/KinWaiCheuk/nnAudio.git#subdirectory=Installation

pip install deeprhythm==0.0.6
```

Note: nnAudio currently needs to be installed separately since the pip package is out of date. Hopefully this will be resolved soon.

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
