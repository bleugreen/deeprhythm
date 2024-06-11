# DeepRhythm: High-Speed Tempo Prediction

## Introduction

DeepRhythm is a convolutional neural network designed for rapid, precise tempo prediction for modern music. It runs on anything that supports Pytorch (I've tested Ubunbu, MacOS, Windows, Raspbian)

## HCQM

(reworded from “Deep-Rhythm for Global Tempo Estimation in Music”, by Foroughmand and Peeters [1].)

The Constant Q Transform (CQT) is a tool used to analyze sound frequencies over time. It breaks down the frequency spectrum into bins that are spaced logarithmically, meaning they're closer together at low frequencies and wider apart at high frequencies. This aligns with how we hear sounds, making it great for music analysis as it captures details of pitches and notes very precisely.

It is normally performed with a hop length around 10-25ms (the window size varies by frequency) and 80-120 bins (covering ~50-5kHz), which results in a solid melodic representation of the given audio.

With the HCQM (Harmonic Constant-Q Modulation), Foroughmand and Peeters creatively repurpose the CQT for rhythm detection. Instead of scanning a few milliseconds, they give it an 8-second window. Rather than the standard 81 bins covering 50 Hz to 1 kHz, it utilizes 256 bins tailored to span from 30 bpm to 286 bpm (approximately 0.5 Hz to 4.76 Hz). This adjustment results in a highly detailed, narrow, and low frequency window, which delineates how prevalent each potential bpm is within the track. For instance, in a song with a tempo of 120 bpm, this method would highlight spikes at 30, 60, 120 (predominantly), and 240 bpm. Each element of the song that recurs on this 8-second scale contributes to peaks in the CQT, e.g. a quarter-notea hi hat would look like a continuous 2 Hz (120 bpm) tone in the transformed data.

Audio is batch-processed using a vectorized Harmonic Constant-Q Modulation (HCQM), drastically reducing computation time by avoiding the usual bottlenecks encountered in feature extraction.

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

To predict the tempo of a song:

```python
from deeprhythm import DeepRhythmPredictor

model = DeepRhythmPredictor()
tempo = model.predict('path/to/song.mp3')
print(f"Predicted Tempo: {tempo} BPM")
```

## References

[1] Hadrien Foroughmand and Geoffroy Peeters, “Deep-Rhythm for Global Tempo Estimation in Music”, in Proceedings of the 20th International Society for Music Information Retrieval Conference, Delft, The Netherlands, Nov. 2019, pp. 636–643. doi: 10.5281/zenodo.3527890.

[2] K. W. Cheuk, H. Anderson, K. Agres and D. Herremans, "nnAudio: An on-the-Fly GPU Audio to Spectrogram Conversion Toolbox Using 1D Convolutional Neural Networks," in IEEE Access, vol. 8, pp. 161981-162003, 2020, doi: 10.1109/ACCESS.2020.3019084.
