import numpy as np
import pytest
import soundfile as sf


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: tests that require model weights (deselect with '-m \"not slow\"')")


@pytest.fixture(scope="session")
def predictor():
    from deeprhythm.model.predictor import DeepRhythmPredictor
    return DeepRhythmPredictor(device='cpu', quiet=True)


def make_click_track(bpm, duration=16, sr=22050):
    """Generate periodic impulses at exact BPM intervals."""
    samples = int(sr * duration)
    audio = np.zeros(samples, dtype=np.float32)
    interval = 60.0 / bpm  # seconds between clicks
    interval_samples = int(interval * sr)
    for i in range(0, samples, interval_samples):
        end = min(i + int(sr * 0.01), samples)  # 10ms click
        audio[i:end] = 0.8
    return audio


def make_sine_wave(freq, duration=16, sr=22050):
    """Generate a pure sine wave."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def make_silence(duration=16, sr=22050):
    """Generate silent audio."""
    return np.zeros(int(sr * duration), dtype=np.float32)


def write_wav(audio, sr, tmp_path, name="test.wav"):
    """Write audio to a temp .wav file, returns path."""
    path = tmp_path / name
    sf.write(str(path), audio, sr)
    return str(path)
