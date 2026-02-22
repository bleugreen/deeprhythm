import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from deeprhythm.utils import AudioTooShortError, bpm_to_class, class_to_bpm, split_audio


def test_bpm_class_roundtrip():
    """bpm_to_class and class_to_bpm should roundtrip within one class width."""
    class_width = (286 - 30) / 256
    for bpm in [30, 60, 90, 120, 150, 200, 285]:
        cls = bpm_to_class(bpm)
        recovered = class_to_bpm(cls)
        assert abs(recovered - bpm) <= class_width, f"Roundtrip failed for {bpm}: got {recovered}"


def test_bpm_to_class_clamps():
    """Values outside [30, 286] should clamp to valid class range."""
    assert bpm_to_class(0) == 0
    assert bpm_to_class(500) == 255


def test_split_audio_basic():
    """split_audio should produce correct number of clips from a synthetic signal."""
    sr = 22050
    clip_length = 8
    num_clips = 3
    audio = np.random.randn(sr * clip_length * num_clips + 1000).astype(np.float32)
    clips = split_audio(audio, sr, clip_length=clip_length)
    assert clips.shape == (num_clips, sr * clip_length)


def test_split_audio_too_short():
    """split_audio should raise AudioTooShortError when audio is shorter than one clip."""
    sr = 22050
    audio = np.zeros(100, dtype=np.float32)
    with pytest.raises(AudioTooShortError):
        split_audio(audio, sr)


def test_split_audio_share_mem():
    """split_audio with share_mem=True should return a shared memory tensor."""
    sr = 22050
    audio = np.random.randn(sr * 8).astype(np.float32)
    clips = split_audio(audio, sr, share_mem=True)
    assert clips.is_shared()


def test_predictor_instantiation():
    """DeepRhythmPredictor should load model and create kernels."""
    from deeprhythm.model.predictor import DeepRhythmPredictor
    predictor = DeepRhythmPredictor(device='cpu', quiet=True)
    assert predictor.model is not None
    assert predictor.specs is not None


def test_predict_sine_wave():
    """Predicting on a synthetic sine wave should return a float in valid BPM range."""
    from deeprhythm.model.predictor import DeepRhythmPredictor

    sr = 22050
    duration = 16
    t = np.linspace(0, duration, sr * duration, dtype=np.float32)
    audio = np.sin(2 * np.pi * 2.0 * t).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, audio, sr)
        tmp_path = f.name

    try:
        predictor = DeepRhythmPredictor(device='cpu', quiet=True)
        result = predictor.predict(tmp_path)
        assert isinstance(result, float)
        assert 30 <= result <= 286
    finally:
        os.unlink(tmp_path)
