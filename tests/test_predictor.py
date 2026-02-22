import numpy as np
import pytest
import soundfile as sf

from deeprhythm.utils import AudioTooShortError


def _make_click_track(bpm, duration=16, sr=22050):
    """Periodic impulses at exact BPM intervals."""
    samples = int(sr * duration)
    audio = np.zeros(samples, dtype=np.float32)
    interval_samples = int(60.0 / bpm * sr)
    for i in range(0, samples, interval_samples):
        end = min(i + int(sr * 0.01), samples)
        audio[i:end] = 0.8
    return audio


def _make_silence(duration=16, sr=22050):
    return np.zeros(int(sr * duration), dtype=np.float32)


def _write_wav(audio, sr, tmp_path, name="test.wav"):
    path = tmp_path / name
    sf.write(str(path), audio, sr)
    return str(path)


pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Prediction Modes
# ---------------------------------------------------------------------------

def test_predict_returns_float_in_range(predictor, tmp_path):
    """Result should be a float in [30, 286] for click track audio."""
    audio = _make_click_track(120, duration=16)
    path = _write_wav(audio, 22050, tmp_path)
    result = predictor.predict(path)
    assert isinstance(result, float)
    assert 30 <= result <= 286


def test_predict_with_confidence(predictor, tmp_path):
    """Should return (bpm, confidence) tuple; confidence in (0, 1]."""
    audio = _make_click_track(120, duration=16)
    path = _write_wav(audio, 22050, tmp_path)
    bpm, conf = predictor.predict(path, include_confidence=True)
    assert isinstance(bpm, float)
    assert 30 <= bpm <= 286
    assert 0 < conf <= 1.0


def test_predict_from_audio_matches_predict(predictor, tmp_path):
    """Same audio via file vs array should produce the same BPM."""
    sr = 22050
    audio = _make_click_track(120, duration=16, sr=sr)
    path = _write_wav(audio, sr, tmp_path)
    bpm_file = predictor.predict(path)
    bpm_array = predictor.predict_from_audio(audio, sr)
    assert bpm_file == bpm_array


def test_predict_per_frame_count(predictor, tmp_path):
    """24s audio -> 3 per-frame predictions."""
    sr = 22050
    audio = _make_click_track(120, duration=24, sr=sr)
    path = _write_wav(audio, sr, tmp_path)
    bpms = predictor.predict_per_frame(path)
    assert len(bpms) == 3


def test_predict_per_frame_with_confidence(predictor, tmp_path):
    """Should return (bpms_list, confidences_list) of equal length."""
    sr = 22050
    audio = _make_click_track(120, duration=24, sr=sr)
    path = _write_wav(audio, sr, tmp_path)
    bpms, confs = predictor.predict_per_frame(path, include_confidence=True)
    assert len(bpms) == len(confs) == 3


def test_predict_per_frame_values_in_range(predictor, tmp_path):
    """All per-frame BPMs should be in [30, 286]."""
    sr = 22050
    audio = _make_click_track(120, duration=24, sr=sr)
    path = _write_wav(audio, sr, tmp_path)
    bpms = predictor.predict_per_frame(path)
    for bpm in bpms:
        assert 30 <= bpm <= 286


# ---------------------------------------------------------------------------
# Rich Synthetic Audio (click tracks)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target_bpm", [90, 120, 140])
def test_predict_click_track(predictor, tmp_path, target_bpm):
    """Click track at target BPM should predict within +/-10%."""
    audio = _make_click_track(target_bpm, duration=24)
    path = _write_wav(audio, 22050, tmp_path)
    result = predictor.predict(path)
    tolerance = target_bpm * 0.10
    assert abs(result - target_bpm) <= tolerance, (
        f"Expected ~{target_bpm} BPM, got {result}"
    )


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

def test_predict_silence(predictor, tmp_path):
    """Silent audio should return a valid BPM (no crash), low confidence."""
    audio = _make_silence(duration=16)
    path = _write_wav(audio, 22050, tmp_path)
    bpm, conf = predictor.predict(path, include_confidence=True)
    assert 30 <= bpm <= 286
    # Confidence on silence should be relatively low
    assert conf < 0.5


def test_predict_exactly_8_seconds(predictor, tmp_path):
    """Single clip boundary should work and return a valid BPM."""
    sr = 22050
    audio = _make_click_track(120, duration=8, sr=sr)
    # Ensure exactly 8s worth of samples
    audio = audio[:sr * 8]
    path = _write_wav(audio, sr, tmp_path)
    result = predictor.predict(path)
    assert 30 <= result <= 286


def test_predict_just_under_8_seconds(predictor, tmp_path):
    """Audio just under 8 seconds should raise AudioTooShortError."""
    sr = 22050
    # 7.99 seconds
    audio = np.zeros(sr * 8 - 100, dtype=np.float32)
    path = _write_wav(audio, sr, tmp_path)
    with pytest.raises(AudioTooShortError):
        predictor.predict(path)


def test_predict_nonexistent_file(predictor):
    """Non-existent file should raise an error."""
    with pytest.raises(Exception):
        predictor.predict("/nonexistent/path/to/audio.wav")


def test_predict_single_vs_multi_clip(predictor, tmp_path):
    """8s vs 16s of same pattern should both return valid results."""
    sr = 22050
    audio_short = _make_click_track(120, duration=8, sr=sr)[:sr * 8]
    audio_long = _make_click_track(120, duration=16, sr=sr)

    path_short = _write_wav(audio_short, sr, tmp_path, name="short.wav")
    path_long = _write_wav(audio_long, sr, tmp_path, name="long.wav")

    bpm_short = predictor.predict(path_short)
    bpm_long = predictor.predict(path_long)

    assert 30 <= bpm_short <= 286
    assert 30 <= bpm_long <= 286
