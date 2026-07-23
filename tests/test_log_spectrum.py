import numpy as np
import pytest
import torch

from deeprhythm.audio_proc.log_spectrum import (
    compute_log_spectrum,
    make_log_spectrum_filter,
)
from deeprhythm.utils import load_and_split_audio, split_audio


def test_log_spectrum_shape_and_range():
    audio = torch.linspace(-1, 1, 22050)
    spectrum = compute_log_spectrum(audio, duration=2)

    assert spectrum.shape == (81, 1 + 2 * 22050 // 512)
    assert float(spectrum.amin()) == pytest.approx(0)
    assert float(spectrum.amax()) == pytest.approx(1)


def test_log_spectrum_rejects_multichannel_audio():
    with pytest.raises(ValueError, match="one-dimensional"):
        compute_log_spectrum(torch.zeros(2, 22050))


def test_log_filter_has_normalized_nonempty_bands():
    matrix = make_log_spectrum_filter()
    assert matrix.shape == (81, 1025)
    assert torch.allclose(matrix.sum(dim=1), torch.ones(81))


def test_load_and_split_audio_uses_canonical_loader(monkeypatch):
    audio = np.zeros(16 * 22050, dtype=np.float32)
    calls = []

    def fake_load(filename, sr):
        calls.append((filename, sr))
        return audio

    monkeypatch.setattr("deeprhythm.utils.load_audio", fake_load)
    clips = load_and_split_audio("track.wav")

    assert calls == [("track.wav", 22050)]
    assert clips.shape == (2, 8 * 22050)


def test_split_audio_honors_clip_budget():
    audio = np.zeros(20 * 22050, dtype=np.float32)
    clips = split_audio(audio, 22050, max_clips=2)

    assert clips.shape == (2, 8 * 22050)