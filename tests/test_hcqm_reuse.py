import torch

from deeprhythm.audio_proc.hcqm import (
    compute_hcqm,
    compute_hcqm_from_stft,
    make_kernels,
)


def test_hcqm_accepts_shared_stft():
    audio = torch.randn(1, 22050 * 8)
    kernels = make_kernels(device="cpu")

    expected = compute_hcqm(audio, *kernels)
    magnitude = kernels[0](audio)
    actual = compute_hcqm_from_stft(magnitude, kernels[1], kernels[2])

    assert torch.equal(actual, expected)