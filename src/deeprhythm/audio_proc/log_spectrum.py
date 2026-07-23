"""Tensor-native log-frequency spectrum for temporal rhythm models."""

import numpy as np
import torch


def make_log_spectrum_filter(n_fft=2048, num_bands=81, *, device=None):
    """Create the logarithmic filter used by the temporal beat-rate branch."""
    num_bins = n_fft // 2 + 1
    edges = np.floor(
        np.logspace(np.log10(1), np.log10(num_bins), num_bands + 1) - 1
    ).astype(int)
    edges[-1] = num_bins
    for index in range(1, len(edges)):
        if edges[index] <= edges[index - 1]:
            edges[index] = edges[index - 1] + 1

    matrix = torch.zeros(num_bands, num_bins, device=device)
    for index in range(num_bands):
        start = edges[index]
        end = edges[index + 1] if index < num_bands - 1 else num_bins
        matrix[index, start:end] = 1 / max(1, end - start)
    return matrix


def compute_log_spectrum(
    audio,
    *,
    sample_rate=22050,
    duration=30,
    n_fft=2048,
    hop_length=512,
    filter_matrix=None,
):
    """Compute a normalized log-frequency spectrum from an already-loaded waveform.

    Reusing the waveform avoids a second decode and resample when this feature is
    paired with the HCQM tempo model.
    """
    audio = torch.as_tensor(audio, dtype=torch.float32)
    if audio.ndim != 1:
        raise ValueError("audio must be a one-dimensional mono waveform")

    target_samples = sample_rate * duration
    audio = audio[:target_samples]
    if audio.numel() < target_samples:
        audio = torch.nn.functional.pad(audio, (0, target_samples - audio.numel()))

    window = torch.hann_window(n_fft, device=audio.device)
    magnitude = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
        pad_mode="constant",
        return_complex=True,
    ).abs()
    decibels = 10 * torch.log10(magnitude.clamp_min(1e-5))
    if filter_matrix is None:
        filter_matrix = make_log_spectrum_filter(
            n_fft=n_fft, device=audio.device
        )
    spectrum = filter_matrix.to(audio.device) @ decibels
    minimum = spectrum.amin()
    return (spectrum - minimum) / (spectrum.amax() - minimum).clamp_min(1e-6)


def compute_log_spectrum_from_clips(
    magnitude,
    *,
    sample_rate=22050,
    duration=30,
    hop_length=512,
    filter_matrix=None,
):
    """Build a temporal spectrum by concatenating existing clip STFT frames.

    Magnitude has shape (clips, frequency, frames) and should contain four
    eight-second clips. The independently centered clip frames are kept: the
    temporal model is robust to those boundaries, and dropping them would
    create a larger timing discontinuity.
    """
    if magnitude.ndim != 3 or magnitude.shape[0] == 0:
        raise ValueError(
            "magnitude must have shape (clips, frequency, frames)"
        )
    if filter_matrix is None:
        n_fft = (magnitude.shape[1] - 1) * 2
        filter_matrix = make_log_spectrum_filter(
            n_fft=n_fft, device=magnitude.device
        )
    decibels = 10 * torch.log10(magnitude.clamp_min(1e-5))
    bands = torch.einsum(
        "bf,cft->cbt", filter_matrix.to(magnitude.device), decibels
    )
    spectrum = bands.permute(1, 0, 2).reshape(bands.shape[1], -1)
    target_frames = 1 + duration * sample_rate // hop_length
    spectrum = spectrum[:, :target_frames]
    if spectrum.shape[1] < target_frames:
        spectrum = torch.nn.functional.pad(
            spectrum, (0, target_frames - spectrum.shape[1])
        )
    minimum = spectrum.amin()
    return (spectrum - minimum) / (spectrum.amax() - minimum).clamp_min(1e-6)