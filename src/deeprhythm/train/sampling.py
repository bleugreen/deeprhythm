"""Tempo balancing and audio-domain tempo augmentation."""

from collections import Counter

import librosa
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


class TempoBalancedSampler(WeightedRandomSampler):
    """Sample clips inversely to their occupied tempo-bin frequency."""

    def __init__(self, tempos, *, domains=None, bin_width=10.0, num_samples=None, replacement=True, generator=None):
        tempos = np.asarray(tempos, dtype=float)
        if tempos.ndim != 1 or tempos.size == 0 or np.any(tempos <= 0):
            raise ValueError("tempos must be a non-empty sequence of positive values")
        if bin_width <= 0:
            raise ValueError("bin_width must be positive")
        bins = np.floor(tempos / bin_width).astype(int)
        if domains is None:
            counts = Counter(bins.tolist())
            weights = [1.0 / counts[value] for value in bins]
        else:
            if len(domains) != len(tempos):
                raise ValueError("domains and tempos must be aligned")
            cells = list(zip(domains, bins.tolist()))
            counts = Counter(cells)
            occupied = Counter(domain for domain, _bin in set(cells))
            weights = [1.0 / (counts[cell] * occupied[cell[0]]) for cell in cells]
        weights = torch.tensor(weights, dtype=torch.double)
        super().__init__(weights, num_samples or len(tempos), replacement, generator=generator)


def stretch_audio_and_tempo(audio, tempo, rate, *, min_bpm=30.0, max_bpm=285.0):
    """Time-stretch audio and relabel its tempo, rejecting out-of-range labels."""
    if tempo <= 0 or rate <= 0:
        raise ValueError("tempo and rate must be positive")
    relabeled = float(tempo) * float(rate)
    if not min_bpm <= relabeled <= max_bpm:
        raise ValueError("augmented tempo is outside the model range")
    stretched = librosa.effects.time_stretch(np.asarray(audio, dtype=np.float32), rate=float(rate))
    return stretched, relabeled