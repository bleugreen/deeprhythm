"""Metrical-level correction baselines and lightweight learned corrector."""

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
import torch

from deeprhythm.model.results import METRICAL_CANDIDATES

FACTORS = tuple(factor for _name, factor in METRICAL_CANDIDATES)
FACTOR_NAMES = tuple(name for name, _factor in METRICAL_CANDIDATES)


def identity(detail: Dict) -> float:
    return float(detail["bpm"])


def global_threshold(detail: Dict, threshold: float) -> float:
    bpm = identity(detail)
    return bpm * 2.0 if bpm < threshold else bpm


def top_k_ratio(detail: Dict, ratio: float) -> float:
    """Double when aggregate probability mass at 2x exceeds a ratio of same-level mass."""
    masses = detail["candidate_masses"]
    same = float(masses["same"]["probability_mass"])
    double = float(masses["double"]["probability_mass"])
    return identity(detail) * (2.0 if double >= ratio * max(same, 1e-12) else 1.0)


def clip_disagreement(detail: Dict, threshold: float) -> float:
    """Double when enough clip argmaxes support the faster metrical level."""
    bpm = identity(detail)
    clips = detail.get("clips", ())
    if not clips:
        return bpm
    faster = sum(float(clip["bpm"]) >= 1.5 * bpm for clip in clips) / len(clips)
    return bpm * (2.0 if faster >= threshold else 1.0)


def _lag_energy(envelope: np.ndarray, lag: float) -> float:
    lag = int(round(lag))
    if lag <= 0 or lag >= envelope.size:
        return 0.0
    left, right = envelope[:-lag], envelope[lag:]
    scale = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / scale) if scale > 0 else 0.0


def onset_periodicity_features(
    waveform, sample_rate: int, predicted_bpm: float, n_fft: int = 1024, hop_length: int = 256
) -> np.ndarray:
    """Extract dependency-free onset autocorrelation, spectral balance, and onset-rate features."""
    if sample_rate <= 0 or predicted_bpm <= 0:
        raise ValueError("sample_rate and predicted_bpm must be positive")
    audio = torch.as_tensor(waveform, dtype=torch.float32).flatten()
    if audio.numel() < n_fft:
        audio = torch.nn.functional.pad(audio, (0, n_fft - audio.numel()))
    window = torch.hann_window(n_fft, dtype=audio.dtype, device=audio.device)
    spectrum = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    magnitude = spectrum.abs()
    flux = torch.relu(magnitude[:, 1:] - magnitude[:, :-1]).mean(dim=0)
    envelope = flux.cpu().numpy().astype(float)
    envelope -= envelope.mean() if envelope.size else 0.0
    frames_per_minute = 60.0 * sample_rate / hop_length
    periodicity = [_lag_energy(envelope, frames_per_minute / (predicted_bpm * factor)) for factor in FACTORS]

    mean_spectrum = magnitude.mean(dim=1)
    split = max(1, mean_spectrum.numel() // 4)
    low = float(mean_spectrum[:split].mean())
    high = float(mean_spectrum[split:].mean()) if split < mean_spectrum.numel() else 0.0
    spectral_balance = np.log1p(high) - np.log1p(low)
    positive = np.maximum(envelope, 0)
    onset_count = (positive > positive.mean() + positive.std()).sum()
    onset_rate = float(onset_count * sample_rate / hop_length / max(audio.numel(), 1))
    return np.asarray(periodicity + [spectral_balance, onset_rate, float(positive.mean()), float(positive.std())])


def target_factor(prediction: float, reference: float) -> int:
    """Return the candidate factor whose corrected tempo is closest to the reference."""
    errors = [abs(prediction * factor / reference - 1.0) for factor in FACTORS]
    return int(np.argmin(errors))


@dataclass
class FactorClassifier:
    """Small multinomial logistic-regression model implemented with NumPy."""

    weights: np.ndarray = None
    mean: np.ndarray = None
    scale: np.ndarray = None

    def fit(self, features, labels, learning_rate=0.1, epochs=500, l2=1e-3):
        x = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=int)
        if x.ndim != 2 or y.shape != (x.shape[0],) or not x.shape[0]:
            raise ValueError("features and labels must be non-empty aligned arrays")
        if np.any((y < 0) | (y >= len(FACTORS))):
            raise ValueError("labels contain an unknown factor")
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        z = np.column_stack([(x - self.mean) / self.scale, np.ones(x.shape[0])])
        self.weights = np.zeros((z.shape[1], len(FACTORS)))
        targets = np.eye(len(FACTORS))[y]
        for _ in range(epochs):
            logits = z @ self.weights
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            gradient = z.T @ (probabilities - targets) / z.shape[0]
            gradient[:-1] += l2 * self.weights[:-1]
            self.weights -= learning_rate * gradient
        return self

    def predict_proba(self, features):
        if self.weights is None:
            raise ValueError("classifier has not been fitted")
        x = np.atleast_2d(np.asarray(features, dtype=float))
        z = np.column_stack([(x - self.mean) / self.scale, np.ones(x.shape[0])])
        logits = z @ self.weights
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def correct(self, detail: Dict, features) -> float:
        return identity(detail) * FACTORS[int(self.predict_proba(features).argmax(axis=1)[0])]

    def to_dict(self):
        if self.weights is None:
            raise ValueError("classifier has not been fitted")
        return {"weights": self.weights.tolist(), "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "factors": list(FACTORS)}

    @classmethod
    def from_dict(cls, row):
        if tuple(row["factors"]) != FACTORS:
            raise ValueError("model factor taxonomy does not match this version")
        return cls(np.asarray(row["weights"]), np.asarray(row["mean"]), np.asarray(row["scale"]))


def select_parameter(details: Sequence[Dict], references: Sequence[float], method: str, candidates: Iterable[float]):
    """Select one scalar baseline parameter exclusively from validation rows."""
    from deeprhythm.bench.tempo import tempo_accuracy

    methods = {"global_threshold": global_threshold, "top_k_ratio": top_k_ratio,
               "clip_disagreement": clip_disagreement}
    if method not in methods:
        raise ValueError(f"unknown selectable method: {method}")
    if len(details) != len(references) or not details:
        raise ValueError("details and references must be non-empty and aligned")
    scored = []
    for value in candidates:
        predictions = [methods[method](detail, value) for detail in details]
        metrics = tempo_accuracy(predictions, references, tolerance=0.04)
        scored.append((metrics["acc1"], metrics["acc2"], -float(value), float(value)))
    return max(scored)[-1]