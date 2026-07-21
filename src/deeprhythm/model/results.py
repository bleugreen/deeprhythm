"""Canonical shaping of classifier outputs for inference APIs and benchmarks."""

from typing import Dict

import torch
from torch import Tensor

from deeprhythm.utils import class_to_bpm

METRICAL_CANDIDATES = (
    ("same", 1.0),
    ("double", 2.0),
    ("half", 0.5),
    ("triple", 3.0),
    ("third", 1.0 / 3.0),
)


def _top_k(probabilities: Tensor, top_k: int):
    count = min(top_k, probabilities.shape[-1])
    values, classes = torch.topk(probabilities, count, dim=-1)
    if probabilities.ndim == 1:
        values = values.unsqueeze(0)
        classes = classes.unsqueeze(0)
    return [
        [
            {"class": int(class_index), "bpm": class_to_bpm(int(class_index)), "probability": float(probability)}
            for class_index, probability in zip(row_classes.tolist(), row_values.tolist())
        ]
        for row_classes, row_values in zip(classes.cpu(), values.cpu())
    ]


def shape_prediction(probabilities: Tensor, top_k: int = 5, tolerance: float = 0.04) -> Dict:
    """Aggregate per-clip probabilities into one auditable prediction."""
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise ValueError("probabilities must contain at least one clip and have shape (clips, classes)")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    probabilities = probabilities.detach().cpu()
    mean_probabilities = probabilities.mean(dim=0)
    confidence, predicted_class = mean_probabilities.max(dim=0)
    predicted_class_value = int(predicted_class)
    predicted_bpm = class_to_bpm(predicted_class_value)
    clip_classes = probabilities.argmax(dim=1)
    clip_top = _top_k(probabilities, top_k)
    aggregate_top = _top_k(mean_probabilities, top_k)[0]

    class_bpms = torch.tensor(
        [class_to_bpm(class_index) for class_index in range(mean_probabilities.shape[0])],
        dtype=mean_probabilities.dtype,
    )
    candidate_masses = {}
    for name, factor in METRICAL_CANDIDATES:
        candidate_bpm = predicted_bpm * factor
        mask = torch.abs(class_bpms / candidate_bpm - 1.0) <= tolerance
        candidate_masses[name] = {
            "factor": factor,
            "bpm": candidate_bpm,
            "probability_mass": float(mean_probabilities[mask].sum()),
        }

    clips = []
    for class_index, top in zip(clip_classes.tolist(), clip_top):
        clips.append({"class": class_index, "bpm": class_to_bpm(class_index), "top_k": top})
    return {
        "bpm": predicted_bpm,
        "confidence": float(confidence),
        "predicted_class": predicted_class_value,
        "num_clips": probabilities.shape[0],
        "clips": clips,
        "aggregate_top_k": aggregate_top,
        "candidate_masses": candidate_masses,
    }