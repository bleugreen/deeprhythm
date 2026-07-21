import pytest
import torch

from deeprhythm.model.results import shape_prediction


def test_shape_prediction_aggregates_clips_deterministically():
    probabilities = torch.zeros((2, 256))
    probabilities[0, 90] = 0.8
    probabilities[0, 91] = 0.2
    probabilities[1, 90] = 0.4
    probabilities[1, 92] = 0.6

    details = shape_prediction(probabilities, top_k=2)

    assert details["predicted_class"] == 90
    assert details["num_clips"] == 2
    assert details["confidence"] == pytest.approx(0.6)
    assert [clip["class"] for clip in details["clips"]] == [90, 92]
    assert [peak["class"] for peak in details["aggregate_top_k"]] == [90, 92]
    assert set(details["candidate_masses"]) == {"same", "double", "half", "triple", "third"}


def test_shape_prediction_rejects_invalid_top_k():
    with pytest.raises(ValueError, match="positive"):
        shape_prediction(torch.ones((1, 256)), top_k=0)