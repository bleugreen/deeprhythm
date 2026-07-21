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

def test_batch_instrumentation_does_not_change_legacy_prediction(monkeypatch, tmp_path):
    import json

    import deeprhythm.batch_infer as batch_infer

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, inputs):
            logits = torch.full((len(inputs), 256), -10.0)
            logits[:, 90] = 10.0
            return logits

    monkeypatch.setattr(batch_infer, "compute_hcqm", lambda audio, *_specs: audio)
    audio = torch.zeros((2, 6, 240, 8))
    plain = tmp_path / "plain.jsonl"
    instrumented = tmp_path / "instrumented.jsonl"
    details = tmp_path / "details.jsonl"
    metadata = [("track.wav", 2, 0)]

    batch_infer.process_and_save(audio, metadata, (None, None, None), Model(), plain, conf=True, quiet=True)
    batch_infer.process_and_save(
        audio, metadata, (None, None, None), Model(), instrumented, conf=True, quiet=True, details_path=details
    )

    assert plain.read_bytes() == instrumented.read_bytes()
    assert json.loads(details.read_text())["filename"] == "track.wav"
