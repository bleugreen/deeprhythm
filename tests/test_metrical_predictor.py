import torch

from deeprhythm.model.metrical import MetricalFusion, temporal_metrical_features
from deeprhythm.model.metrical_predictor import MetricalDeepRhythmPredictor, _trim_edge_clips


def test_metrical_features_match_fusion_contract():
    temporal = torch.randn(1, 20, 36)
    phase = torch.randn(1, 20, 2)
    base = torch.softmax(torch.randn(120), dim=0)
    level = torch.softmax(torch.randn(4), dim=0)

    features = temporal_metrical_features(temporal, phase, base, level)

    assert features.shape == (162,)
    expected_confidence = phase[0].norm(p=-1)
    assert torch.equal(features[[146, 149, 152, 155]], expected_confidence.repeat(4))
    assert MetricalFusion()(features[None]).shape == (1, 4)


def test_packaged_v08_bundle_loads_on_cpu():
    predictor = MetricalDeepRhythmPredictor(device="cpu")

    assert predictor.alpha == 0.4
    assert predictor.hierarchical.training is False
    assert predictor.temporal.training is False


def test_v08_track_aggregation_matches_training_edge_trim():
    clips = torch.arange(15)

    assert torch.equal(_trim_edge_clips(clips), clips[1:-1])
    assert torch.equal(_trim_edge_clips(clips[:5]), clips[:5])