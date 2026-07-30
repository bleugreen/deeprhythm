import numpy as np
import pytest
import torch

from deeprhythm.bench.correction import evaluate_rows, join_rows, run_experiment
from deeprhythm.correction import (
    FACTORS,
    FactorClassifier,
    clip_disagreement,
    global_threshold,
    onset_periodicity_features,
    select_parameter,
    top_k_ratio,
)


def detail(bpm, same=0.8, double=0.1, clips=None):
    return {
        "filename": "unused.wav",
        "bpm": bpm,
        "confidence": same,
        "candidate_masses": {
            "same": {"probability_mass": same},
            "double": {"probability_mass": double},
        },
        "clips": [{"bpm": value} for value in (clips or [bpm])],
    }


def test_floor_baselines_are_deterministic():
    row = detail(60, same=0.2, double=0.3, clips=[60, 120, 120])
    assert global_threshold(row, 90) == 120
    assert top_k_ratio(row, 1.25) == 120
    assert clip_disagreement(row, 0.5) == 120
    assert clip_disagreement(row, 0.8) == 60


def test_validation_selects_best_global_threshold():
    rows = [detail(60), detail(130)]
    assert select_parameter(rows, [120, 130], "global_threshold", [50, 90, 150]) == 90


def test_onset_periodicity_features_are_finite_and_fixed_width():
    sample_rate = 8000
    time = torch.arange(sample_rate * 2) / sample_rate
    audio = torch.sin(2 * torch.pi * 2 * time)
    features = onset_periodicity_features(audio, sample_rate, 120)
    assert features.shape == (len(FACTORS) + 4,)
    assert np.isfinite(features).all()


def test_factor_classifier_learns_and_round_trips():
    features = np.vstack([np.full(3, -2.0), np.full(3, -1.0), np.full(3, 1.0), np.full(3, 2.0)])
    labels = [1, 1, 0, 0]
    model = FactorClassifier().fit(features, labels, epochs=800)
    assert model.predict_proba(features).argmax(axis=1).tolist() == labels
    restored = FactorClassifier.from_dict(model.to_dict())
    assert restored.predict_proba(features) == pytest.approx(model.predict_proba(features))


def test_join_rows_rejects_held_out_fold(tmp_path):
    path = tmp_path / "audio.wav"
    manifest = [{"audio_path": str(path), "tempo": 120, "fold": "test"}]
    details = [{**detail(60), "filename": str(path)}]
    with pytest.raises(ValueError, match="held-out test"):
        join_rows(manifest, details)


def test_experiment_reports_full_evaluation():
    train = [
        {"tempo": 120, "genre": "dance", "detail": detail(60), "features": [1, 0]},
        {"tempo": 80, "genre": "slow", "detail": detail(80), "features": [0, 1]},
    ]
    val = [
        {"tempo": 120, "genre": "dance", "detail": detail(60), "features": [1, 0]},
        {"tempo": 80, "genre": "slow", "detail": detail(80), "features": [0, 1]},
    ]
    result = run_experiment(train, val)
    assert {"identity", "global_threshold", "top_k_ratio", "clip_disagreement", "learned"} <= result.keys()
    learned = result["learned"]
    assert {"metrics_2pct", "metrics_4pct", "taxonomy", "slices", "calibration", "latency"} <= learned.keys()
    assert learned["metrics_4pct"]["acc1"] == 1.0


def test_evaluation_uses_canonical_acc1_acc2_relations():
    rows = [{"tempo": 120, "genre": "x"}, {"tempo": 100, "genre": "x"}]
    report = evaluate_rows(rows, [60, 100])
    assert report["metrics_2pct"] == pytest.approx({"acc1": 0.5, "acc2": 1.0})
    assert report["taxonomy"] == {"correct": 1, "half": 1}