import pytest

from deeprhythm.bench.taxonomy import analyze, expected_calibration_error, metrical_relation


@pytest.mark.parametrize(
    ("prediction", "reference", "relation"),
    [(120, 120, "correct"), (60, 120, "half"), (240, 120, "double"), (40, 120, "third"),
     (180, 60, "triple"), (83, 120, "other")],
)
def test_metrical_relation(prediction, reference, relation):
    assert metrical_relation(prediction, reference) == relation


def test_analyze_joins_paths_and_reports_calibration(tmp_path):
    audio = tmp_path / "track.wav"
    result = analyze(
        [{"audio_path": str(audio), "tempo": 120, "genre": "dance"}],
        [{"filename": str(audio), "bpm": 60, "confidence": 0.8}],
    )
    assert result["counts"]["relation"] == {"half": 1}
    assert result["counts"]["genre"] == {"dance:half": 1}
    assert result["calibration"]["ece"] == pytest.approx(0.8)


def test_expected_calibration_error_validates_inputs():
    with pytest.raises(ValueError, match="between"):
        expected_calibration_error([1.1], [True])