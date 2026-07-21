import json

import pytest

from deeprhythm.bench.tempo import evaluate, load_manifest, tempo_accuracy


def test_tempo_accuracy_distinguishes_exact_and_metrical_matches():
    scores = tempo_accuracy([120, 60, 90], [120, 120, 100], tolerance=0.02)

    assert scores == pytest.approx({"acc1": 1 / 3, "acc2": 2 / 3})


def test_tempo_accuracy_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="equally sized"):
        tempo_accuracy([120], [120, 130])
    with pytest.raises(ValueError, match="positive"):
        tempo_accuracy([0], [120])
    with pytest.raises(ValueError, match="tolerance"):
        tempo_accuracy([120], [120], tolerance=0)


def test_load_manifest_resolves_paths_and_rejects_duplicates(tmp_path):
    audio_path = tmp_path / "track.wav"
    manifest = tmp_path / "manifest.jsonl"
    row = {"audio_path": str(audio_path), "tempo": 120}
    manifest.write_text(json.dumps(row) + "\n")

    paths, tempos = load_manifest(manifest)
    assert paths == [str(audio_path.resolve())]
    assert tempos == [120.0]

    manifest.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="unique"):
        load_manifest(manifest)

def test_evaluate_reports_total_and_per_track_timing():
    results = evaluate([120, 60], [120, 120], elapsed_seconds=0.05)

    assert results["tracks"] == 2
    assert results["elapsed_seconds"] == 0.05
    assert results["batched_milliseconds_per_track"] == 25.0
    assert results["tolerance"] == 0.04
    assert results["acc1"] == 0.5
    assert results["acc2"] == 1.0
