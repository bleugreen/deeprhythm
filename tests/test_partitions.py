import json

import pytest

from deeprhythm.partitions import _assign_ballroom, _ballroom_group, validate_checksums, validate_partitions


def row(path, group, fold, fingerprint, dataset="example"):
    return {"dataset": dataset, "audio_path": path, "md5": "0" * 32, "tempo": 120,
            "group_key": group, "fold": fold, "fingerprint": fingerprint, "provenance": "test"}


def test_validation_accepts_disjoint_rows():
    assert validate_partitions([row("a.wav", "artist:a", "train", "fp:a"), row("b.wav", "artist:b", "test", "fp:b")])


@pytest.mark.parametrize("field", ["group", "fingerprint"])
def test_validation_rejects_leakage(field):
    first = row("a.wav", "artist:a", "train", "fp:a")
    second = row("b.wav", "artist:a" if field == "group" else "artist:b", "test", "fp:a")
    with pytest.raises(ValueError, match="spans"):
        validate_partitions([first, second])


def test_validation_rejects_cross_dataset_fingerprint_leakage():
    with pytest.raises(ValueError, match="fingerprint.*spans"):
        validate_partitions([
            row("external.wav", "artist:a", "train", "fp:same", dataset="external"),
            row("test.wav", "artist:b", "test", "fp:same", dataset="ballroom"),
        ])


def test_ballroom_group_uses_only_filename_metadata():
    assert _ballroom_group("Albums-Latin_Jam2-03") == "album:latin_jam2"
    assert _ballroom_group("Media-104705") == "media-prefix:1047"


def test_ballroom_assignment_is_deterministic_and_grouped():
    groups = {"album:a": [("waltz", 4)] * 3, "album:b": [("waltz", 4)] * 2, "album:c": [("tango", 6)]}
    assert _assign_ballroom(groups) == _assign_ballroom(dict(reversed(list(groups.items()))))


def test_committed_manifests_are_valid_and_consistent():
    from pathlib import Path
    root = Path(__file__).parents[1] / "data" / "splits"
    rows = []
    for dataset in ("giantsteps", "gtzan", "ballroom"):
        dataset_rows = []
        for fold in ("train", "val", "test"):
            path = root / dataset / f"{fold}.jsonl"
            loaded = [json.loads(line) for line in path.read_text().splitlines()]
            assert loaded and all(row_["fold"] == fold for row_ in loaded)
            dataset_rows.extend(loaded)
        validate_partitions(dataset_rows)
        rows.extend(dataset_rows)
    assert {row_["dataset"] for row_ in rows} == {"giantsteps", "gtzan", "ballroom"}
    validate_partitions(rows)
    validate_checksums(root)