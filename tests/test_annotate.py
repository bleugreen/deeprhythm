import json

import numpy as np
import pytest

from deeprhythm.annotate import AnnotationStore, active_excerpt, metrical_grids


def test_metrical_grids_preserve_phase_and_change_rate():
    grids = metrical_grids([0.0, 1.0, 2.0], 3.0)
    assert grids["half"].tolist() == [0.0, 2.0]
    assert grids["current"].tolist() == [0.0, 1.0, 2.0]
    assert grids["double"].tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]


def test_active_excerpt_selects_onset_region():
    audio = np.zeros(4000, dtype=np.float32)
    audio[3000:3100] = 1
    offset, excerpt = active_excerpt(audio, 1000, duration=1)
    assert offset >= 2
    assert excerpt.max() == 1


def test_annotation_store_validates_and_appends(tmp_path):
    (tmp_path / "manifest.jsonl").write_text(json.dumps({"id": "track"}) + "\n")
    store = AnnotationStore(tmp_path)
    store.append("track", "correct")
    assert store.completed_ids() == {"track"}
    with pytest.raises(ValueError, match="unknown label"):
        store.append("track", "wrong")
    with pytest.raises(ValueError, match="unknown manifest"):
        store.append("missing", "correct")