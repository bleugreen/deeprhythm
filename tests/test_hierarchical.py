import pytest
import torch

from deeprhythm.model.hierarchical import (
    HierarchicalDeepRhythmModel,
    hierarchy_to_tempo,
    tempo_to_hierarchy,
)


@pytest.mark.parametrize("tempo", [30.0, 59.5, 60.0, 87.5, 119.5, 120.0, 175.0, 239.0, 240.0, 284.0])
def test_hierarchical_tempo_round_trip(tempo):
    base, level = tempo_to_hierarchy(torch.tensor([tempo]))
    assert hierarchy_to_tempo(base, level).item() == pytest.approx(tempo, abs=0.5)


def test_hierarchical_model_has_independent_outputs():
    model = HierarchicalDeepRhythmModel()
    base, level = model(torch.zeros(2, 6, 240, 8))
    assert base.shape == (2, 120)
    assert level.shape == (2, 4)