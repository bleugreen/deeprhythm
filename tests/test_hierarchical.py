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


def test_residual_level_head_preserves_existing_outputs_at_initialization():
    torch.manual_seed(7)
    original = HierarchicalDeepRhythmModel().eval()
    residual = HierarchicalDeepRhythmModel(residual_level=True).eval()
    residual.load_state_dict(original.state_dict(), strict=False)
    inputs = torch.randn(2, 6, 240, 8)

    expected_base, expected_level = original(inputs)
    actual_base, actual_level = residual(inputs)

    assert torch.equal(actual_base, expected_base)
    assert torch.equal(actual_level, expected_level)


def test_residual_level_head_can_change_level_without_changing_base():
    model = HierarchicalDeepRhythmModel(residual_level=True).eval()
    inputs = torch.randn(2, 6, 240, 8)
    expected_base, expected_level = model(inputs)
    with torch.no_grad():
        model.level_residual.network[-1].bias[2] = 1.0

    actual_base, actual_level = model(inputs)

    assert torch.equal(actual_base, expected_base)
    assert torch.equal(actual_level[:, :2], expected_level[:, :2])
    assert torch.equal(actual_level[:, 3], expected_level[:, 3])
    assert torch.equal(actual_level[:, 2], expected_level[:, 2] + 1.0)