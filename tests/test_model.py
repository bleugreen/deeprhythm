import os
import tempfile

import numpy as np
import pytest
import soundfile as sf
import torch

from deeprhythm.model.frame_cnn import DeepRhythmModel

# ---------------------------------------------------------------------------
# Model Architecture (no weights needed)
# ---------------------------------------------------------------------------

def test_model_forward_shape():
    """Random input (4, 6, 240, 8) -> output (4, 256)."""
    model = DeepRhythmModel()
    model.eval()
    x = torch.randn(4, 6, 240, 8)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 256)


def test_model_output_not_uniform():
    """Output logits shouldn't all be the same value."""
    model = DeepRhythmModel()
    model.eval()
    x = torch.randn(1, 6, 240, 8)
    with torch.no_grad():
        out = model(x)
    assert out.std().item() > 0, "All output logits are identical"


def test_model_single_sample():
    """Batch size 1 should work: (1, 6, 240, 8) -> (1, 256)."""
    model = DeepRhythmModel()
    model.eval()
    x = torch.randn(1, 6, 240, 8)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 256)


def test_model_eval_deterministic():
    """In eval mode, same input should produce same output (no dropout stochasticity)."""
    model = DeepRhythmModel()
    model.eval()
    x = torch.randn(2, 6, 240, 8)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2)


def test_model_num_classes_custom():
    """DeepRhythmModel(num_classes=128) -> output dim is 128."""
    model = DeepRhythmModel(num_classes=128)
    model.eval()
    x = torch.randn(1, 6, 240, 8)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 128)


def test_model_parameter_count():
    """Total trainable params should be in expected ballpark (~485K)."""
    model = DeepRhythmModel()
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Allow a reasonable range around expected count
    assert 1_000_000 < total < 2_000_000, f"Parameter count {total} outside expected range"


# ---------------------------------------------------------------------------
# Predictor (requires model weights — slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_predictor_instantiation():
    """DeepRhythmPredictor should load model and create kernels."""
    from deeprhythm.model.predictor import DeepRhythmPredictor
    predictor = DeepRhythmPredictor(device='cpu', quiet=True)
    assert predictor.model is not None
    assert predictor.specs is not None


@pytest.mark.slow
def test_predict_sine_wave():
    """Predicting on a synthetic sine wave should return a float in valid BPM range."""
    from deeprhythm.model.predictor import DeepRhythmPredictor

    sr = 22050
    duration = 16
    t = np.linspace(0, duration, sr * duration, dtype=np.float32)
    audio = np.sin(2 * np.pi * 2.0 * t).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, audio, sr)
        tmp_path = f.name

    try:
        predictor = DeepRhythmPredictor(device='cpu', quiet=True)
        result = predictor.predict(tmp_path)
        assert isinstance(result, float)
        assert 30 <= result <= 286
    finally:
        os.unlink(tmp_path)
