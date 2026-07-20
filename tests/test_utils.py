import hashlib
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import requests

import deeprhythm.utils as utils
from deeprhythm.utils import (
    AudioLoadError,
    AudioTooShortError,
    bpm_to_class,
    class_to_bpm,
    get_device,
    get_weights,
    load_weights,
    split_audio,
)


class StreamedResponse:
    def __init__(self, chunks=(), error=None):
        self.chunks = chunks
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        if self.error:
            raise self.error

    def iter_content(self, chunk_size):
        assert chunk_size == utils.MODEL_DOWNLOAD_CHUNK_SIZE
        yield from self.chunks


@pytest.fixture
def model_artifact(monkeypatch, tmp_path):
    payload = b"verified model bytes"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(utils, "MODEL_WEIGHTS_SIZE", len(payload))
    monkeypatch.setattr(utils, "MODEL_WEIGHTS_SHA256", hashlib.sha256(payload).hexdigest())
    cache_dir = tmp_path / ".local" / "share" / "deeprhythm"
    destination = cache_dir / utils.MODEL_WEIGHTS_FILENAME
    return payload, cache_dir, destination


def test_get_weights_accepts_valid_cache_without_network(model_artifact, monkeypatch):
    payload, cache_dir, destination = model_artifact
    cache_dir.mkdir(parents=True)
    destination.write_bytes(payload)
    request = Mock(side_effect=AssertionError("network must not be used"))
    monkeypatch.setattr(utils.requests, "get", request)

    assert get_weights(quiet=True) == str(destination)
    request.assert_not_called()


def test_get_weights_downloads_from_pinned_url_and_atomically_replaces_corrupt_cache(
    model_artifact, monkeypatch
):
    payload, cache_dir, destination = model_artifact
    cache_dir.mkdir(parents=True)
    destination.write_bytes(b"corrupt")
    request = Mock(return_value=StreamedResponse([payload[:5], payload[5:]]))
    monkeypatch.setattr(utils.requests, "get", request)
    real_replace = utils.os.replace
    replace = Mock(side_effect=real_replace)
    monkeypatch.setattr(utils.os, "replace", replace)

    assert get_weights(quiet=True) == str(destination)
    request.assert_called_once_with(
        utils.MODEL_WEIGHTS_URL,
        stream=True,
        timeout=(utils.MODEL_DOWNLOAD_CONNECT_TIMEOUT, utils.MODEL_DOWNLOAD_READ_TIMEOUT),
    )
    source, target = replace.call_args.args
    assert target == str(destination)
    assert destination.read_bytes() == payload
    assert not Path(source).exists()
    assert list(cache_dir.glob("*.tmp")) == []


@pytest.mark.parametrize(
    ("response", "expected_message"),
    [
        (StreamedResponse(error=requests.HTTPError("500 Server Error")), "Failed to download"),
        (StreamedResponse([b"partial"], requests.Timeout("timed out")), "Failed to download"),
    ],
)
def test_get_weights_network_failures_do_not_publish(
    response, expected_message, model_artifact, monkeypatch
):
    _, cache_dir, destination = model_artifact
    monkeypatch.setattr(utils.requests, "get", Mock(return_value=response))

    with pytest.raises(utils.ModelWeightsError, match=expected_message):
        get_weights(quiet=True)

    assert not destination.exists()
    assert list(cache_dir.glob("*.tmp")) == []


def test_get_weights_interrupted_stream_preserves_existing_cache(model_artifact, monkeypatch):
    payload, cache_dir, destination = model_artifact
    cache_dir.mkdir(parents=True)
    destination.write_bytes(b"existing corrupt bytes")

    class InterruptedResponse(StreamedResponse):
        def iter_content(self, chunk_size):
            yield payload[:5]
            raise requests.ConnectionError("stream interrupted")

    monkeypatch.setattr(utils.requests, "get", Mock(return_value=InterruptedResponse()))

    with pytest.raises(utils.ModelWeightsError, match="stream interrupted"):
        get_weights(quiet=True)

    assert destination.read_bytes() == b"existing corrupt bytes"
    assert list(cache_dir.glob("*.tmp")) == []


@pytest.mark.parametrize(
    ("download", "expected_message"),
    [
        (b"verified model bytes!", "exceed the expected size"),
        (b"short", "have size"),
        (b"different model bytes", "SHA-256"),
    ],
)
def test_get_weights_rejects_invalid_downloads(download, expected_message, model_artifact, monkeypatch):
    _, cache_dir, destination = model_artifact
    monkeypatch.setattr(utils.requests, "get", Mock(return_value=StreamedResponse([download])))

    with pytest.raises(utils.ModelWeightsError, match=expected_message):
        get_weights(quiet=True)

    assert not destination.exists()
    assert list(cache_dir.glob("*.tmp")) == []


def test_load_weights_uses_safe_torch_deserialization(monkeypatch):
    state_dict = {"layer": "weights"}
    monkeypatch.setattr(utils, "get_weights", Mock(return_value="/verified/model.pth"))
    torch_load = Mock(return_value=state_dict)
    monkeypatch.setattr(utils.torch, "load", torch_load)

    assert load_weights("cpu", quiet=True) == state_dict
    torch_load.assert_called_once_with(
        "/verified/model.pth", map_location="cpu", weights_only=True
    )

# ---------------------------------------------------------------------------
# BPM <-> class conversion
# ---------------------------------------------------------------------------

def test_bpm_class_roundtrip():
    """bpm_to_class and class_to_bpm should roundtrip within one class width."""
    class_width = (286 - 30) / 256
    for bpm in [30, 60, 90, 120, 150, 200, 285]:
        cls = bpm_to_class(bpm)
        recovered = class_to_bpm(cls)
        assert abs(recovered - bpm) <= class_width, f"Roundtrip failed for {bpm}: got {recovered}"


def test_bpm_to_class_clamps():
    """Values outside [30, 286] should clamp to valid class range."""
    assert bpm_to_class(0) == 0
    assert bpm_to_class(500) == 255


def test_bpm_to_class_monotonic():
    """Increasing BPM should produce non-decreasing class indices."""
    prev = -1
    for bpm in range(30, 287):
        cls = bpm_to_class(bpm)
        assert cls >= prev, f"Non-monotonic at BPM {bpm}: class {cls} < {prev}"
        prev = cls


def test_class_to_bpm_monotonic():
    """Increasing class index should produce increasing BPM."""
    prev = -1.0
    for cls in range(256):
        bpm = class_to_bpm(cls)
        assert bpm > prev, f"Non-monotonic at class {cls}: BPM {bpm} <= {prev}"
        prev = bpm


def test_bpm_class_boundaries():
    """Test exact boundary values."""
    assert bpm_to_class(30.0) == 0
    assert bpm_to_class(285.0) in (254, 255)


def test_bpm_class_full_range_coverage():
    """Every class 0-255 should map to a BPM in [30, 286]."""
    for cls in range(256):
        bpm = class_to_bpm(cls)
        assert 30 <= bpm <= 286, f"Class {cls} maps to out-of-range BPM {bpm}"


def test_bpm_to_class_custom_range():
    """Custom min/max/num_classes should work."""
    assert bpm_to_class(60, min_bpm=60, max_bpm=180, num_classes=120) == 0
    assert bpm_to_class(179, min_bpm=60, max_bpm=180, num_classes=120) == 119


# ---------------------------------------------------------------------------
# split_audio
# ---------------------------------------------------------------------------

def test_split_audio_basic():
    """split_audio should produce correct number of clips."""
    sr = 22050
    num_clips = 3
    audio = np.random.randn(sr * 8 * num_clips + 1000).astype(np.float32)
    clips = split_audio(audio, sr, clip_length=8)
    assert clips.shape == (num_clips, sr * 8)


def test_split_audio_too_short():
    """split_audio should raise AudioTooShortError for short audio."""
    import pytest
    sr = 22050
    audio = np.zeros(100, dtype=np.float32)
    with pytest.raises(AudioTooShortError):
        split_audio(audio, sr)


def test_split_audio_share_mem():
    """split_audio with share_mem=True returns a shared-memory tensor."""
    sr = 22050
    audio = np.random.randn(sr * 8).astype(np.float32)
    clips = split_audio(audio, sr, share_mem=True)
    assert clips.is_shared()


def test_split_audio_exact_boundary():
    """Audio length exactly N * clip_samples -> N clips."""
    sr = 22050
    for n in (1, 2, 5):
        audio = np.zeros(sr * 8 * n, dtype=np.float32)
        clips = split_audio(audio, sr)
        assert clips.shape[0] == n


def test_split_audio_just_over():
    """N * clip_samples + 1 sample -> still N clips (remainder dropped)."""
    sr = 22050
    n = 3
    audio = np.zeros(sr * 8 * n + 1, dtype=np.float32)
    clips = split_audio(audio, sr)
    assert clips.shape[0] == n


def test_split_audio_single_clip():
    """Exactly 8 seconds -> 1 clip."""
    sr = 22050
    audio = np.ones(sr * 8, dtype=np.float32)
    clips = split_audio(audio, sr)
    assert clips.shape == (1, sr * 8)


def test_split_audio_preserves_values():
    """Clip content should match original audio slices."""
    sr = 22050
    clip_len = 8
    clip_samples = sr * clip_len
    audio = np.arange(clip_samples * 2, dtype=np.float32)
    clips = split_audio(audio, sr, clip_length=clip_len)
    np.testing.assert_array_equal(clips[0].numpy(), audio[:clip_samples])
    np.testing.assert_array_equal(clips[1].numpy(), audio[clip_samples:2 * clip_samples])


def test_split_audio_custom_clip_length():
    """Non-default clip length (4 seconds) should work."""
    sr = 22050
    audio = np.zeros(sr * 12, dtype=np.float32)
    clips = split_audio(audio, sr, clip_length=4)
    assert clips.shape == (3, sr * 4)


# ---------------------------------------------------------------------------
# get_device
# ---------------------------------------------------------------------------

def test_get_device_returns_valid():
    """get_device should return one of the known device strings."""
    device = get_device()
    assert device in ('cuda', 'mps', 'cpu')


# ---------------------------------------------------------------------------
# Error classes
# ---------------------------------------------------------------------------

def test_audio_too_short_is_value_error():
    """AudioTooShortError should be a ValueError."""
    assert issubclass(AudioTooShortError, ValueError)
    with __import__('pytest').raises(ValueError):
        raise AudioTooShortError("test")


def test_audio_load_error_is_io_error():
    """AudioLoadError should be an IOError."""
    assert issubclass(AudioLoadError, IOError)
    with __import__('pytest').raises(IOError):
        raise AudioLoadError("test")
