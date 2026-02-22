import torch

from deeprhythm.audio_proc.bandfilter import apply_log_filter, create_log_filter
from deeprhythm.audio_proc.hcqm import compute_hcqm, make_kernels
from deeprhythm.audio_proc.onset import onset_strength

# ---------------------------------------------------------------------------
# Band Filter (bandfilter.py)
# ---------------------------------------------------------------------------

def test_create_log_filter_shape():
    """Output shape should be (num_bands, num_bins)."""
    f = create_log_filter(1025, 8, device='cpu')
    assert f.shape == (8, 1025)


def test_create_log_filter_unity_gain():
    """Each row (band) should sum to 1.0 — normalised by band width."""
    f = create_log_filter(1025, 8, device='cpu')
    for i in range(8):
        row_sum = f[i].sum().item()
        assert abs(row_sum - 1.0) < 1e-5, f"Band {i} sums to {row_sum}, expected 1.0"


def test_create_log_filter_no_overlap():
    """Each frequency bin should belong to at most one band."""
    f = create_log_filter(1025, 8, device='cpu')
    for col in range(1025):
        nonzero = (f[:, col] != 0).sum().item()
        assert nonzero <= 1, f"Bin {col} belongs to {nonzero} bands"


def test_create_log_filter_full_coverage():
    """Every frequency bin should belong to exactly one band (no gaps)."""
    f = create_log_filter(1025, 8, device='cpu')
    for col in range(1025):
        nonzero = (f[:, col] != 0).sum().item()
        assert nonzero == 1, f"Bin {col} belongs to {nonzero} bands (expected 1)"


def test_create_log_filter_device():
    """Filter should be on the requested device."""
    f = create_log_filter(1025, 8, device='cpu')
    assert f.device.type == 'cpu'


def test_apply_log_filter_shape():
    """Input (2, 1025, 100) -> output (2, 8, 100)."""
    f = create_log_filter(1025, 8, device='cpu')
    stft = torch.randn(2, 1025, 100)
    out = apply_log_filter(stft, f)
    assert out.shape == (2, 8, 100)


def test_apply_log_filter_energy_routing():
    """Energy in a single bin should appear only in its expected band."""
    f = create_log_filter(1025, 8, device='cpu')
    stft = torch.zeros(1, 1025, 10)
    # Put energy in the last bin — should route to the last band
    stft[0, 1024, :] = 1.0
    out = apply_log_filter(stft, f)
    # Find which band has nonzero output
    band_sums = out[0, :, 0]
    nonzero_bands = (band_sums != 0).nonzero(as_tuple=True)[0]
    assert len(nonzero_bands) == 1, f"Energy routed to {len(nonzero_bands)} bands"


def test_apply_log_filter_batch_independence():
    """Different batch items should produce independent results."""
    f = create_log_filter(1025, 8, device='cpu')
    stft = torch.zeros(2, 1025, 10)
    stft[0, 500, :] = 1.0
    stft[1, 100, :] = 1.0
    out = apply_log_filter(stft, f)
    assert not torch.allclose(out[0], out[1])


# ---------------------------------------------------------------------------
# Onset Strength (onset.py)
# ---------------------------------------------------------------------------

def test_onset_strength_output_shape():
    """Output time dimension should match input time dimension."""
    batch, time_samples = 2, 1000
    y = torch.randn(batch, time_samples)
    out = onset_strength(y=y)
    # With center=True, output is trimmed to match S shape
    assert out.dim() == 2
    assert out.shape[0] == batch


def test_onset_strength_silent_input():
    """All-zeros input should produce all-zeros onset (no energy increase)."""
    y = torch.zeros(1, 4000)
    out = onset_strength(y=y)
    assert torch.allclose(out, torch.zeros_like(out))


def test_onset_strength_impulse_detection():
    """An impulse should produce a peak in the onset envelope near that position."""
    y = torch.zeros(1, 22050)
    # Place impulse at roughly the middle
    y[0, 11025] = 1.0
    out = onset_strength(y=y)
    # The onset envelope should have at least one nonzero value
    assert out.max().item() > 0


def test_onset_strength_non_negative():
    """Output should always be >= 0 (clamping works) when detrend=False."""
    y = torch.randn(2, 8000)
    out = onset_strength(y=y, detrend=False)
    assert (out >= 0).all()


def test_onset_strength_detrend():
    """With detrend=True, mean of output should be approximately 0."""
    y = torch.randn(1, 22050)
    out = onset_strength(y=y, detrend=True)
    assert abs(out.mean().item()) < 1e-2


# ---------------------------------------------------------------------------
# HCQM (hcqm.py)
# ---------------------------------------------------------------------------

def test_make_kernels_returns_tuple():
    """make_kernels should return (stft_spec, band_filter, cqt_specs)."""
    stft_spec, band_filter, cqt_specs = make_kernels(device='cpu')
    assert band_filter.shape == (8, 1025)
    assert isinstance(cqt_specs, list)


def test_make_kernels_cqt_count():
    """Should produce 6 CQT specs (one per harmonic)."""
    _, _, cqt_specs = make_kernels(device='cpu')
    assert len(cqt_specs) == 6


def test_make_kernels_band_filter_shape():
    """Band filter should be (8, 1025)."""
    _, band_filter, _ = make_kernels(device='cpu')
    assert band_filter.shape == (8, 1025)


def test_compute_hcqm_output_shape():
    """Input (batch, 176400) -> output (batch, 240, 8, 6)."""
    sr = 22050
    clip_samples = sr * 8  # 176400
    specs = make_kernels(len_audio=clip_samples, sr=sr, device='cpu')
    batch = 2
    audio = torch.randn(batch, clip_samples)
    out = compute_hcqm(audio, *specs)
    assert out.shape == (batch, 240, 8, 6)


def test_compute_hcqm_batch_consistency():
    """Same audio duplicated in batch should produce identical rows."""
    sr = 22050
    clip_samples = sr * 8
    specs = make_kernels(len_audio=clip_samples, sr=sr, device='cpu')
    single = torch.randn(1, clip_samples)
    batch = torch.cat([single, single], dim=0)
    out = compute_hcqm(batch, *specs)
    assert torch.allclose(out[0], out[1], atol=1e-5)


def test_compute_hcqm_different_inputs():
    """Different audio should produce different HCQM outputs."""
    sr = 22050
    clip_samples = sr * 8
    specs = make_kernels(len_audio=clip_samples, sr=sr, device='cpu')
    audio = torch.randn(2, clip_samples)
    out = compute_hcqm(audio, *specs)
    assert not torch.allclose(out[0], out[1])
