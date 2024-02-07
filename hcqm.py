import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
import nnAudio.features as feat

N_BINS = 240
N_BANDS = 8

def onset_strength(
    y=None, sr=22050, S=None, n_fft=2048, hop_length=512, lag=1, max_size=1, ref=None,
    detrend=False, center=True, feature=None, aggregate=None, device='cuda'
):
    # Ensure y is reshaped to (batch, channels, time) if it's not already
    if y is not None and y.dim() == 2:
        y = y.unsqueeze(1)

    S = torchaudio.transforms.AmplitudeToDB(top_db=80)(y)
    ref = S

    # Compute difference to reference, spaced by lag
    onset_env = S[..., lag:] - ref[..., :-lag]
    onset_env = torch.clamp(onset_env, min=0.0)  # Discard negatives

    if aggregate is None:
        aggregate = torch.mean
    if callable(aggregate):
        onset_env = aggregate(onset_env, dim=-2)

    # Padding and detrending
    pad_width = lag
    if center:
        pad_width += n_fft // (2 * hop_length)
    onset_env = F.pad(onset_env, (pad_width, 0), "constant", 0)

    if detrend:
        onset_env -= onset_env.mean(dim=-1, keepdim=True)

    if center:
        onset_env = onset_env[..., :S.shape[-1]]
    return onset_env

def create_log_filter(num_bins, num_bands, device='cuda'):
    log_bins = np.logspace(np.log10(1), np.log10(num_bins), num=num_bands+1, base=10.0) - 1
    log_bins = np.unique(np.round(log_bins).astype(int))

    filter_matrix = torch.zeros(num_bands, num_bins, device=device)

    for i in range(num_bands):
        if i < num_bands - 1:
            start_bin, end_bin = log_bins[i], log_bins[i + 1]
        else:
            start_bin, end_bin = log_bins[i], num_bins
        filter_matrix[i, start_bin:end_bin] = 1 / (end_bin - start_bin)

    return filter_matrix

def apply_log_filter(stft_output, filter_matrix):
    stft_output_transposed = stft_output.transpose(1, 2)
    filtered_output_transposed = torch.matmul(stft_output_transposed, filter_matrix.T)
    filtered_output = filtered_output_transposed.transpose(1, 2)
    return filtered_output

def compute_hcqm(y, stft_spec, band_filter, cqt_specs):
    device = y.device
    stft = stft_spec(y)
    stft_bands = apply_log_filter(stft, band_filter)

    stft_bands_flat = stft_bands.reshape(stft.size(0)*stft_bands.size(1), stft_bands.size(2))
    osf_flat = onset_strength(y=stft_bands_flat, device=device)

    hcqm = torch.zeros((stft.size(0)*N_BANDS, N_BINS, 6))
    for h, spec in enumerate(cqt_specs):
        hcqm[:, :, h] = spec(osf_flat).mean(-1)

    hcqm = hcqm.reshape(stft_bands.size(0), N_BINS, N_BANDS, 6)
    return hcqm

def make_specs(len_audio=22050*8, sr=22050, device='cuda'):
    n_fft = 2048
    hop = 512
    n_fft_bins = int(1+n_fft/2)
    num_bands = 8
    band_filter = create_log_filter(n_fft_bins, num_bands, device=device)
    stft_spec = feat.stft.STFT(sr=sr, n_fft=n_fft, hop_length=hop, output_format='Magnitude').to(device=device)
    cqt_specs = []
    for h in [1/2, 1, 2, 3, 4, 5]:
        fmin = (32.7*h)/60# Convert from BPM to Hz
        sr_cqt = len_audio//(hop*8)
        fmax =sr_cqt/2
        # Calculate number of octaves
        num_octaves = np.log2(fmax/fmin)
        bins_per_octave = N_BINS / num_octaves
        cqt_spec = feat.cqt.CQT(sr=sr_cqt, hop_length=len_audio//hop, n_bins=N_BINS, bins_per_octave=bins_per_octave, fmin=fmin, output_format='Magnitude', pad_mode='constant').to(device=device)
        cqt_specs.append(cqt_spec)
    return stft_spec, band_filter, cqt_specs