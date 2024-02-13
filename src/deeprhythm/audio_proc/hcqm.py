import numpy as np
import torch
import nnAudio.features as feat
from deeprhythm.audio_proc.bandfilter import create_log_filter, apply_log_filter
from deeprhythm.audio_proc.onset import onset_strength

N_BINS = 240
N_BANDS = 8

def make_kernels(len_audio=22050*8, sr=22050, device='cuda'):
    """
    Create the kernels for the STFT and CQT based on the input parameters.

    Parameters
    ----------
    len_audio : int
        The length of the audio signal in samples.
    sr : int
        The sampling rate of the audio signal.
    device : str
        The device to use for computation ('cuda' or 'cpu').

    Returns:
    stft_spec : STFT object
        An object to compute the Short-Time Fourier Transform (STFT).
    band_filter : Tensor
        A filter matrix of shape (N_BANDS, n_fft_bins) to apply to the STFT.
    cqt_specs : list of CQT objects
        A list of Constant-Q Transform (CQT) objects for different harmonics.
    """
    n_fft = 2048
    hop = 512
    n_fft_bins = int(1+n_fft/2)
    band_filter = create_log_filter(n_fft_bins, N_BANDS, device=device)
    stft_spec = feat.stft.STFT(sr=sr, n_fft=n_fft, hop_length=hop, output_format='Magnitude', verbose=False).to(device=device)
    cqt_specs = []
    for h in [1/2, 1, 2, 3, 4, 5]:
        # Convert from BPM to Hz
        fmin = (32.7*h)/60
        sr_cqt = len_audio//(hop*8)
        fmax =sr_cqt/2
        num_octaves = np.log2(fmax/fmin)
        bins_per_octave = N_BINS / num_octaves
        cqt_spec = feat.cqt.CQT(sr=sr_cqt,
                                hop_length=len_audio//hop,
                                n_bins=N_BINS,
                                bins_per_octave=bins_per_octave,
                                fmin=fmin,
                                output_format='Magnitude',
                                verbose=False,
                                pad_mode='constant').to(device=device)
        cqt_specs.append(cqt_spec)
    return stft_spec, band_filter, cqt_specs

def compute_hcqm(y, stft_spec, band_filter, cqt_specs):
    """
    Compute the Harmonic Constant-Q Modulation (HCQM) for an input signal.

    As described by Foroughmand & Peeters in
    "Deep-Rhythm for Tempo Estimation and Rhythm Pattern Recognition", 2019

    Parameters:
    - y (Tensor): The input signal tensor of shape (batch_size, num_samples).
    - stft_spec (STFT object): An object to compute the Short-Time Fourier Transform (STFT).
    - band_filter (Tensor): A filter matrix of shape (num_bands, num_bins) to apply to the STFT.
    - cqt_specs (list of CQT objects): A list of Constant-Q Transform (CQT) objects for different harmonics / bands

    Returns:
    - hcqm (Tensor): The computed HCQM of shape (batch_size, N_BINS, N_BANDS, N_HARMONICS), where 6 corresponds to the number of different harmonics analyzed.
    """
    stft = stft_spec(y)
    stft_bands = apply_log_filter(stft, band_filter)
    stft_bands_flat = stft_bands.reshape(stft.size(0)*stft_bands.size(1), stft_bands.size(2))
    osf_flat = onset_strength(y=stft_bands_flat)
    hcqm = torch.zeros((stft.size(0)*N_BANDS, N_BINS, 6))
    for h, spec in enumerate(cqt_specs):
        hcqm[:, :, h] = spec(osf_flat).mean(-1)
    hcqm = hcqm.reshape(stft_bands.size(0), N_BINS, N_BANDS, 6)
    return hcqm
