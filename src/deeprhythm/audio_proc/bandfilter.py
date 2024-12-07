import numpy as np
import torch


def create_log_filter(num_bins, num_bands, device='cuda'):
    """
    Create a logarithmically spaced filter matrix.

    Parameters
    ----------
    num_bins : int
        Number of frequency bins in the spectrogram
    num_bands : int
        Number of logarithmically spaced bands
    device : str, optional
        Target device for the filter matrix

    Returns
    -------
    torch.Tensor
        Filter matrix of shape (num_bands, num_bins)
    """
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
    """
    Apply logarithmic filter matrix to STFT output.

    Parameters
    ----------
    stft_output : torch.Tensor
        STFT output of shape (batch_size, num_bins, num_frames)
    filter_matrix : torch.Tensor
        Filter matrix of shape (num_bands, num_bins)

    Returns
    -------
    torch.Tensor
        Filtered output of shape (batch_size, num_bands, num_frames)
    """
    stft_output_transposed = stft_output.transpose(1, 2)
    filtered_output_transposed = torch.matmul(stft_output_transposed, filter_matrix.T)
    filtered_output = filtered_output_transposed.transpose(1, 2)
    return filtered_output