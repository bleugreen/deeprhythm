import torch
import torchaudio
import torch.nn.functional as F

def onset_strength(
    y=None, n_fft=2048, hop_length=512, lag=1, ref=None,
    detrend=False, center=True,  aggregate=None
):
    """
    Compute the onset strength of an audio signal or a spectrogram.

    The onset strength is a measure of the increase in energy of an audio signal.

    Parameters
    ----------
    y : torch.Tensor, optional
        The raw audio waveform, expected to be a 2D tensor of shape (batch_size, time_samples).
        If provided, it will be used to compute the spectrogram internally. Default is None.
    n_fft : int, optional
        The number of FFT components. Default is 2048.
    hop_length : int, optional
        The number of samples between successive frames. Default is 512.
    lag : int, optional
        The lag between frames for computing the difference in energy. Default is 1.
    ref : torch.Tensor, optional
        The reference spectrogram to which the energy difference is computed. If None, the
        spectrogram provided by `S` or computed from `y` is used as the reference. Default is None.
    detrend : bool, optional
        If True, remove the mean from the onset envelope. Default is False.
    center : bool, optional
        If True, pad the time dimension of the onset envelope so that frames are centered around
        their timestamps. Default is True.
    aggregate : callable, optional
        A function to aggregate the channels dimension (e.g., torch.mean, torch.sum). If None,
        the mean is used. Default is None.

    Returns
    -------
    torch.Tensor
        The onset strength envelope, a 2D tensor of shape (batch_size, time_frames).

    """
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
