import os

import librosa
import requests
import torch

model_url = 'https://github.com/bleugreen/deeprhythm/raw/main/weights/'


class AudioTooShortError(ValueError):
    """Raised when audio file is shorter than minimum required length."""
    pass


class AudioLoadError(IOError):
    """Raised when audio file cannot be loaded."""
    pass


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def get_weights(filename="deeprhythm-0.7.pth", quiet=False):
    home_dir = os.path.expanduser("~")
    model_dir = os.path.join(home_dir, ".local", "share", "deeprhythm")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, filename)

    if not os.path.isfile(model_path):
        print("Downloading model weights...")
        try:
            r = requests.get(model_url + filename, allow_redirects=True)
            if r.status_code == 200:
                with open(model_path, 'wb') as f:
                    f.write(r.content)
                print("Model weights downloaded successfully.")
            else:
                print(f"Failed to download model weights. HTTP Error: {r.status_code}")
        except Exception as e:
            print(f"An error occurred during the download: {e}")
    else:
        if not quiet:
            print("Model weights already exist.")

    return model_path


def split_audio(audio, sr, clip_length=8, share_mem=False):
    """
    Split audio into fixed-length clips and return a stacked tensor.

    Parameters:
    - audio: Audio array (e.g. from librosa.load).
    - sr: Sampling rate.
    - clip_length: Length of each clip in seconds.
    - share_mem: Whether to put the tensor in shared memory (for multiprocessing).

    Returns:
    A tensor of shape [num_clips, clip_samples].

    Raises:
    AudioTooShortError: If audio is too short for even one clip.
    """
    clips = []
    clip_samples = sr * clip_length
    for i in range(0, len(audio), clip_samples):
        if i + clip_samples <= len(audio):
            clip_tensor = torch.tensor(audio[i:i + clip_samples], dtype=torch.float32)
            clips.append(clip_tensor)
    if not clips:
        raise AudioTooShortError(
            f"Audio must be at least {clip_length} seconds long to extract clips. "
            f"Provided audio has {len(audio)/sr:.2f} seconds."
        )

    stacked_clips = torch.stack(clips, dim=0)
    if share_mem:
        stacked_clips.share_memory_()
    return stacked_clips


def load_and_split_audio(filename, sr=22050, clip_length=8, share_mem=False):
    """
    Load an audio file and split it into fixed-length clips.

    Parameters:
    - filename: Path to the audio file.
    - sr: Sampling rate to use for loading the audio.
    - clip_length: Length of each clip in seconds.
    - share_mem: Whether to put the tensor in shared memory (for multiprocessing).

    Returns:
    A tensor of shape [num_clips, clip_samples].

    Raises:
    AudioTooShortError: If audio is too short for even one clip.
    AudioLoadError: If the audio file cannot be loaded.
    """
    try:
        audio, _ = librosa.load(filename, sr=sr)
        return split_audio(audio, sr, clip_length=clip_length, share_mem=share_mem)
    except AudioTooShortError:
        raise
    except Exception as e:
        raise AudioLoadError(
            f"Failed to load audio file '{filename}': {str(e)}"
        ) from e


def bpm_to_class(bpm, min_bpm=30, max_bpm=286, num_classes=256):
    """Map a BPM value to a class index."""
    class_width = (max_bpm - min_bpm) / num_classes
    class_index = int((bpm - min_bpm) // class_width)
    return max(0, min(num_classes - 1, class_index))


def class_to_bpm(class_index, min_bpm=30, max_bpm=286, num_classes=256):
    """Map a class index back to a BPM value (to the center of the class interval)."""
    class_width = (max_bpm - min_bpm) / num_classes
    bpm = min_bpm + class_width * (class_index)
    return bpm
