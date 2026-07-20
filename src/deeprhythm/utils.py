import hashlib
import os
import tempfile

import librosa
import requests
import torch

MODEL_WEIGHTS_FILENAME = "deeprhythm-0.7.pth"
MODEL_WEIGHTS_URL = (
    "https://raw.githubusercontent.com/bleugreen/deeprhythm/"
    "c3548590d12f5d97ccd39dee27e9591e522c05e2/weights/deeprhythm-0.7.pth"
)
MODEL_WEIGHTS_SHA256 = "c7cc8cc0425929cd2bf695474d7ec1fd63ed0d0a4a68f361d4e4b57bd9b3d9c4"
MODEL_WEIGHTS_SIZE = 5_443_228
MODEL_DOWNLOAD_CONNECT_TIMEOUT = 10
MODEL_DOWNLOAD_READ_TIMEOUT = 60
MODEL_DOWNLOAD_TIMEOUT = (MODEL_DOWNLOAD_CONNECT_TIMEOUT, MODEL_DOWNLOAD_READ_TIMEOUT)
MODEL_DOWNLOAD_CHUNK_SIZE = 64 * 1024


class AudioTooShortError(ValueError):
    """Raised when audio file is shorter than minimum required length."""
    pass


class ModelWeightsError(RuntimeError):
    """Raised when verified model weights cannot be acquired."""


def _weights_are_valid(path):
    try:
        if os.path.getsize(path) != MODEL_WEIGHTS_SIZE:
            return False
        digest = hashlib.sha256()
        with open(path, "rb") as weights_file:
            for chunk in iter(lambda: weights_file.read(MODEL_DOWNLOAD_CHUNK_SIZE), b""):
                digest.update(chunk)
        return digest.hexdigest() == MODEL_WEIGHTS_SHA256
    except OSError:
        return False


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


def get_weights(quiet=False):
    home_dir = os.path.expanduser("~")
    model_dir = os.path.join(home_dir, ".local", "share", "deeprhythm")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, MODEL_WEIGHTS_FILENAME)

    if _weights_are_valid(model_path):
        if not quiet:
            print("Model weights already exist and are verified.")
        return model_path

    temporary_path = None
    try:
        print("Downloading model weights...")
        with requests.get(
            MODEL_WEIGHTS_URL,
            stream=True,
            timeout=MODEL_DOWNLOAD_TIMEOUT,
        ) as response:
            response.raise_for_status()
            digest = hashlib.sha256()
            downloaded_size = 0
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=model_dir,
                prefix=f".{MODEL_WEIGHTS_FILENAME}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = temporary_file.name
                for chunk in response.iter_content(chunk_size=MODEL_DOWNLOAD_CHUNK_SIZE):
                    if not chunk:
                        continue
                    downloaded_size += len(chunk)
                    if downloaded_size > MODEL_WEIGHTS_SIZE:
                        raise ModelWeightsError(
                            f"Model weights exceed the expected size of {MODEL_WEIGHTS_SIZE} bytes"
                        )
                    digest.update(chunk)
                    temporary_file.write(chunk)

        if downloaded_size != MODEL_WEIGHTS_SIZE:
            raise ModelWeightsError(
                f"Model weights have size {downloaded_size} bytes; expected {MODEL_WEIGHTS_SIZE}"
            )
        if digest.hexdigest() != MODEL_WEIGHTS_SHA256:
            raise ModelWeightsError("Model weights failed SHA-256 verification")

        os.replace(temporary_path, model_path)
        temporary_path = None
        if not quiet:
            print("Model weights downloaded and verified successfully.")
        return model_path
    except requests.RequestException as exc:
        raise ModelWeightsError(f"Failed to download model weights: {exc}") from exc
    except OSError as exc:
        raise ModelWeightsError(f"Failed to install model weights: {exc}") from exc
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def load_weights(device, quiet=False):
    """Acquire verified weights and safely deserialize their state dictionary."""
    path = get_weights(quiet=quiet)
    return torch.load(path, map_location=device, weights_only=True)


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
