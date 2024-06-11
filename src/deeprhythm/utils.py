import librosa
import torch
import zipfile

import os
import requests

model_url = 'https://github.com/Mitchell57/deeprhythm/raw/main/'


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def get_weights(filename="deeprhythm-0.7.pth", quiet=False):
    # Construct the path to save the model weights
    home_dir = os.path.expanduser("~")
    model_dir = os.path.join(home_dir, ".local", "share", "deeprhythm")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, filename)

    # Check if the model weights already exist
    if not os.path.isfile(model_path):
        print("Downloading model weights...")
        # Download the model weights
        try:
            r = requests.get(model_url+filename, allow_redirects=True)
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


def load_and_split_audio(filename, sr=22050, clip_length=8, share_mem=False):
    """
    Load an audio file, split it into 8-second clips, and return a single tensor of all clips.

    Parameters:
    - filename: Path to the audio file.
    - sr: Sampling rate to use for loading the audio.
    - clip_length: Length of each clip in seconds.

    Returns:
    A tensor of shape [clips, audio] where each row is an 8-second clip.
    """

    clips = []
    clip_samples = sr * clip_length
    try:
        audio, _ = librosa.load(filename, sr=sr)
        for i in range(0, len(audio), clip_samples):
            if i + clip_samples <= len(audio):
                clip_tensor = torch.tensor(audio[i:i + clip_samples], dtype=torch.float32)
                clips.append(clip_tensor)
        if clips:
            stacked_clips = torch.stack(clips, dim=0)
        else:
            return None

        if share_mem:
            stacked_clips.share_memory_()

        return stacked_clips
    except Exception as e:
        print(e, filename)

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