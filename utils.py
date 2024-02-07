import librosa
import torch

def load_and_split_audio(filename, sr=22050, clip_length=8):
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
    except Exception as e:
        print(e, filename)

    if clips:
        stacked_clips = torch.stack(clips, dim=0)
    else:
        return None

    return stacked_clips
