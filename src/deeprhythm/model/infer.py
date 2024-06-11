import torch
import time
import os
from deeprhythm.utils import get_weights
from deeprhythm.utils import load_and_split_audio
from deeprhythm.audio_proc.hcqm import make_kernels, compute_hcqm
from deeprhythm.utils import class_to_bpm
from deeprhythm.model.frame_cnn import DeepRhythmModel

def load_cnn_model(path='deeprhythm-0.7.pth', device=None, quiet=False):
    model = DeepRhythmModel(256)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(path):
        path = get_weights(quiet=quiet)
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    model = model.to(device=device)
    model.eval()
    return model

def predict_global_bpm(input_path, model_path='deeprhythm-0.7.pth', model=None, specs=None, device='cpu'):
    if model is None:
        model = load_cnn_model(model_path, device=device)
    clips = load_and_split_audio(input_path, sr=22050)
    model_device = next(model.parameters()).device
    if specs is None:
        stft, band, cqt = make_kernels(device=model_device)
    else:
        stft, band, cqt = specs
    input_batch = compute_hcqm(clips.to(device=model_device), stft, band, cqt).permute(0,3,1,2)
    model.eval()
    start = time.time()
    with torch.no_grad():
        input_batch = input_batch.to(device=model_device)
        outputs = model(input_batch)
        probabilities = torch.softmax(outputs, dim=1)
        mean_probabilities = probabilities.mean(dim=0)
        _, predicted_class = torch.max(mean_probabilities, 0)
        predicted_global_bpm = class_to_bpm(predicted_class.item())
    return predicted_global_bpm, time.time()-start


def predict_global_bpm_cont(input_path, model_path='deeprhythm-0.5.pth', model=None, specs=None, device='cuda'):
    if model is None:
        model = load_cnn_model(model_path, device=device)
    clips = load_and_split_audio(input_path, sr=22050)
    model_device = next(model.parameters()).device
    if specs is None:
        stft, band, cqt = make_kernels(device=model_device)
    else:
        stft, band, cqt = specs
    input_batch = compute_hcqm(clips.to(device=model_device), stft, band, cqt).permute(0,3,1,2)
    model.eval()
    start = time.time()
    with torch.no_grad():
        input_batch = input_batch.to(device=model_device)
        outputs = model(input_batch)
        probabilities = torch.softmax(outputs, dim=1)
        expected_bpm = torch.sum(probabilities * torch.arange(256).float().to(device), dim=1)
    mean_expected_bpm = torch.mean(expected_bpm)
    return mean_expected_bpm, time.time()-start