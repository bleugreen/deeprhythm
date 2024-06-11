import torch
from deeprhythm.utils import load_and_split_audio
from deeprhythm.audio_proc.hcqm import make_kernels, compute_hcqm
from deeprhythm.utils import class_to_bpm
from deeprhythm.model.frame_cnn import DeepRhythmModel
from deeprhythm.utils import get_weights, get_device

class DeepRhythmPredictor:
    def __init__(self, model_path='deeprhythm-0.5.pth', device=None, quiet=False):
        self.model_path = get_weights(quiet=quiet)
        if device is None:
            self.device = get_device()
        else:
            self.device = torch.device(device)
        self.model = self.load_model()
        self.specs = self.make_kernels()

    def load_model(self):
        model = DeepRhythmModel()
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model = model.to(device=self.device)
        model.eval()
        return model

    def make_kernels(self, device=None):
        if device is None:
            device = self.device
        stft, band, cqt = make_kernels(device=device)
        return stft, band, cqt

    def predict(self, filename, include_confidence=False):
        clips = load_and_split_audio(filename, sr=22050)
        input_batch = compute_hcqm(clips.to(device=self.device), *self.specs).permute(0,3,1,2)
        self.model.eval()
        with torch.no_grad():
            input_batch = input_batch.to(device=self.device)
            outputs = self.model(input_batch)
            probabilities = torch.softmax(outputs, dim=1)
            mean_probabilities = probabilities.mean(dim=0)
            confidence_score, predicted_class = torch.max(mean_probabilities, 0)
            predicted_global_bpm = class_to_bpm(predicted_class.item())
        if include_confidence:
            return predicted_global_bpm, confidence_score.item(),
        return predicted_global_bpm

    def predict_batch(self, dirname):
        # Placeholder for batch prediction logic
        # This would involve iterating over files in dirname, using self.predict on each,
        # and aggregating or returning results as needed.
        pass
