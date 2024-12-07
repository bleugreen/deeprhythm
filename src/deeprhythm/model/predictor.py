import torch
from deeprhythm.utils import load_and_split_audio, split_audio
from deeprhythm.audio_proc.hcqm import make_kernels, compute_hcqm
from deeprhythm.utils import class_to_bpm
from deeprhythm.model.frame_cnn import DeepRhythmModel
from deeprhythm.utils import get_weights, get_device
from deeprhythm.batch_infer import get_audio_files, main as batch_infer_main
import json
import tempfile
import os

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
    
    def predict_from_audio(self, audio, sr, include_confidence=False):
        clips = split_audio(audio, sr)
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

    def predict_batch(self, dirname, include_confidence=False, workers=8, batch=128, quiet=True):
        """
        Predict BPM for all audio files in a directory using efficient batch processing.
        
        Args:
            dirname: Directory containing audio files
            include_confidence: Whether to include confidence scores in results
            
        Returns:
            dict: Mapping of filenames to their predicted BPMs (and optionally confidence scores)
        """
        # Create a temporary file to store batch results
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.jsonl', delete=False) as tmp_file:
            temp_path = tmp_file.name
            
        try:
            # Run batch inference
            batch_infer_main(
                dataset=get_audio_files(dirname),
                data_path=temp_path,
                device=str(self.device),
                conf=include_confidence,
                quiet=quiet,
                n_workers=workers,
                max_len_batch=batch
            )
            
            # Read and parse results
            results = {}
            with open(temp_path, 'r') as f:
                for line in f:
                    result = json.loads(line.strip())
                    filename = result.pop('filename')
                    if include_confidence:
                        results[filename] = (result['bpm'], result['confidence'])
                    else:
                        results[filename] = result['bpm']
                        
            return results
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def predict_per_frame(self, filename, include_confidence=False):
        clips = load_and_split_audio(filename, sr=22050)
        input_batch = compute_hcqm(clips.to(device=self.device), *self.specs).permute(0,3,1,2)
        self.model.eval()
        with torch.no_grad():
            input_batch = input_batch.to(device=self.device)
            outputs = self.model(input_batch)
            probabilities = torch.softmax(outputs, dim=1)
            confidence_scores, predicted_classes = torch.max(probabilities, dim=1)
            predicted_bpms = [class_to_bpm(cls.item()) for cls in predicted_classes]
            
        if include_confidence:
            return predicted_bpms, confidence_scores.tolist()
        return predicted_bpms
