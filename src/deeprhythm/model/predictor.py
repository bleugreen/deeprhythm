import json
import os
import tempfile
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from deeprhythm.audio_proc.hcqm import compute_hcqm, make_kernels
from deeprhythm.batch_infer import get_audio_files
from deeprhythm.batch_infer import main as batch_infer_main
from deeprhythm.model.frame_cnn import DeepRhythmModel
from deeprhythm.utils import class_to_bpm, get_device, get_weights, load_and_split_audio, split_audio


class DeepRhythmPredictor:
    def __init__(self, device: Optional[str] = None, quiet: bool = False):
        """Initialize the DeepRhythm BPM predictor.

        Args:
            model_path: Path to the model weights file
            device: Device to run inference on ('cuda' or 'cpu')
            quiet: Whether to suppress progress messages
        """
        self.model_path = get_weights(quiet=quiet)
        self.device = torch.device(device if device else get_device())
        self.model = self._load_model()
        self.specs = self._make_kernels()

    def _load_model(self) -> DeepRhythmModel:
        """Load and initialize the model."""
        model = DeepRhythmModel()
        model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        model = model.to(device=self.device)
        model.eval()
        return model

    def _make_kernels(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Initialize HCQM computation kernels."""
        return make_kernels(device=self.device)

    def _process_clips(self, clips: Tensor, include_confidence: bool = False) -> Union[float, Tuple[float, float]]:
        """Process audio clips and return BPM prediction.

        Args:
            clips: Tensor of audio clips
            include_confidence: Whether to return confidence score

        Returns:
            Predicted BPM or tuple of (BPM, confidence)
        """
        input_batch = compute_hcqm(clips.to(device=self.device), *self.specs).permute(0,3,1,2)
        
        with torch.no_grad():
            outputs = self.model(input_batch.to(device=self.device))
            probabilities = torch.softmax(outputs, dim=1)
            mean_probabilities = probabilities.mean(dim=0)
            confidence_score, predicted_class = torch.max(mean_probabilities, 0)
            predicted_bpm = class_to_bpm(predicted_class.item())
            
        return (predicted_bpm, confidence_score.item()) if include_confidence else predicted_bpm

    def predict(self, filename: str, include_confidence: bool = False) -> Union[float, Tuple[float, float]]:
        """Predict BPM from an audio file.

        Args:
            filename: Path to the audio file
            include_confidence: Whether to return confidence score

        Returns:
            Predicted BPM or tuple of (BPM, confidence)
        """
        clips = load_and_split_audio(filename, sr=22050)
        return self._process_clips(clips, include_confidence)
    
    def predict_from_audio(
        self, audio: List[float], sr: int, include_confidence: bool = False
    ) -> Union[float, Tuple[float, float]]:
        """Predict BPM from audio tensor.

        Args:
            audio: Audio list
            sr: Sample rate
            include_confidence: Whether to return confidence score

        Returns:
            Predicted BPM or tuple of (BPM, confidence)
        """
        clips = split_audio(audio, sr)
        return self._process_clips(clips, include_confidence)

    def predict_per_frame(
        self, filename: str, include_confidence: bool = False
    ) -> Union[List[float], Tuple[List[float], List[float]]]:
        """Predict BPM for each frame in an audio file.

        Args:
            filename: Path to the audio file
            include_confidence: Whether to return confidence scores

        Returns:
            List of BPMs or tuple of (BPMs, confidence scores)
        """
        clips = load_and_split_audio(filename, sr=22050)
        input_batch = compute_hcqm(clips.to(device=self.device), *self.specs).permute(0,3,1,2)
        
        with torch.no_grad():
            outputs = self.model(input_batch.to(device=self.device))
            probabilities = torch.softmax(outputs, dim=1)
            confidence_scores, predicted_classes = torch.max(probabilities, dim=1)
            predicted_bpms = [class_to_bpm(cls.item()) for cls in predicted_classes]
            
        return (predicted_bpms, confidence_scores.tolist()) if include_confidence else predicted_bpms

    def predict_batch(self, dirname: str, include_confidence: bool = False, workers: int = 8, 
                     batch: int = 128, quiet: bool = True) -> Dict[str, Union[float, Tuple[float, float]]]:
        """Predict BPM for all audio files in a directory using efficient batch processing.
        
        Args:
            dirname: Directory containing audio files
            include_confidence: Whether to include confidence scores
            workers: Number of worker processes
            batch: Batch size for processing
            quiet: Whether to suppress progress messages
            
        Returns:
            Dictionary mapping filenames to predictions
        """
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.jsonl', delete=False) as tmp_file:
            temp_path = tmp_file.name
            
        try:
            batch_infer_main(
                dataset=get_audio_files(dirname),
                data_path=temp_path,
                device=str(self.device),
                conf=include_confidence,
                quiet=quiet,
                n_workers=workers,
                max_len_batch=batch
            )
            
            results = {}
            with open(temp_path, 'r') as f:
                for line in f:
                    result = json.loads(line.strip())
                    filename = result.pop('filename')
                    results[filename] = (result['bpm'], result['confidence']) if include_confidence else result['bpm']
                        
            return results
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
