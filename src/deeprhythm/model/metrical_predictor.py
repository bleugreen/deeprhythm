"""Opt-in v0.8 predictor with shared HCQM and temporal beat evidence."""

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from deeprhythm.audio_proc.hcqm import compute_hcqm_from_stft, make_kernels
from deeprhythm.audio_proc.log_spectrum import compute_log_spectrum_from_clips, make_log_spectrum_filter
from deeprhythm.model.hierarchical import HierarchicalDeepRhythmModel, hierarchy_to_tempo
from deeprhythm.model.metrical import (
    CircularBeatDecoder,
    MetricalFusion,
    TemporalBeatEncoder,
    temporal_metrical_features,
)
from deeprhythm.utils import get_device, load_audio, split_audio

WEIGHT_DIR = Path(__file__).resolve().parent.parent / "weights"


def _trim_edge_clips(clips):
    """Match the edge-trimmed track aggregation used to train v0.8."""
    return clips[1:-1] if len(clips) > 5 else clips


def _fusion_cache_precision(tensor):
    """Reproduce the float16 feature cache on which the frozen fusion was fit."""
    return tensor.to(torch.float16).to(torch.float32)


class MetricalDeepRhythmPredictor:
    """Predict tempo with the frozen, reproducible shared-STFT v0.8 model."""

    def __init__(self, device=None):
        self.device = torch.device(device or get_device())
        self.hierarchical = HierarchicalDeepRhythmModel(residual_level=True).to(self.device)
        hierarchy = torch.load(WEIGHT_DIR / "hierarchical-v0.8.pt", map_location=self.device, weights_only=False)
        self.hierarchical.load_state_dict(hierarchy["model_state_dict"])
        self.residual_scale = hierarchy["residual_scale"]

        self.temporal = TemporalBeatEncoder().to(self.device)
        self.beat_decoder = CircularBeatDecoder().to(self.device)
        temporal = torch.load(WEIGHT_DIR / "temporal-beat-v0.8.pt", map_location=self.device, weights_only=False)
        temporal_state = {
            key: value
            for key, value in temporal["model"].items()
            if key.startswith(("feature_extraction.", "tcn_beat."))
        }
        self.temporal.load_state_dict(temporal_state)
        self.beat_decoder.load_state_dict(temporal["beat"])

        fusion = torch.load(WEIGHT_DIR / "shared-fusion-v0.8.pt", map_location=self.device, weights_only=False)
        self.fusion = MetricalFusion(fusion["state_dict"]["mean"].numel()).to(self.device)
        self.fusion.load_state_dict(fusion["state_dict"])
        self.alpha = fusion["alpha"]
        self.specs = make_kernels(device=self.device)
        self.temporal_filter = make_log_spectrum_filter(device=self.device)
        for model in (self.hierarchical, self.temporal, self.beat_decoder, self.fusion):
            model.eval()

    def predict(self, filename, include_details=False):
        audio = load_audio(filename, sr=22050)
        return self.predict_from_audio(audio, include_details=include_details)

    def predict_from_audio(self, audio, include_details=False):
        clips = split_audio(audio, 22050)
        available_clips = len(clips)
        if available_clips < 4:
            padded = np.pad(np.asarray(audio)[: 32 * 22050], (0, max(0, 32 * 22050 - len(audio))))
            phase_clips = torch.from_numpy(padded.astype(np.float32)).view(4, 8 * 22050)
            clips = torch.cat([clips, phase_clips[available_clips:]], dim=0)
        clips = clips.to(self.device)
        with torch.no_grad():
            magnitude = self.specs[0](clips)
            track_magnitude = _trim_edge_clips(magnitude[:available_clips])
            hcqm = compute_hcqm_from_stft(track_magnitude, self.specs[1], self.specs[2])
            embedding = self.hierarchical.backbone.encode(hcqm.permute(0, 3, 1, 2).to(self.device))
            base_probability = F.softmax(self.hierarchical.base_head(embedding), dim=1).mean(0)
            level_logits = self.hierarchical.level_head(embedding)
            level_logits += self.residual_scale * self.hierarchical.level_residual(embedding)
            level_probability = F.softmax(level_logits, dim=1).mean(0)
            spectrum = compute_log_spectrum_from_clips(magnitude[:4], filter_matrix=self.temporal_filter)
            base_probability = _fusion_cache_precision(base_probability)
            level_probability = _fusion_cache_precision(level_probability)
            spectrum = _fusion_cache_precision(spectrum)
            temporal = self.temporal(spectrum[None])
            phase = self.beat_decoder(temporal)
            features = temporal_metrical_features(temporal, phase, base_probability, level_probability)
            fused_logits = level_probability.clamp_min(1e-8).log() + self.alpha * self.fusion(features[None])[0]
            base_class = base_probability.argmax()
            level = fused_logits.argmax()
            bpm = float(hierarchy_to_tempo(base_class.cpu(), level.cpu()))
        if not include_details:
            return bpm
        return {
            "bpm": bpm,
            "base_bpm": 60 + 0.5 * int(base_class),
            "level_factor": (0.5, 1.0, 2.0, 4.0)[int(level)],
            "num_clips": len(track_magnitude),
            "base_probability": base_probability.cpu().tolist(),
            "level_probability": level_probability.cpu().tolist(),
        }