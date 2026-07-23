"""Shared-STFT temporal evidence for hierarchical tempo inference."""

import math

import torch
from pytorch_tcn import TCN
from torch import nn
from torch.nn import functional as F

from deeprhythm.model.hierarchical import LEVEL_FACTORS


class TemporalFeatureExtraction(nn.Module):
    def __init__(self, num_bands=81, num_channels=36):
        super().__init__()
        self.conv1 = nn.Conv1d(num_bands, num_channels, 3, padding=1)
        self.pool1 = nn.MaxPool1d(3, stride=1, padding=1)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(0.1)
        self.conv2 = nn.Conv1d(num_channels, num_channels, 3, padding=1)
        self.pool2 = nn.MaxPool1d(3, stride=1, padding=1)
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(0.1)
        self.conv3 = nn.Conv1d(num_channels, num_channels, 3, padding=1)
        self.pool3 = nn.MaxPool1d(3, stride=1, padding=1)
        self.elu3 = nn.ELU()
        self.dropout3 = nn.Dropout(0.1)

    def forward(self, inputs):
        inputs = self.dropout1(self.elu1(self.pool1(self.conv1(inputs))))
        inputs = self.dropout2(self.elu2(self.pool2(self.conv2(inputs))))
        return self.dropout3(self.elu3(self.pool3(self.conv3(inputs))))


class TemporalBeatEncoder(nn.Module):
    def __init__(self, channels=36):
        super().__init__()
        self.feature_extraction = TemporalFeatureExtraction(num_channels=channels)
        self.tcn_beat = TCN(
            num_inputs=channels,
            num_channels=[channels] * 16,
            kernel_size=5,
            dropout=0.1,
            causal=False,
            use_skip_connections=True,
            kernel_initializer="kaiming_normal",
            use_norm="layer_norm",
            activation="relu",
            dilation_reset=8,
        )

    def forward(self, spectrum):
        return self.tcn_beat(self.feature_extraction(spectrum)).transpose(1, 2)


class CircularBeatDecoder(nn.Module):
    def __init__(self, channels=36):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(channels, 72), nn.ELU(), nn.Dropout(0.1), nn.Linear(72, 2))

    def forward(self, temporal):
        return self.net(temporal)


class MetricalFusion(nn.Module):
    def __init__(self, input_features=162):
        super().__init__()
        self.register_buffer("mean", torch.zeros(input_features))
        self.register_buffer("scale", torch.ones(input_features))
        self.net = nn.Sequential(
            nn.Linear(input_features, 96),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Linear(48, len(LEVEL_FACTORS)),
        )

    def forward(self, features):
        return self.net((features - self.mean) / self.scale)


def temporal_metrical_features(temporal, raw_phase, base_probability, level_probability, frame_rate=22050 / 512):
    """Shape the frozen temporal representation exactly as used during training."""
    if temporal.shape[0] != 1:
        raise ValueError("metrical feature shaping expects one track at a time")
    temporal = temporal[0]
    raw_phase = raw_phase[0]
    hidden = torch.cat([temporal.mean(0), temporal.std(0), temporal.amax(0), temporal.amin(0)])
    confidence = raw_phase.norm(dim=-1)
    unit = F.normalize(raw_phase, dim=-1)
    phase = torch.complex(unit[:, 0], unit[:, 1])
    times = torch.arange(len(phase), device=phase.device) / frame_rate
    base_bpm = 60 + 0.5 * int(base_probability.argmax())
    values = []
    for factor in LEVEL_FACTORS:
        omega = 2 * math.pi * base_bpm * factor / 60
        demodulated = phase * torch.exp(torch.complex(torch.zeros_like(times), -omega * times))
        values.extend([demodulated.mean().abs(), demodulated.square().mean().abs(), confidence.mean()])
    entropy = -(level_probability * level_probability.clamp_min(1e-8).log()).sum()
    return torch.cat(
        [hidden, torch.stack(values), level_probability, level_probability.max()[None], entropy[None]]
    )