import torch.nn as nn
import torch.nn.functional as F
import torch
from src.model.frame_cnn import DeepRhythmModel

class Attention(nn.Module):
    def __init__(self, feature_dim,):
        super(Attention, self).__init__()
        self.feature_dim = feature_dim
        self.attention_weights = nn.Linear(self.feature_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_length, feature_dim)
        weights = self.attention_weights(x)  # Compute attention scores
        weights = torch.softmax(weights, dim=1)  # Normalize scores across seq_length
        weighted_sum = torch.sum(x * weights, dim=1)  # Compute weighted sum
        return weighted_sum, weights

class GlobalBPMPredictor(nn.Module):
    def __init__(self, cnn_model_path='model_weights/deeprhythm0.2.pth', feature_dim=256, num_classes=256, device=None):
        super(GlobalBPMPredictor, self).__init__()
        self.cnn_model = DeepRhythmModel(num_classes=num_classes)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=torch.device(self.device)))
        self.cnn_model = self.cnn_model.to(device=self.device)
        self.attention = Attention(feature_dim).to(self.device)
        self.prediction_layer = nn.Linear(feature_dim, num_classes).to(self.device)

    def forward(self, batch):
        # Flatten the batch
        flat_list_of_clips = [clip for song in batch for clip in song]
        flat_list_of_clips = torch.stack(flat_list_of_clips).to(self.device)  # Shape: (batch*X, 6, 240, 8)
        # Compute CNN predictions for each clip
        cnn_outputs = self.cnn_model(flat_list_of_clips.permute(0, 3, 1, 2))  # Shape: (batch*X, 256)
        # Restack into (batch, X, 256), need original batch and X for each item in batch
        restacked_cnn_outputs = cnn_outputs.split([len(song) for song in batch])  # A list of tensors
        restacked_cnn_outputs = [song_tensors.unsqueeze(0) for song_tensors in restacked_cnn_outputs]  # Shape: (1, X, 256)
        predictions = []
        for cnn_out in restacked_cnn_outputs:
            weighted_outputs, _ = self.attention(cnn_out)  # Shape: (batch, X, 256)
            prediction = self.prediction_layer(weighted_outputs)
            predictions.append(prediction)
        stacked_pred = torch.stack(predictions)
        return stacked_pred.squeeze(1)