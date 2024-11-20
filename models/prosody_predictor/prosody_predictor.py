# models/prosody_predictor/prosody_predictor.py

import torch
import torch.nn as nn

class ProsodyPredictor(nn.Module):
    def __init__(self, emotion_embedding_dim=128, prosody_dim=3):  # Pitch, Energy, Duration
        super(ProsodyPredictor, self).__init__()
        self.fc1 = nn.Linear(emotion_embedding_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, prosody_dim)
    
    def forward(self, emotion_embeddings):
        x = self.fc1(emotion_embeddings)
        x = self.relu(x)
        prosody_features = self.fc2(x)
        return prosody_features  # Returns pitch, energy, duration
