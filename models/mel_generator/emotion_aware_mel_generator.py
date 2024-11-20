# models/mel_generator/emotion_aware_mel_generator.py

import torch
import torch.nn as nn

class EmotionAwareMelGenerator(nn.Module):
    def __init__(self, text_encoder_dim, prosody_dim, hidden_dim, mel_dim):
        super(EmotionAwareMelGenerator, self).__init__()
        self.fc_text = nn.Linear(text_encoder_dim, hidden_dim)
        self.fc_prosody = nn.Linear(prosody_dim, hidden_dim)
        self.fc_combined = nn.Linear(hidden_dim * 2, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_mel = nn.Linear(hidden_dim, mel_dim)
    
    def forward(self, text_features, prosody_features):
        text_processed = self.fc_text(text_features)
        prosody_processed = self.fc_prosody(prosody_features)
        combined = torch.cat((text_processed, prosody_processed), dim=1)
        combined = self.fc_combined(combined).unsqueeze(1)  # Add time dimension
        mel_outputs, _ = self.decoder(combined)
        mel_spectrogram = self.fc_mel(mel_outputs.squeeze(1))
        return mel_spectrogram
