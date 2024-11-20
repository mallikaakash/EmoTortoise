# models/emotortoise_synthesizer.py

import torch
import torch.nn as nn
from models.emotion_embedding.emotion_embedding import EmotionEmbedding
from models.prosody_predictor.prosody_predictor import ProsodyPredictor
from models.mel_generator.emotion_aware_mel_generator import EmotionAwareMelGenerator

class EmoTortoiseSynthesizer(nn.Module):
    def __init__(self, num_emotions=9, emotion_embedding_dim=128, prosody_dim=3, text_encoder_dim=256, hidden_dim=256, mel_dim=80):
        super(EmoTortoiseSynthesizer, self).__init__()
        self.emotion_embedding = EmotionEmbedding(num_emotions, emotion_embedding_dim)
        self.prosody_predictor = ProsodyPredictor(emotion_embedding_dim, prosody_dim)
        self.mel_generator = EmotionAwareMelGenerator(
            text_encoder_dim=text_encoder_dim,
            prosody_dim=prosody_dim,
            hidden_dim=hidden_dim,
            mel_dim=mel_dim
        )
        # Placeholder for text encoder (you need to define or integrate an actual text encoder)
        self.text_encoder = nn.Linear(300, text_encoder_dim)  # Example: input_dim=300 (e.g., from pre-trained embeddings)
    
    def forward(self, text_inputs, emotion_indices):
        # Encode text
        text_features = self.text_encoder(text_inputs)
        
        # Embed emotions
        emotion_embeds = self.emotion_embedding(emotion_indices)
        
        # Predict prosody
        prosody_features = self.prosody_predictor(emotion_embeds)
        
        # Generate mel-spectrogram
        mel_spectrogram = self.mel_generator(text_features, prosody_features)
        return mel_spectrogram
