# models/emotion_embedding/emotion_embedding.py

import torch
import torch.nn as nn

class EmotionEmbedding(nn.Module):
    def __init__(self, num_emotions=9, embedding_dim=128):
        super(EmotionEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_emotions, embedding_dim)
    
    def forward(self, emotion_indices):
        return self.embedding(emotion_indices)
