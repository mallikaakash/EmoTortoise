# scripts/synthesize_audio.py

import torch
from transformers import BertTokenizer
from models.emotortoise_synthesizer import EmoTortoiseSynthesizer
import numpy as np
import librosa
import soundfile as sf

def load_model(model_path, device):
    model = EmoTortoiseSynthesizer(
        num_emotions=9,
        emotion_embedding_dim=128,
        prosody_dim=3,
        text_encoder_dim=256,
        hidden_dim=256,
        mel_dim=80
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def synthesize(text, emotion, model, tokenizer, device):
    encoding = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=50)
    input_ids = encoding['input_ids'].squeeze(0).float().to(device)
    emotion_tensor = torch.tensor([emotion], dtype=torch.long).to(device)
    
    with torch.no_grad():
        mel_spectrogram = model(input_ids.unsqueeze(0), emotion_tensor)
    
    mel = mel_spectrogram.squeeze(0).cpu().numpy()
    
    # Placeholder for vocoder: Replace with actual vocoder implementation
    # Example: Using librosa's Griffin-Lim algorithm for demonstration
    audio = librosa.feature.inverse.mel_to_audio(mel, sr=22050, n_fft=1024, hop_length=256)
    
    return audio

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'outputs/checkpoints/emotortoise_tts_epoch_100.pth'
    model = load_model(model_path, device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Example synthesis
    input_text = "I am thrilled to embark on this new journey!"
    emotion = 1  # Corresponding to 'Hasya' (joy)
    
    synthesized_audio = synthesize(input_text, emotion, model, tokenizer, device)
    
    # Save audio
    sf.write('outputs/results/synthesized_audio.wav', synthesized_audio, 22050)
    print("Audio synthesis complete. Saved to 'outputs/results/synthesized_audio.wav'")
