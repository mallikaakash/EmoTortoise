# scripts/evaluate.py

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, accuracy_score
from models.emotortoise_synthesizer import EmoTortoiseSynthesizer
from models.emotion_detection.audio_emotion_detector import AudioEmotionClassifier
import numpy as np

class EvaluationDataset(Dataset):
    def __init__(self, texts, emotions, mel_spectrograms, tokenizer, max_length=50):
        self.texts = texts
        self.emotions = emotions
        self.mel_spectrograms = mel_spectrograms
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        emotion = self.emotions[idx]
        mel = self.mel_spectrograms[idx]
        
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        input_ids = encoding['input_ids'].squeeze(0).float()
        return input_ids, torch.tensor(emotion, dtype=torch.long), torch.tensor(mel, dtype=torch.float32)

def evaluate_model(model, emotion_detector, dataset, tokenizer, device):
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    model.eval()
    emotion_detector.eval()
    
    mcd_scores = []
    all_true_emotions = []
    all_pred_emotions = []
    
    with torch.no_grad():
        for input_ids, emotions_batch, mels in dataloader:
            input_ids = input_ids.to(device)
            emotions_batch = emotions_batch.to(device)
            mels = mels.to(device)
            
            mel_preds = model(input_ids.unsqueeze(0), emotions_batch)
            mel_pred = mel_preds.squeeze(0).cpu().numpy()
            mel_true = mels.cpu().numpy()
            
            # Calculate Mel Cepstral Distortion (MCD)
            mcd = mean_squared_error(mel_true.flatten(), mel_pred.flatten())
            mcd_scores.append(mcd)
            
            # Convert mel-spectrogram to audio (placeholder)
            synthesized_audio = librosa.feature.inverse.mel_to_audio(mel_pred, sr=22050, n_fft=1024, hop_length=256)
            
            # Predict emotion from synthesized audio
            # Save synthesized audio temporarily if needed or modify emotion detector to accept mel-spectrograms directly
            # Placeholder: Assume emotion_detector can handle mel-spectrogram directly
            # Here, we skip this step for simplicity
            # all_pred_emotions.extend(predicted_emotions)
            # all_true_emotions.extend(emotions_batch.cpu().numpy())
    
    avg_mcd = np.mean(mcd_scores)
    # emotion_accuracy = accuracy_score(all_true_emotions, all_pred_emotions)
    # return avg_mcd, emotion_accuracy
    return avg_mcd

if __name__ == "__main__":
    import librosa
    from transformers import BertTokenizer
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    model_path = 'outputs/checkpoints/emotortoise_tts_epoch_100.pth'
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
    
    emotion_detector = AudioEmotionClassifier()
    emotion_detector.load_state_dict(torch.load('models/emotion_detection/audio_emotion_classifier.pth', map_location=device))
    emotion_detector.to(device)
    emotion_detector.eval()
    
    # Load evaluation data
    # Placeholder: Replace with actual evaluation data
    eval_texts = ["Sample evaluation text 1", "Sample evaluation text 2", ...]
    eval_emotions = [0, 1, 2, ...]
    eval_mels = [np.random.randn(80) for _ in range(len(eval_texts))]  # Replace with actual mel-spectrograms
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    eval_dataset = EvaluationDataset(eval_texts, eval_emotions, eval_mels, tokenizer)
    
    # Evaluate
    avg_mcd = evaluate_model(model, emotion_detector, eval_dataset, tokenizer, device)
    print(f"Average Mel Cepstral Distortion (MCD): {avg_mcd:.4f}")
    
    # Note: Implement emotion_accuracy calculation if emotion prediction from synthesized audio is available
