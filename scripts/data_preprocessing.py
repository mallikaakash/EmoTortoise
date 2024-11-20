# scripts/data_preprocessing.py

import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_audio(file_path, sr=22050):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio

def extract_features(audio, sr=22050):
    # Extract Mel-Frequency Cepstral Coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

def preprocess_dataset(csv_path, audio_dir, output_dir):
    df = pd.read_csv(csv_path)
    features = []
    labels = []
    
    for index, row in df.iterrows():
        file_path = os.path.join(audio_dir, row['filename'])
        audio = load_audio(file_path)
        mfccs = extract_features(audio)
        features.append(mfccs)
        labels.append(row['emotion'])  # Ensure labels align with Navarasa
    
    feature_array = np.array(features)
    label_array = np.array(labels)
    
    # Save processed data
    np.save(os.path.join(output_dir, 'features.npy'), feature_array)
    np.save(os.path.join(output_dir, 'labels.npy'), label_array)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        feature_array, label_array, test_size=0.2, random_state=42, stratify=label_array
    )
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

if __name__ == "__main__":
    csv_path = 'data/raw/emotion_labels.csv'
    audio_dir = 'data/raw/audio_files/'
    output_dir = 'data/processed/navarasa/'
    os.makedirs(output_dir, exist_ok=True)
    preprocess_dataset(csv_path, audio_dir, output_dir)
