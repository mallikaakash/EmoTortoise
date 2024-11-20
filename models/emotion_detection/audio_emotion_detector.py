# models/emotion_detection/audio_emotion_detector.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

class AudioEmotionDataset(Dataset):
    def __init__(self, features_path, labels_path):
        self.features = np.load(features_path)
        self.labels = np.load(labels_path)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class AudioEmotionClassifier(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, num_classes=9):
        super(AudioEmotionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_audio_emotion_detector(train_loader, model, criterion, optimizer, device):
    model.train()
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def evaluate_audio_emotion_detector(test_loader, model, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load datasets
    train_dataset = AudioEmotionDataset('data/processed/navarasa/X_train.npy', 'data/processed/navarasa/y_train.npy')
    test_dataset = AudioEmotionDataset('data/processed/navarasa/X_test.npy', 'data/processed/navarasa/y_test.npy')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model, loss, optimizer
    model = AudioEmotionClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20
    for epoch in range(epochs):
        train_audio_emotion_detector(train_loader, model, criterion, optimizer, device)
        accuracy = evaluate_audio_emotion_detector(test_loader, model, device)
        print(f"Epoch {epoch+1}/{epochs}, Test Accuracy: {accuracy:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), 'models/emotion_detection/audio_emotion_classifier.pth')
