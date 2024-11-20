# models/emotion_detection/text_emotion_detector.py

from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

class TextEmotionDetector:
    def __init__(self, model_name='bert-base-uncased', num_labels=9):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to('cuda')
    
    def predict_emotion(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1)
            predicted = torch.argmax(probabilities, dim=1).item()
        return predicted  # Integer representing emotion class
