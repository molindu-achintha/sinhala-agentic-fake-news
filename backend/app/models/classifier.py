"""
Trainable BERT-based classifier for binary/multilabel label(s).
"""
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class Classifier:
    def __init__(self, model_name_or_path: str = "xlm-roberta-base", num_labels: int = 2):
        self.model_name = model_name_or_path
        self.num_labels = num_labels
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, path: str = None):
        load_path = path if path else self.model_name
        print(f"Loading classifier from {load_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(load_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(load_path, num_labels=self.num_labels)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            # Placeholder for initial run if model not found
            self.model = None

    def save_model(self, path: str):
        if self.model:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
        else:
            print("No model to save.")

    def predict(self, texts: list[str]) -> list[dict]:
        if not self.model:
            # Dummy prediction for testing before training
            return [{"label": "needs_more_evidence", "prob": 0.5} for _ in texts]
        
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        results = []
        labels = ["fake", "true"] # Example labels
        if self.num_labels > 2:
            labels = ["true", "false", "partially_false", "misleading", "needs_more_evidence"]

        probs = probabilities.cpu().numpy()
        for p in probs:
            idx = np.argmax(p)
            results.append({
                "label": labels[idx] if idx < len(labels) else "unknown",
                "prob": float(p[idx])
            })
        return results
