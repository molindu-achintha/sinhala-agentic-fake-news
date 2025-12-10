"""
Classifier using HuggingFace Inference API.
Uses Zero-Shot classification as a flexible baseline.
"""
import requests
import os
from tenacity import retry, stop_after_attempt, wait_exponential

class Classifier:
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        # Defaulting to a strong zero-shot model, but user can override via env
        # Note: 'xlm-roberta-base' is a base model, for classification logic via API 
        # we often use a dedicated endpoint or zero-shot if we don't have a fine-tuned model served.
        # Here we will assume the user wants to use a zero-shot approach or a specific fine-tuned endpoint.
        self.api_url = f"https://router.huggingface.co/models/{model_name}"
        # We need to access settings differently here to avoid circular imports if any, 
        # but sticking to standard pattern:
        from ..config import get_settings
        settings = get_settings()
        self.api_key = settings.HF_API_KEY
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def predict_zero_shot(self, text: str, candidate_labels: list[str]) -> dict:
        if not self.api_key:
             return {"labels": candidate_labels, "scores": [1.0/len(candidate_labels)]*len(candidate_labels)}

        payload = {
            "inputs": text,
            "parameters": {"candidate_labels": candidate_labels},
             "options": {"wait_for_model": True}
        }
        
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        
        if response.status_code != 200:
             # Fallback or error
             print(f"Classifier API Error: {response.text}")
             return {"labels": candidate_labels, "scores": [0.0]*len(candidate_labels)}
             
        return response.json()

    def predict(self, texts: list[str]) -> list[dict]:
        """
        Wrapper to match previous interface. 
        Note: HF Inference API 'predict' usually takes one input at a time for zero-shot 
        or a list for standard classification.
        """
        results = []
        labels = ["fake", "true", "misleading", "needs_more_evidence"]
        
        for text in texts:
            # We use zero-shot as a proxy for our logic since we don't have a trained custom model endpoint
            api_res = self.predict_zero_shot(text, labels)
            
            # Extract top result
            if "labels" in api_res and "scores" in api_res:
                top_label = api_res["labels"][0]
                top_score = api_res["scores"][0]
                results.append({"label": top_label, "prob": top_score})
            else:
                results.append({"label": "unknown", "prob": 0.0})
                
        return results

    def load_model(self, path=None):
        pass # No-op for API

    def save_model(self, path):
        pass # No-op for API
