"""
Normalize and preprocess dataset.
"""
import pandas as pd
import json
import os
import sys

# Add backend to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend')))
from app.utils.text_normalize import normalize_text
from app.utils.sin_tokenizer import split_sentences

def preprocess():
    input_file = 'data/dataset/labeled.csv'
    output_file = 'data/dataset/processed.jsonl'
    
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    df = pd.read_csv(input_file)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            text = normalize_text(row['text'])
            claim = normalize_text(row['claim'])
            
            # Simple sentence splitting on text
            sentences = split_sentences(text)
            
            doc = {
                "id": row['id'],
                "text": text,
                "sentences": sentences,
                "claim": claim,
                "label": row['label'],
                "sources": row['sources'], # Assuming stringified json
                "pub_date": row['pub_date']
            }
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    preprocess()
