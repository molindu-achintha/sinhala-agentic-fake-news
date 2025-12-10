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

import glob

def preprocess():
    dataset_dir = 'data/dataset'
    output_file = 'data/dataset/processed.jsonl'
    
    csv_files = glob.glob(os.path.join(dataset_dir, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {dataset_dir}")
        return

    print(f"Found {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")
    
    total_docs = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for input_file in csv_files:
            print(f"Processing {input_file}...")
            try:
                df = pd.read_csv(input_file)
                # handle potential different column names or missing columns
                # We expect: text, claim, label. If claim missing, use text.
                
                for _, row in df.iterrows():
                    # Robust column access
                    text_raw = row.get('text', row.get('News Content', row.get('content', ''))) 
                    if not isinstance(text_raw, str): continue
                    
                    text = normalize_text(text_raw)
                    
                    claim_raw = row.get('claim', row.get('Claim', ''))
                    if not isinstance(claim_raw, str) or not claim_raw:
                         # fallback to text if claim not present
                         claim = text
                    else:
                        claim = normalize_text(claim_raw)
                    
                    # Normalize label
                    label = str(row.get('label', row.get('Label', 'unknown'))).lower()
                    
                    sentences = split_sentences(text)
                    
                    doc = {
                        "id": str(row.get('id', total_docs)), # Ensure ID present
                        "text": text,
                        "sentences": sentences,
                        "claim": claim,
                        "label": label,
                        "sources": str(row.get('sources', '[]')),
                        "pub_date": str(row.get('pub_date', ''))
                    }
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                    total_docs += 1
            except Exception as e:
                print(f"Error processing {input_file}: {e}")
    
    print(f"Processed {total_docs} documents saved to {output_file}")

if __name__ == "__main__":
    preprocess()
