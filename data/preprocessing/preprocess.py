"""
Enhanced preprocessing with NLP analysis:
- POS Tagging
- Named Entity Recognition
- Stemming
- Claim Detection
"""
import pandas as pd
import json
import os
import sys
import glob
from tqdm import tqdm

# Add backend to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend')))
from app.utils.text_normalize import normalize_text
from app.utils.sin_tokenizer import split_sentences
from app.utils.sinhala_nlp import get_sinhala_nlp, SinhalaNLP


def preprocess():
    dataset_dir = 'data/dataset'
    output_file = 'data/dataset/processed.jsonl'
    
    csv_files = glob.glob(os.path.join(dataset_dir, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {dataset_dir}")
        return

    print(f"Found {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")
    
    # Initialize NLP processor
    print("Initializing Sinhala NLP processor...")
    nlp = get_sinhala_nlp()
    print("NLP processor ready (POS Tagger, NER, Stemmer)")
    
    total_docs = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for input_file in csv_files:
            print(f"\n📄 Processing {os.path.basename(input_file)}...")
            try:
                df = pd.read_csv(input_file)
                
                for _, row in tqdm(df.iterrows(), total=len(df), desc="Documents"):
                    # Robust column access
                    text_raw = row.get('text', row.get('News Content', row.get('content', ''))) 
                    if not isinstance(text_raw, str) or not text_raw.strip(): 
                        continue
                    
                    # Basic normalization
                    text = normalize_text(text_raw)
                    
                    # NLP ANALYSIS 
                    
                    # Tokenize
                    tokens = nlp.tokenize(text)
                    
                    # POS Tagging
                    pos_tags = nlp.pos_tag(text)
                    
                    # Named Entity Recognition
                    entities = nlp.extract_entities(text)
                    
                    # Extract nouns and verbs
                    nouns = [word for word, tag in pos_tags if tag in ['NN', 'NNP']]
                    verbs = [word for word, tag in pos_tags if tag == 'VB']
                    
                    # Stem key nouns
                    stemmed_nouns = [nlp.stem(n) for n in nouns[:10]]  # Limit for efficiency
                    
                    # Detect claim indicators
                    claim_indicators = nlp.detect_claim_indicators(text)
                    has_claim = len(claim_indicators) > 0
                    
                    # Detect negation
                    has_negation = nlp.detect_negation(text)
                    
                    # Sentence splitting
                    sentences = split_sentences(text)
                    
                    # Handle claim column
                    claim_raw = row.get('claim', row.get('Claim', ''))
                    if not isinstance(claim_raw, str) or not claim_raw:
                        claim = text
                    else:
                        claim = normalize_text(claim_raw)
                    
                    # Normalize label
                    label = str(row.get('label', row.get('Label', 'unknown'))).lower()
                    
                    # Build enhanced document
                    doc = {
                        "id": str(row.get('id', total_docs)),
                        "text": text,
                        "sentences": sentences,
                        "claim": claim,
                        "label": label,
                        "sources": str(row.get('sources', '[]')),
                        "pub_date": str(row.get('pub_date', '')),
                        
                        # === NLP FEATURES ===
                        "nlp": {
                            "token_count": len(tokens),
                            "sentence_count": len(sentences),
                            "pos_tags": pos_tags[:20],  # Limit for storage
                            "entities": entities,
                            "nouns": nouns[:15],
                            "verbs": verbs[:10],
                            "stemmed_nouns": stemmed_nouns,
                            "has_claim_indicator": has_claim,
                            "claim_indicators": claim_indicators,
                            "has_negation": has_negation
                        }
                    }
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                    total_docs += 1
                    
            except Exception as e:
                print(f"Error processing {input_file}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\nProcessed {total_docs} documents with NLP features → {output_file}")
    print("   Features added: POS tags, Entities, Nouns, Verbs, Stems, Claim indicators")


if __name__ == "__main__":
    preprocess()
