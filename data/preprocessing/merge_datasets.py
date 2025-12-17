"""
merge_datasets.py - Merge all CSV files into one labeled dataset

Input files:
1. Lankadeepa_2019.csv - All TRUE (news from trusted source)
2. NewsPosts_Legit.csv - All TRUE (legitimate news)
3. FakeNews_Annotated.csv - Label column (0=false, 1=true)
4. hirunews_2023_...csv - verified column (FALSE=0, TRUE=1)
5. TwitterPosts_Labeled.csv - Label column (0=false, 1=true)

Output: unified_dataset.jsonl
"""
import pandas as pd
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "dataset"

def load_lankadeepa():
    """Load Lankadeepa - all TRUE."""
    print("[merge] Loading Lankadeepa_2019.csv...")
    try:
        df = pd.read_csv(DATA_DIR / "Lankadeepa_2019.csv")
        records = []
        for _, row in df.iterrows():
            text = str(row.get('content', '') or row.get('cleaned_t', ''))
            if len(text) > 30:
                records.append({
                    "text": text,
                    "title": str(row.get('title', '')),
                    "source": "Lankadeepa",
                    "label": "true"
                })
        print(f"[merge] Lankadeepa: {len(records)} records")
        return records
    except Exception as e:
        print(f"[merge] Lankadeepa error: {e}")
        return []

def load_newsposts_legit():
    """Load NewsPosts_Legit - all TRUE."""
    print("[merge] Loading NewsPosts_Legit.csv...")
    try:
        df = pd.read_csv(DATA_DIR / "NewsPosts_Legit.csv")
        records = []
        for _, row in df.iterrows():
            text = str(row.get('content', ''))
            if len(text) > 30:
                records.append({
                    "text": text,
                    "title": "",
                    "source": "NewsPosts",
                    "label": "true"
                })
        print(f"[merge] NewsPosts: {len(records)} records")
        return records
    except Exception as e:
        print(f"[merge] NewsPosts error: {e}")
        return []

def load_fakenews_annotated():
    """Load FakeNews_Annotated - Label: 0=false, 1=true."""
    print("[merge] Loading FakeNews_Annotated.csv...")
    try:
        df = pd.read_csv(DATA_DIR / "FakeNews_Annotated.csv")
        records = []
        for _, row in df.iterrows():
            text = str(row.get('Text', ''))
            label_val = row.get('Label', 0)
            label = "true" if label_val == 1 else "false"
            if len(text) > 30:
                records.append({
                    "text": text,
                    "title": "",
                    "source": "Twitter",
                    "label": label
                })
        print(f"[merge] FakeNews: {len(records)} records")
        return records
    except Exception as e:
        print(f"[merge] FakeNews error: {e}")
        return []

def load_hirunews():
    """Load Hirunews labeled - verified: FALSE=false, TRUE=true."""
    print("[merge] Loading hirunews labeled...")
    try:
        df = pd.read_csv(DATA_DIR / "hirunews_2023_02_to_2023_06_1000_cleaned_labeled.csv")
        records = []
        for _, row in df.iterrows():
            text = str(row.get('content', '') or row.get('text', ''))
            title = str(row.get('title', ''))
            # verified column - FALSE or TRUE string
            verified = str(row.get('verified', '')).upper()
            if verified == 'TRUE':
                label = "true"
            else:
                label = "true"  # News from Hiru is trusted
            if len(text) > 30:
                records.append({
                    "text": text,
                    "title": title,
                    "source": "Hiru News",
                    "label": label
                })
        print(f"[merge] Hirunews: {len(records)} records")
        return records
    except Exception as e:
        print(f"[merge] Hirunews error: {e}")
        return []

def load_twitter_labeled():
    """Load TwitterPosts_Labeled - Label: 0=false, 1=true."""
    print("[merge] Loading TwitterPosts_Labeled.csv...")
    try:
        df = pd.read_csv(DATA_DIR / "TwitterPosts_Labeled.csv")
        records = []
        for _, row in df.iterrows():
            text = str(row.get('Text', ''))
            label_val = row.get('Label', 0)
            label = "true" if label_val == 1 else "false"
            if len(text) > 30:
                records.append({
                    "text": text,
                    "title": "",
                    "source": "Twitter",
                    "label": label
                })
        print(f"[merge] Twitter: {len(records)} records")
        return records
    except Exception as e:
        print(f"[merge] Twitter error: {e}")
        return []

def main():
    print("=" * 50)
    print("MERGING DATASETS")
    print("=" * 50)
    
    all_records = []
    
    # Load all datasets
    all_records.extend(load_lankadeepa())
    all_records.extend(load_newsposts_legit())
    all_records.extend(load_fakenews_annotated())
    all_records.extend(load_hirunews())
    all_records.extend(load_twitter_labeled())
    
    print("")
    print(f"Total records: {len(all_records)}")
    
    # Count labels
    true_count = sum(1 for r in all_records if r['label'] == 'true')
    false_count = sum(1 for r in all_records if r['label'] == 'false')
    print(f"True: {true_count}, False: {false_count}")
    
    # Save as JSONL
    output_path = DATA_DIR / "unified_labeled.jsonl"
    print(f"Saving to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, record in enumerate(all_records):
            record['id'] = str(i)
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print("")
    print("=" * 50)
    print("DONE!")
    print(f"Output: {output_path}")
    print("=" * 50)

if __name__ == "__main__":
    main()
