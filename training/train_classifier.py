"""
Fine-tune a transformer classifier on processed.jsonl.
"""
import argparse
import os
import sys

# Stub for real training logic
def train_classifier(data_path, output_dir):
    print(f"Training classifier using data from {data_path}...")
    print("Loading base model...")
    # Real code would use HuggingFace Trainer
    # ...
    print(f"Saving model to {output_dir}...")
    # model.save_pretrained(output_dir)
    print("Training complete (stub).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to processed.jsonl")
    parser.add_argument("--output_dir", default="backend/app/models/checkpoint", help="Output directory")
    args = parser.parse_args()
    
    train_classifier(args.data, args.output_dir)
