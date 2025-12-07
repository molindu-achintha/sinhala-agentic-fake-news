"""
Compute accuracy, precision, recall, F1.
"""
import argparse

def evaluate(model_path, test_data):
    print(f"Evaluating model {model_path} on {test_data}...")
    print("Accuracy: 0.85 (Dummy)")
    print("Precision: 0.82 (Dummy)")
    print("Recall: 0.88 (Dummy)")
    print("F1: 0.85 (Dummy)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test_data", required=True)
    args = parser.parse_args()
    
    evaluate(args.model, args.test_data)
