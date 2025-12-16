"""
Evaluation Metrics for Fake News Detection System.

Calculates accuracy, precision, recall, F1-score, and confusion matrix.
"""
from typing import List, Dict, Tuple
from collections import Counter
import json


class EvaluationMetrics:
    """
    Calculate and store evaluation metrics for the fake news detection system.
    """
    
    # Map various label formats to standardized labels
    LABEL_MAPPING = {
        'true': 'true',
        'real': 'true',
        'verified': 'true',
        'likely_true': 'true',
        'false': 'false',
        'fake': 'false',
        'likely_false': 'false',
        'misleading': 'misleading',
        'partially_true': 'misleading',
        'needs_verification': 'needs_verification',
        'needs_more_evidence': 'needs_verification',
        'unknown': 'needs_verification'
    }
    
    # Labels for classification
    LABELS = ['true', 'false', 'misleading', 'needs_verification']
    
    def __init__(self):
        self.predictions = []
        self.ground_truth = []
        self.raw_results = []
    
    def add_result(self, predicted: str, actual: str, metadata: Dict = None):
        """Add a single prediction result."""
        pred_normalized = self.LABEL_MAPPING.get(predicted.lower(), 'needs_verification')
        actual_normalized = self.LABEL_MAPPING.get(actual.lower(), 'needs_verification')
        
        self.predictions.append(pred_normalized)
        self.ground_truth.append(actual_normalized)
        self.raw_results.append({
            'predicted': predicted,
            'actual': actual,
            'predicted_normalized': pred_normalized,
            'actual_normalized': actual_normalized,
            'correct': pred_normalized == actual_normalized,
            'metadata': metadata or {}
        })
    
    def accuracy(self) -> float:
        """Calculate overall accuracy."""
        if not self.predictions:
            return 0.0
        correct = sum(1 for p, a in zip(self.predictions, self.ground_truth) if p == a)
        return correct / len(self.predictions)
    
    def precision(self, label: str = None) -> Dict[str, float]:
        """
        Calculate precision for each label or a specific label.
        Precision = TP / (TP + FP)
        """
        if label:
            tp = sum(1 for p, a in zip(self.predictions, self.ground_truth) if p == label and a == label)
            fp = sum(1 for p, a in zip(self.predictions, self.ground_truth) if p == label and a != label)
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        result = {}
        for lbl in self.LABELS:
            result[lbl] = self.precision(lbl)
        return result
    
    def recall(self, label: str = None) -> Dict[str, float]:
        """
        Calculate recall for each label or a specific label.
        Recall = TP / (TP + FN)
        """
        if label:
            tp = sum(1 for p, a in zip(self.predictions, self.ground_truth) if p == label and a == label)
            fn = sum(1 for p, a in zip(self.predictions, self.ground_truth) if p != label and a == label)
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        result = {}
        for lbl in self.LABELS:
            result[lbl] = self.recall(lbl)
        return result
    
    def f1_score(self, label: str = None) -> Dict[str, float]:
        """
        Calculate F1 score for each label or a specific label.
        F1 = 2 * (precision * recall) / (precision + recall)
        """
        if label:
            p = self.precision(label)
            r = self.recall(label)
            return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
        
        result = {}
        for lbl in self.LABELS:
            result[lbl] = self.f1_score(lbl)
        return result
    
    def macro_f1(self) -> float:
        """Calculate macro-averaged F1 score."""
        f1_scores = self.f1_score()
        return sum(f1_scores.values()) / len(f1_scores) if f1_scores else 0.0
    
    def weighted_f1(self) -> float:
        """Calculate weighted F1 score based on support."""
        f1_scores = self.f1_score()
        label_counts = Counter(self.ground_truth)
        total = len(self.ground_truth)
        
        weighted = sum(f1_scores.get(lbl, 0) * label_counts.get(lbl, 0) for lbl in self.LABELS)
        return weighted / total if total > 0 else 0.0
    
    def confusion_matrix(self) -> Dict[str, Dict[str, int]]:
        """Generate confusion matrix."""
        matrix = {actual: {pred: 0 for pred in self.LABELS} for actual in self.LABELS}
        
        for pred, actual in zip(self.predictions, self.ground_truth):
            if actual in matrix and pred in matrix[actual]:
                matrix[actual][pred] += 1
        
        return matrix
    
    def classification_report(self) -> Dict:
        """Generate full classification report."""
        precision = self.precision()
        recall = self.recall()
        f1 = self.f1_score()
        
        # Support (count of each actual label)
        support = Counter(self.ground_truth)
        
        report = {
            'accuracy': self.accuracy(),
            'macro_f1': self.macro_f1(),
            'weighted_f1': self.weighted_f1(),
            'total_samples': len(self.predictions),
            'per_class': {}
        }
        
        for label in self.LABELS:
            report['per_class'][label] = {
                'precision': precision.get(label, 0.0),
                'recall': recall.get(label, 0.0),
                'f1_score': f1.get(label, 0.0),
                'support': support.get(label, 0)
            }
        
        report['confusion_matrix'] = self.confusion_matrix()
        
        return report
    
    def print_report(self):
        """Print a formatted classification report."""
        report = self.classification_report()
        
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(f"\nTotal Samples: {report['total_samples']}")
        print(f"Overall Accuracy: {report['accuracy']:.4f} ({report['accuracy']*100:.1f}%)")
        print(f"Macro F1 Score: {report['macro_f1']:.4f}")
        print(f"Weighted F1 Score: {report['weighted_f1']:.4f}")
        
        print("\n" + "-" * 60)
        print(f"{'Label':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 60)
        
        for label, metrics in report['per_class'].items():
            print(f"{label:<20} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
                  f"{metrics['f1_score']:>10.4f} {metrics['support']:>10}")
        
        print("\n" + "-" * 60)
        print("CONFUSION MATRIX (rows=actual, cols=predicted)")
        print("-" * 60)
        
        # Header
        header = f"{'':>20}" + "".join(f"{lbl:>15}" for lbl in self.LABELS)
        print(header)
        
        cm = report['confusion_matrix']
        for actual in self.LABELS:
            row = f"{actual:>20}"
            for pred in self.LABELS:
                row += f"{cm[actual][pred]:>15}"
            print(row)
        
        print("=" * 60)
        
        return report
    
    def save_report(self, filepath: str):
        """Save report to JSON file."""
        report = self.classification_report()
        report['raw_results'] = self.raw_results
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Report saved to: {filepath}")
