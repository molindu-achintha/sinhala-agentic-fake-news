"""
Scoring utilities for reasoning agent.
"""

def calculate_confidence(source_scores: list[float], content_scores: list[float]) -> float:
    """
    Weighted average of scores.
    """
    if not source_scores and not content_scores:
        return 0.0
    
    total_score = sum(source_scores) * 0.4 + sum(content_scores) * 0.6
    count = len(source_scores) * 0.4 + len(content_scores) * 0.6
    
    if count == 0: return 0.0
    
    val = total_score / count
    return min(max(val, 0.0), 1.0)
