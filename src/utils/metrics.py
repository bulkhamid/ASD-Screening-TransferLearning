"""
Metrics computation utilities for model evaluation.

This module provides functions to compute various classification metrics
including AUC, accuracy, precision, recall, and F1 score.
"""
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def compute_metrics(y_true, y_prob, threshold=0.5):
    """
    Compute classification metrics from true labels and predicted probabilities.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities in range [0, 1]
        threshold: Decision threshold for converting probabilities to predictions
    
    Returns:
        Dictionary containing:
            - auc: Area under the ROC curve
            - accuracy: Classification accuracy
            - precision: Precision score
            - recall: Recall score
            - f1: F1 score
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    # Handle edge case where all labels are the same class
    if len(set(y_true)) > 1:
        auc_score = float(roc_auc_score(y_true, y_prob))
    else:
        auc_score = 0.5  # Random classifier performance
    
    return {
        "auc": auc_score,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

