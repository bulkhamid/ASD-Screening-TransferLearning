
# utils/metrics.py
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(y_true, y_prob, threshold=0.5, ranking_scores=None):
    """Compute binary metrics while allowing unsaturated scores for AUC."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    ranking_scores = y_prob if ranking_scores is None else np.asarray(ranking_scores)
    y_pred = (y_prob >= threshold).astype(int)
    return {
      "auc":   float(roc_auc_score(y_true, ranking_scores)) if len(set(y_true))>1 else 0.5,
      "accuracy": accuracy_score(y_true, y_pred),
      "precision": precision_score(y_true, y_pred, zero_division=0),
      "recall": recall_score(y_true, y_pred, zero_division=0),
      "f1": f1_score(y_true, y_pred, zero_division=0),
    }
