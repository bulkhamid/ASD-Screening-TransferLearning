
# utils/metrics.py
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(y_true, y_prob):
    y_pred = (y_prob>=0.5).astype(int)
    return {
      "auc":   float(roc_auc_score(y_true, y_prob)) if len(set(y_true))>1 else 0.5,
      "accuracy": accuracy_score(y_true, y_pred),
      "precision": precision_score(y_true, y_pred, zero_division=0),
      "recall": recall_score(y_true, y_pred, zero_division=0),
      "f1": f1_score(y_true, y_pred, zero_division=0),
    }
