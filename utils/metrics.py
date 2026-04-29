import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


def compute_metrics(y_true, y_pred, y_score=None):
    results = {}
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    results['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    results['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    results['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    results['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    results['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)

    if y_score is not None:
        try:
            num_classes = y_score.shape[1]
            y_true_onehot = np.eye(num_classes)[y_true]
            results['auc_ovr'] = roc_auc_score(y_true_onehot, y_score, average='macro', multi_class='ovr')
        except ValueError:
            results['auc_ovr'] = 0.0

    results['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

    return results
