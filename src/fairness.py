"""Simple subgroup performance audit helpers."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, accuracy_score, confusion_matrix
)

def subgroup_metrics(y_true, probability, group, threshold=0.30):
    frame = pd.DataFrame({
        "y": np.asarray(y_true),
        "p": np.asarray(probability),
        "group": np.asarray(group),
    })

    rows = []
    for g, d in frame.groupby("group", dropna=False):
        pred = (d["p"] >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(d["y"], pred, labels=[0,1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) else np.nan

        rows.append({
            "group": g,
            "N": len(d),
            "prevalence": d["y"].mean(),
            "ROC-AUC": roc_auc_score(d["y"], d["p"]),
            "PR-AUC": average_precision_score(d["y"], d["p"]),
            "sensitivity": recall_score(d["y"], pred, zero_division=0),
            "specificity": specificity,
            "precision": precision_score(d["y"], pred, zero_division=0),
            "F1": f1_score(d["y"], pred, zero_division=0),
            "accuracy": accuracy_score(d["y"], pred),
            "flagged_rate": pred.mean(),
        })

    return pd.DataFrame(rows)
