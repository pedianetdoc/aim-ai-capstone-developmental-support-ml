"""Evaluation helpers for classification and threshold analysis."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score,
    recall_score, accuracy_score, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict

SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "f1": "f1",
    "precision": "precision",
    "sensitivity": "recall",
    "accuracy": "accuracy",
}

def cv_summary(pipe, X, y, folds=5, random_state=42):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    scores = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    return {
        "ROC-AUC": scores["test_roc_auc"].mean(),
        "PR-AUC": scores["test_pr_auc"].mean(),
        "F1": scores["test_f1"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Sensitivity": scores["test_sensitivity"].mean(),
        "Accuracy": scores["test_accuracy"].mean(),
    }

def out_of_fold_probabilities(pipe, X, y, folds=5, random_state=42):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    return cross_val_predict(
        pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]

def metrics_at_threshold(y_true, probability, threshold):
    pred = (np.asarray(probability) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return {
        "threshold": threshold,
        "sensitivity": recall_score(y_true, pred, zero_division=0),
        "specificity": specificity,
        "precision": precision_score(y_true, pred, zero_division=0),
        "NPV": npv,
        "F1": f1_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
        "flagged_rate": pred.mean(),
    }

def threshold_table(y_true, probability, thresholds=(.20,.25,.30,.35,.40,.45,.50)):
    return pd.DataFrame([
        metrics_at_threshold(y_true, probability, t) for t in thresholds
    ])
