"""
src/evaluation_helpers.py
=========================
Statistical evaluation metrics, diagnostic curves, and visualization
helpers for supervised classification and unsupervised clustering.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    brier_score_loss, confusion_matrix, roc_curve, precision_recall_curve
)


def evaluate_classifier(name, y_true, y_pred, y_prob=None):
    """
    Computes a standard evaluation suite for binary classification:
    Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score,
    Matthews Correlation Coefficient (MCC), ROC-AUC, PR-AUC, and Brier Score.

    Parameters
    ----------
    name : str
        Model descriptor for tabular reporting.
    y_true : array-like
        Ground-truth binary labels (1=Positive/Poor, 0=Negative/Good).
    y_pred : array-like
        Predicted binary class assignments.
    y_prob : array-like, optional
        Predicted posterior probability of positive class.

    Returns
    -------
    dict
        Dictionary of computed performance metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall (Sensitivity)": rec,
        "Specificity": spec,
        "F1-Score": f1,
        "MCC": mcc,
    }

    if y_prob is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)
        metrics["PR-AUC"] = average_precision_score(y_true, y_prob)
        metrics["Brier Score"] = brier_score_loss(y_true, y_prob)
    else:
        metrics["ROC-AUC"] = np.nan
        metrics["PR-AUC"] = np.nan
        metrics["Brier Score"] = np.nan

    return metrics


def plot_confusion_matrices(models_dict, X_test, y_test, save_path=None):
    """
    Plots normalized confusion matrices for comparative classification models.
    """
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models_dict.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        sns.heatmap(
            cm, annot=True, fmt='.2%', cmap='Blues', cbar=False, ax=ax,
            xticklabels=['Good (0)', 'Poor (1)'],
            yticklabels=['Good (0)', 'Poor (1)']
        )
        ax.set_title(f'Confusion Matrix: {name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_pr_comparison(models_prob_dict, y_test, save_path=None):
    """
    Plots comparative ROC and Precision-Recall diagnostic curves against theoretical baselines.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC discrimination curves
    for name, y_prob in models_prob_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        ax1.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', lw=2)

    ax1.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Baseline (AUC = 0.50)')
    ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax1.set_ylabel('True Positive Rate (Recall)', fontsize=11)
    ax1.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Precision-Recall curves (prevalence benchmark)
    baseline_pr = np.mean(y_test)
    for name, y_prob in models_prob_dict.items():
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        ax2.plot(recall, precision, label=f'{name} (PR-AUC = {pr_auc:.3f})', lw=2)

    ax2.axhline(y=baseline_pr, color='k', linestyle='--', lw=1.5,
                label=f'Prevalence Baseline ({baseline_pr:.2f})')
    ax2.set_xlabel('Recall (Sensitivity)', fontsize=11)
    ax2.set_ylabel('Precision (Positive Predictive Value)', fontsize=11)
    ax2.set_title('Precision-Recall (PR) Curves', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower left', frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_elbow_and_silhouette(k_range, inertias, sil_scores, save_path=None):
    """
    Plots Within-Cluster Sum of Squares (Inertia/Elbow) and mean Silhouette scores across cluster candidate counts.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(k_range, inertias, 'bo-', lw=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax1.set_ylabel('Inertia / WCSS', fontsize=11)
    ax1.set_title('K-Means: Elbow Method for Optimal k', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.plot(k_range, sil_scores, 'ro-', lw=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax2.set_ylabel('Mean Silhouette Score', fontsize=11)
    ax2.set_title('K-Means: Silhouette Score Analysis', fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

