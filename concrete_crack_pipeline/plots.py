from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def training_history(history, output_path: Path, title: str):
    if history is None: return
    h = history.history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(h.get("loss", []), label="Train"); axes[0].plot(h.get("val_loss", []), label="Validation"); axes[0].set_title(f"{title} loss"); axes[0].legend()
    axes[1].plot(h.get("accuracy", []), label="Train"); axes[1].plot(h.get("val_accuracy", []), label="Validation"); axes[1].set_title(f"{title} accuracy"); axes[1].legend()
    fig.tight_layout(); fig.savefig(output_path, dpi=200); plt.close(fig)


def confusion_matrix(cm, title, output_path):
    fig = plt.figure(figsize=(5, 4)); sns.heatmap(np.asarray(cm), annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title); plt.tight_layout(); fig.savefig(output_path, dpi=300); plt.close(fig)


def roc(y_true, probabilities, model_name, output_path):
    fpr, tpr, _ = roc_curve(y_true, probabilities); auc = roc_auc_score(y_true, probabilities)
    fig = plt.figure(figsize=(6, 6)); plt.plot(fpr, tpr, label=f"AUC={auc:.4f}"); plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title(f"ROC - {model_name}"); plt.legend(); plt.tight_layout(); fig.savefig(output_path, dpi=300); plt.close(fig)


def precision_recall(y_true, probabilities, model_name, output_path):
    precision, recall, _ = precision_recall_curve(y_true, probabilities); ap = average_precision_score(y_true, probabilities)
    fig = plt.figure(figsize=(6, 6)); plt.plot(recall, precision, label=f"AP={ap:.4f}"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"Precision-Recall - {model_name}"); plt.legend(); plt.tight_layout(); fig.savefig(output_path, dpi=300); plt.close(fig)


def calibration(y_true, probabilities, title, output_path):
    bins = np.linspace(0, 1, 11); confidence, observed = [], []
    for i in range(len(bins)-1):
        mask = (probabilities >= bins[i]) & ((probabilities <= bins[i+1]) if i == len(bins)-2 else (probabilities < bins[i+1]))
        if mask.any(): confidence.append(probabilities[mask].mean()); observed.append(y_true[mask].mean())
    fig = plt.figure(figsize=(6, 6)); plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration"); plt.scatter(confidence, observed, s=60, label="Model"); plt.xlabel("Mean predicted probability"); plt.ylabel("Observed fraction positive"); plt.title(title); plt.legend(); plt.tight_layout(); fig.savefig(output_path, dpi=300); plt.close(fig)


def summary_plots(results_df, output_dir: Path):
    for metric in ["TestAccuracy", "F1", "MCC", "ROC_AUC", "PR_AUC"]:
        fig = plt.figure(figsize=(11, 6)); sns.boxplot(data=results_df, x="Model", y=metric); sns.stripplot(data=results_df, x="Model", y=metric, color="black", alpha=0.35); plt.xticks(rotation=30, ha="right"); plt.title(f"{metric} across outer folds"); plt.tight_layout(); fig.savefig(output_dir / f"internal_{metric}.png", dpi=300); plt.close(fig)
