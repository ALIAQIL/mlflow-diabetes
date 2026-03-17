import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
import os


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Calcule toutes les metriques. Retourne un dict {nom: valeur}."""
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }


def plot_confusion_matrix(y_true, y_pred, run_name, output_dir) -> str:
    """Genere et sauvegarde la matrice de confusion. Retourne le chemin."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Sain (0)", "Diabetique (1)"])
    ax.set_yticklabels(["Sain (0)", "Diabetique (1)"])
    ax.set_xlabel("Predit")
    ax.set_ylabel("Reel")
    ax.set_title(f"Matrice de Confusion — {run_name}")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center", fontsize=18, color=color
            )
    path = os.path.join(output_dir, f"confusion_matrix_{run_name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    return path


def plot_roc_curve(y_true, y_proba, run_name, output_dir) -> str:
    """Genere et sauvegarde la courbe ROC. Retourne le chemin."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})"
    )
    ax.plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier"
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {run_name}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, f"roc_curve_{run_name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    return path
