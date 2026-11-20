"""Shared evaluation: metrics, confusion matrix, MLflow logging, comparison table."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from . import config as C

COMPARISON = C.REPORTS_DIR / "model_comparison.json"


def metrics(y_true, y_pred) -> dict:
    per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    return {"macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            **{f"f1_{n}": float(v) for n, v in zip(C.LABELS, per_class)}}


def plot_cm(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2], normalize="true")
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set(xticks=range(3), yticks=range(3), xticklabels=C.LABELS, yticklabels=C.LABELS,
           xlabel="Predicted", ylabel="Actual (from stars)", title=title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:.0%}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def record(name: str, y_true, y_pred, params: dict | None = None, extra: dict | None = None) -> dict:
    """Score a model on the shared eval sample, log to MLflow, update comparison file."""
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    m = metrics(np.asarray(y_true), np.asarray(y_pred))
    if extra:
        m.update(extra)
    cm_path = C.FIGURES_DIR / f"cm_{name}.png"
    plot_cm(y_true, y_pred, name, cm_path)

    mlflow.set_tracking_uri(C.MLFLOW_URI)
    mlflow.set_experiment("review-sentiment")
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params or {})
        mlflow.log_metrics({k: v for k, v in m.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(str(cm_path))

    table = json.loads(COMPARISON.read_text()) if COMPARISON.exists() else {}
    table[name] = m
    COMPARISON.write_text(json.dumps(table, indent=2))
    print(f"[{name}] macro-F1 {m['macro_f1']:.3f} | F1 neg {m['f1_negative']:.3f} "
          f"neu {m['f1_neutral']:.3f} pos {m['f1_positive']:.3f}")
    return m
