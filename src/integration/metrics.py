"""Metrics utilities for evaluation and reporting.

Provides the `Metrics` class which computes common classification
metrics (accuracy, precision, recall, F1), confusion matrix, and
optionally ROC-AUC. Results can be exported to CSV and the confusion
matrix saved as a PNG image.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Metrics:
    """Compute and persist evaluation metrics.

    Example:
        m = Metrics()
        results = m.compute(y_true, y_pred, labels=labels, y_score=y_score)
        m.save_metrics_csv(results, out_path)
        m.save_confusion_matrix(results['confusion_matrix'], labels, out_path_png)
    """

    def compute(
        self,
        y_true: Iterable[Any],
        y_pred: Iterable[Any],
        labels: Optional[List[Any]] = None,
        y_score: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute metrics from ground truth and predictions.

        Args:
            y_true: Ground-truth labels (iterable).
            y_pred: Predicted labels (iterable).
            labels: Optional list of label names/order for the confusion matrix.
            y_score: Optional prediction scores/probabilities used for ROC-AUC.

        Returns:
            A dictionary containing metrics and the confusion matrix.
        """

        try:
            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
                confusion_matrix,
                roc_auc_score,
            )
        except Exception as exc:  # pragma: no cover - dependency issue
            logger.exception("sklearn is required for Metrics.compute: %s", exc)
            raise

        y_true_list = list(y_true)
        y_pred_list = list(y_pred)

        acc = accuracy_score(y_true_list, y_pred_list)
        prec = precision_score(y_true_list, y_pred_list, average="macro", zero_division=0)
        rec = recall_score(y_true_list, y_pred_list, average="macro", zero_division=0)
        f1 = f1_score(y_true_list, y_pred_list, average="macro", zero_division=0)

        cm = confusion_matrix(y_true_list, y_pred_list, labels=labels)

        out: Dict[str, Any] = {
            "accuracy": float(acc),
            "precision_macro": float(prec),
            "recall_macro": float(rec),
            "f1_macro": float(f1),
            "confusion_matrix": cm,
            "labels": list(labels) if labels is not None else None,
        }

        # ROC-AUC: only compute if y_score is provided
        if y_score is not None:
            try:
                # If y_score is 1D and binary labels, compute directly
                if y_score.ndim == 1:
                    auc = roc_auc_score(y_true_list, y_score)
                else:
                    # multiclass: use ovo/ovr (ovr default)
                    auc = roc_auc_score(y_true_list, y_score, multi_class="ovr")
                out["roc_auc"] = float(auc)
            except Exception:
                logger.exception("Failed to compute ROC-AUC; skipping")

        return out

    def save_metrics_csv(self, metrics: Dict[str, Any], out_file: Path) -> Path:
        """Persist scalar metrics to a CSV file.

        The CSV will contain two columns: `metric` and `value`.
        """

        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with out_file.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["metric", "value"])
                for k, v in metrics.items():
                    if k == "confusion_matrix":
                        continue
                    if isinstance(v, (list, tuple, np.ndarray)):
                        writer.writerow([k, np.array(v).tolist()])
                    else:
                        writer.writerow([k, v])
            logger.info("Saved metrics CSV to %s", out_file)
        except Exception:
            logger.exception("Failed to save metrics CSV to %s", out_file)
            raise
        return out_file

    def save_confusion_matrix(self, cm: np.ndarray, labels: Optional[List[Any]], out_file: Path) -> Path:
        """Save a confusion matrix visualization to `out_file` (PNG).

        Args:
            cm: Confusion matrix as a 2D ndarray.
            labels: Optional list of labels corresponding to matrix axes.
            out_file: Path to write the PNG image.
        """

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - dependency issue
            logger.exception("matplotlib is required to save confusion matrix: %s", exc)
            raise

        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        if labels is None:
            labels = [str(i) for i in range(cm.shape[0])]

        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=labels,
            yticklabels=labels,
            ylabel="True label",
            xlabel="Predicted label",
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        fmt = "d"
        thresh = cm.max() / 2.0 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(int(cm[i, j]), fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()
        try:
            fig.savefig(out_file)
            logger.info("Saved confusion matrix to %s", out_file)
        except Exception:
            logger.exception("Failed to save confusion matrix to %s", out_file)
            raise
        finally:
            plt.close(fig)

        return out_file


__all__ = ["Metrics"]
