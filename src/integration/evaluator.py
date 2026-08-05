"""Evaluator orchestration for running dataset evaluations.

The `Evaluator` class runs a recognition `pipeline` against a labelled
dataset, collects predictions, computes metrics using `Metrics`, saves
prediction CSVs, metrics CSV, and confusion matrix images. It also
optionally annotates and saves images using `ResultVisualizer`.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from .metrics import Metrics
from .pipeline import Prediction, RecognitionPipeline
from .visualizer import ResultVisualizer

logger = logging.getLogger(__name__)


class Evaluator:
    """Run evaluation over a labelled dataset and produce reports.

    Args:
        pipeline: An instance of `RecognitionPipeline` used to produce
            predictions.
        results_dir: Base results directory (predictions/reports will be
            created beneath it).
        visualizer: Optional `ResultVisualizer` to annotate and save images.
    """

    def __init__(self, pipeline: RecognitionPipeline, results_dir: Optional[Path] = None, visualizer: Optional[ResultVisualizer] = None) -> None:
        self.pipeline = pipeline
        self.results_dir = Path(results_dir or pipeline.results_dir)
        self.predictions_dir = self.results_dir / "predictions"
        self.reports_dir = self.results_dir / "reports"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.visualizer = visualizer
        self.metrics = Metrics()

    def evaluate(self, dataset: Iterable[Tuple[Path, Any]], annotate: bool = False) -> List[dict]:
        """Evaluate the pipeline on a labelled dataset.

        Args:
            dataset: Iterable of tuples `(image_path, true_label)`.
            annotate: If True and a visualizer is provided, save annotated images.

        Returns:
            A list of per-image result dictionaries.
        """

        results: List[dict] = []
        for img_path, true_label in dataset:
            try:
                pred = self.pipeline.run(Path(img_path))
            except Exception:
                logger.exception("Pipeline failed on %s", img_path)
                pred = Prediction(
                    image_path=str(img_path),
                    name=None,
                    similarity=None,
                    status="pipeline_error",
                    bounding_boxes=None,
                    elapsed_ms=0.0,
                    metadata={},
                )

            record = {
                "image_path": str(img_path),
                "true_label": true_label,
                "predicted_label": pred.name,
                "similarity": pred.similarity,
                "status": pred.status,
                "elapsed_ms": pred.elapsed_ms,
            }
            results.append(record)

            # Optionally annotate
            if annotate and self.visualizer is not None:
                try:
                    out_annotated = self.visualizer.annotate_and_save(Path(img_path), pred, display=False)
                    logger.debug("Annotated image saved to %s", out_annotated)
                except Exception:
                    logger.exception("Failed to annotate image %s", img_path)

        return results

    def export_predictions_csv(self, results: Sequence[dict], out_file: Optional[Path] = None) -> Path:
        """Export per-image predictions to CSV.

        Columns: image_path,true_label,predicted_label,similarity,status,elapsed_ms
        """

        out_file = Path(out_file or (self.predictions_dir / "predictions.csv"))
        out_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with out_file.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["image_path", "true_label", "predicted_label", "similarity", "status", "elapsed_ms"])
                for r in results:
                    writer.writerow([r.get("image_path"), r.get("true_label"), r.get("predicted_label"), r.get("similarity"), r.get("status"), r.get("elapsed_ms")])
            logger.info("Saved predictions CSV to %s", out_file)
        except Exception:
            logger.exception("Failed to save predictions CSV to %s", out_file)
            raise
        return out_file

    def generate_reports(self, results: Sequence[dict], labels: Optional[List[Any]] = None, scores: Optional[List[float]] = None) -> dict:
        """Compute metrics and save metrics CSV and confusion matrix.

        Args:
            results: Sequence of per-image result dicts as returned by `evaluate`.
            labels: Optional list of label names to order the confusion matrix.
            scores: Optional list/array of prediction scores for ROC-AUC.

        Returns:
            The metrics dictionary returned by `Metrics.compute`.
        """

        y_true = [r["true_label"] for r in results]
        y_pred = [r["predicted_label"] for r in results]

        metrics_dict = self.metrics.compute(y_true, y_pred, labels=labels, y_score=None)

        # Save metrics CSV
        metrics_csv = self.reports_dir / "metrics.csv"
        self.metrics.save_metrics_csv(metrics_dict, metrics_csv)

        # Save confusion matrix image
        cm = metrics_dict.get("confusion_matrix")
        if cm is not None:
            cm_png = self.reports_dir / "confusion_matrix.png"
            try:
                self.metrics.save_confusion_matrix(cm, labels, cm_png)
            except Exception:
                logger.exception("Failed to save confusion matrix to %s", cm_png)

        return metrics_dict


__all__ = ["Evaluator"]
