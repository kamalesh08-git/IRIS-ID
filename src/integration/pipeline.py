"""Integration pipeline for the recognition system.

This module provides `RecognitionPipeline` which integrates preprocessing
and recognition modules (injected as callables) to run the end-to-end
inference flow on single images or batches. It manages output paths,
timing, and basic failure handling.

Note: The actual implementations of face detection, alignment, mask
detection, periocular cropping, embedding generation, and FAISS matching
are expected to be provided by other team members and injected as
callables to this pipeline.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Structured prediction result returned by the pipeline."""

    image_path: str
    name: Optional[str]
    similarity: Optional[float]
    status: str
    bounding_boxes: Optional[List[List[int]]]
    elapsed_ms: float
    metadata: Dict[str, Any]


class RecognitionPipeline:
    """Recognition pipeline orchestrator.

    The pipeline is deliberately implementation-agnostic: concrete
    preprocessing and recognition functions are passed in by the caller.

    Args:
        preprocess_fn: Callable that takes an image path and returns a
            preprocessed artifact (could be image array, cropped image
            path, or other structure) required by `recognize_fn`.
        recognize_fn: Callable that takes the output of `preprocess_fn`
            and returns a dictionary with keys like `name`,
            `similarity`, and `bounding_boxes`.
        results_dir: Directory where outputs (predictions) will be
            stored. Subfolders `predictions` and `reports` are created.
    """

    def __init__(
        self,
        preprocess_fn: Callable[[Path], Any],
        recognize_fn: Callable[[Any], Dict[str, Any]],
        results_dir: Optional[Path] = None,
    ) -> None:
        self.preprocess_fn = preprocess_fn
        self.recognize_fn = recognize_fn
        self.results_dir = Path(results_dir or Path.cwd() / "results")
        self.predictions_dir = self.results_dir / "predictions"
        self.reports_dir = self.results_dir / "reports"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Create results directories if they do not exist."""

        for p in (self.results_dir, self.predictions_dir, self.reports_dir):
            p.mkdir(parents=True, exist_ok=True)
            logger.debug("Ensured directory exists: %s", p)

    def run(self, image_path: Path) -> Prediction:
        """Run the full pipeline for a single image.

        Args:
            image_path: Path to the input image file.

        Returns:
            A `Prediction` dataclass with structured results.
        """

        start = time.time()
        image_path = Path(image_path)
        logger.info("Processing image: %s", image_path)

        try:
            pre = self.preprocess_fn(image_path)
            logger.debug("Preprocessing complete for %s", image_path)
        except Exception as exc:  # pragma: no cover - external module may raise
            logger.exception("Preprocessing failed for %s", image_path)
            elapsed_ms = (time.time() - start) * 1000.0
            pred = Prediction(
                image_path=str(image_path),
                name=None,
                similarity=None,
                status="preprocess_failed",
                bounding_boxes=None,
                elapsed_ms=elapsed_ms,
                metadata={"error": str(exc)},
            )
            self._persist_prediction(pred)
            return pred

        try:
            result = self.recognize_fn(pre)
            logger.debug("Recognition result for %s: %s", image_path, result)
        except Exception as exc:  # pragma: no cover - external module may raise
            logger.exception("Recognition failed for %s", image_path)
            elapsed_ms = (time.time() - start) * 1000.0
            pred = Prediction(
                image_path=str(image_path),
                name=None,
                similarity=None,
                status="recognition_failed",
                bounding_boxes=None,
                elapsed_ms=elapsed_ms,
                metadata={"error": str(exc)},
            )
            self._persist_prediction(pred)
            return pred

        elapsed_ms = (time.time() - start) * 1000.0
        pred = Prediction(
            image_path=str(image_path),
            name=result.get("name"),
            similarity=result.get("similarity"),
            status=result.get("status", "unknown"),
            bounding_boxes=result.get("bounding_boxes"),
            elapsed_ms=elapsed_ms,
            metadata={k: v for k, v in result.items() if k not in {"name", "similarity", "status", "bounding_boxes"}},
        )

        self._persist_prediction(pred)
        return pred

    def run_batch(self, image_paths: Iterable[Path]) -> List[Prediction]:
        """Run pipeline on a batch of images.

        This method is a simple convenience wrapper that collects predictions
        for every image and returns them as a list.
        """

        preds: List[Prediction] = []
        for p in image_paths:
            try:
                preds.append(self.run(Path(p)))
            except Exception:  # pragma: no cover - defensive
                logger.exception("Unexpected failure running pipeline on %s", p)
        return preds

    def _persist_prediction(self, pred: Prediction) -> None:
        """Persist the prediction as a JSON file under `predictions_dir`.

        The filename is generated from the input image stem and a timestamp
        to avoid collisions.
        """

        stem = Path(pred.image_path).stem
        filename = f"{stem}_{int(time.time() * 1000)}.json"
        out = self.predictions_dir / filename
        try:
            with out.open("w", encoding="utf-8") as fh:
                json.dump(asdict(pred), fh, ensure_ascii=False, indent=2)
            logger.debug("Saved prediction to %s", out)
        except Exception:
            logger.exception("Failed to save prediction to %s", out)


__all__ = ["RecognitionPipeline", "Prediction"]
