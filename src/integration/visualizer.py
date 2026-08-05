"""Result visualization utilities for the recognition pipeline.

This module provides `ResultVisualizer` which draws bounding boxes,
labels (person name), similarity scores, and recognition status onto
images and saves or optionally displays the annotated images.

The visualizer is intentionally lightweight and depends on OpenCV for
drawing operations. If OpenCV is not available in the runtime, callers
should provide pre-annotated images or install OpenCV (`opencv-python`).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import cv2

from .pipeline import Prediction

logger = logging.getLogger(__name__)


class ResultVisualizer:
    """Draw prediction results on images and save/display them.

    Responsibilities
    - Draw bounding boxes (if present)
    - Draw the person's name and similarity score
    - Draw recognition status (e.g., `matched`, `no_match`, `preprocess_failed`)
    - Save annotated image to disk and optionally display it

    Args:
        color_map: Optional mapping for different statuses to BGR colors.
    """

    DEFAULT_COLORS = {
        "matched": (0, 200, 0),
        "no_match": (0, 0, 200),
        "unknown": (200, 200, 0),
        "preprocess_failed": (0, 0, 255),
        "recognition_failed": (0, 0, 255),
    }

    def __init__(self, color_map: Optional[dict] = None) -> None:
        self.color_map = {**self.DEFAULT_COLORS, **(color_map or {})}

    def _choose_color(self, status: Optional[str]) -> tuple:
        if status and status in self.color_map:
            return self.color_map[status]
        return (180, 180, 180)

    def annotate_and_save(
        self,
        image_path: Path,
        prediction: Prediction,
        out_path: Optional[Path] = None,
        display: bool = False,
        wait_ms: int = 1000,
    ) -> Path:
        """Annotate `image_path` with `prediction` and save the result.

        Args:
            image_path: Path to the original image to annotate.
            prediction: `Prediction` object from the pipeline.
            out_path: Optional path where to save the annotated image. If
                omitted, the file is saved under `results/annotated/` next
                to existing result folders.
            display: If True, attempt to display the image using OpenCV's
                GUI (may not work in headless environments).
            wait_ms: Milliseconds to wait when displaying the image.

        Returns:
            Path to the saved annotated image file.
        """

        image_path = Path(image_path)
        if out_path is None:
            out_dir = image_path.parent / "annotated_results"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{image_path.stem}_annotated{image_path.suffix}"

        img = cv2.imread(str(image_path))
        if img is None:
            logger.error("Failed to read image for annotation: %s", image_path)
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        color = self._choose_color(prediction.status)

        boxes: Iterable = prediction.bounding_boxes or []
        if isinstance(boxes, list):
            for box in boxes:
                try:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                except Exception:
                    logger.debug("Skipping invalid bounding box: %s", box)

        # Prepare label text
        label_parts: List[str] = []
        if prediction.name:
            label_parts.append(str(prediction.name))
        if prediction.similarity is not None:
            label_parts.append(f"{prediction.similarity:.3f}")
        label_parts.append(prediction.status or "")
        label = " | ".join([p for p in label_parts if p])

        # Put label at top-left of first bounding box or top-left of image
        if boxes:
            try:
                first = boxes[0]
                x, y = int(first[0]), max(0, int(first[1]) - 10)
            except Exception:
                x, y = 10, 30
        else:
            x, y = 10, 30

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(img.shape[1] / 800, 1.2))
        thickness = 2
        # Draw text background for readability
        (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(img, (x - 2, y - h - 4), (x + w + 2, y + 4), (0, 0, 0), -1)
        cv2.putText(img, label, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

        # Save
        try:
            cv2.imwrite(str(out_path), img)
            logger.info("Saved annotated image to %s", out_path)
        except Exception:
            logger.exception("Failed to write annotated image to %s", out_path)
            raise

        if display:
            try:
                cv2.imshow("Annotated", img)
                cv2.waitKey(wait_ms)
                cv2.destroyAllWindows()
            except Exception:
                logger.exception("Failed to display annotated image (likely headless environment)")

        return out_path


__all__ = ["ResultVisualizer"]
