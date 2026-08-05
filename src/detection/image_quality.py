"""
Image quality analysis for masked-face recognition preprocessing.

This module provides an ``ImageQualityAnalyzer`` that evaluates an input face
image using lightweight, reproducible computer-vision heuristics:
- blur score
- brightness
- contrast
- eye visibility
- resolution

It then produces:
- an overall quality score in the range ``[0, 1]``
- a coarse quality label: ``Excellent``, ``Good``, ``Fair``, or ``Poor``
- practical recommendations such as ``Increase lighting``, ``Move closer``,
  or ``Image blurry``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ImageInput = Union[str, Path, np.ndarray]


@dataclass
class QualityMetrics:
    """Structured quality metrics for a single image."""

    blur_score: float
    brightness: float
    contrast: float
    eye_visibility: float
    resolution: float
    overall_quality_score: float
    quality_level: str
    recommendations: List[str]

    def as_dict(self) -> Dict[str, Union[float, str, List[str]]]:
        """Serialize the metric payload to a dictionary."""
        return {
            "blur_score": float(self.blur_score),
            "brightness": float(self.brightness),
            "contrast": float(self.contrast),
            "eye_visibility": float(self.eye_visibility),
            "resolution": float(self.resolution),
            "overall_quality_score": float(self.overall_quality_score),
            "quality_level": self.quality_level,
            "recommendations": list(self.recommendations),
        }


class ImageQualityAnalyzer:
    """
    Lightweight face-image quality analyzer.

    The analyzer is intentionally heuristic-driven and designed for robust
    preprocessing decisions rather than exact photographic metrics.
    """

    def __init__(self, output_dir: Union[str, Path] = "processed/quality") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_image(image_input: ImageInput) -> np.ndarray:
        """Load an image from a file path or numpy array."""
        if isinstance(image_input, np.ndarray):
            image = image_input
        else:
            image_path = Path(image_input)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None or image.size == 0:
            raise ValueError("Input image is empty or unreadable.")

        return image

    @staticmethod
    def _compute_blur_score(image: np.ndarray) -> float:
        """Estimate blur. Lower is blurrier. Return a score in [0, 1]."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(np.var(laplacian))
        # Normalize the score in a rough way. Higher variance indicates sharper image.
        score = min(1.0, max(0.0, variance / 300.0))
        return round(score, 4)

    @staticmethod
    def _compute_brightness(image: np.ndarray) -> float:
        """Estimate brightness as mean luminance normalized to [0, 1]."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray)) / 255.0
        return round(brightness, 4)

    @staticmethod
    def _compute_contrast(image: np.ndarray) -> float:
        """Estimate contrast as normalized standard deviation of luminance."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = float(np.std(gray)) / 255.0
        return round(contrast, 4)

    @staticmethod
    def _compute_eye_visibility(image: np.ndarray) -> float:
        """Estimate eye visibility with a simple dark-pupil / bright-eye heuristic."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eyes_region = gray[int(gray.shape[0] * 0.25):int(gray.shape[0] * 0.55), :]
        if eyes_region.size == 0:
            return 0.0

        mean_intensity = float(np.mean(eyes_region)) / 255.0
        # A moderate mean intensity typically implies visible eyes in a usable face image.
        score = max(0.0, min(1.0, 1.0 - abs(mean_intensity - 0.45) * 2.0))
        return round(score, 4)

    @staticmethod
    def _compute_resolution(image: np.ndarray) -> float:
        """Normalize resolution to [0, 1] using a practical target size of 300x300."""
        height, width = image.shape[:2]
        area = width * height
        target_area = 300 * 300
        score = min(1.0, area / target_area)
        return round(score, 4)

    @staticmethod
    def _quality_level(score: float) -> str:
        """Convert the overall quality score into a label."""
        if score >= 0.8:
            return "Excellent"
        if score >= 0.6:
            return "Good"
        if score >= 0.4:
            return "Fair"
        return "Poor"

    @staticmethod
    def _recommendations(metrics: QualityMetrics) -> List[str]:
        """Generate human-readable recommendations from the measured metrics."""
        recommendations: List[str] = []

        if metrics.brightness < 0.35:
            recommendations.append("Increase lighting")
        elif metrics.brightness > 0.8:
            recommendations.append("Reduce lighting glare")

        if metrics.resolution < 0.4:
            recommendations.append("Move closer")

        if metrics.blur_score < 0.35:
            recommendations.append("Image blurry")

        if metrics.eye_visibility < 0.45:
            recommendations.append("Improve eye visibility")

        if metrics.contrast < 0.15:
            recommendations.append("Increase contrast")

        if not recommendations:
            recommendations.append("Image quality looks acceptable")

        return recommendations

    def analyze(self, image_input: ImageInput) -> QualityMetrics:
        """
        Analyze image quality and return a structured result.

        Args:
            image_input: Image path or loaded NumPy array.

        Returns:
            QualityMetrics: Structured quality metrics with score and recommendations.
        """
        image = self._load_image(image_input)
        stem = Path(str(image_input)).stem if isinstance(image_input, (str, Path)) else "quality_image"

        blur_score = self._compute_blur_score(image)
        brightness = self._compute_brightness(image)
        contrast = self._compute_contrast(image)
        eye_visibility = self._compute_eye_visibility(image)
        resolution = self._compute_resolution(image)

        # Weighted quality score. Higher is better.
        weights = {
            "blur": 0.30,
            "brightness": 0.20,
            "contrast": 0.15,
            "eye_visibility": 0.20,
            "resolution": 0.15,
        }

        overall_score = (
            blur_score * weights["blur"]
            + (1.0 - abs(brightness - 0.5) * 2.0) * weights["brightness"]
            + contrast * weights["contrast"]
            + eye_visibility * weights["eye_visibility"]
            + resolution * weights["resolution"]
        )
        overall_score = max(0.0, min(1.0, round(overall_score, 4)))

        metrics = QualityMetrics(
            blur_score=blur_score,
            brightness=brightness,
            contrast=contrast,
            eye_visibility=eye_visibility,
            resolution=resolution,
            overall_quality_score=overall_score,
            quality_level=self._quality_level(overall_score),
            recommendations=[],
        )
        metrics.recommendations = self._recommendations(metrics)

        result_path = self.output_dir / f"{stem}_quality.json"
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics.as_dict(), handle, indent=2)

        logger.info(
            "Quality analysis complete: score=%.4f, level=%s | saved=%s",
            metrics.overall_quality_score,
            metrics.quality_level,
            result_path,
        )
        return metrics


def analyze_image_quality(image_input: ImageInput, output_dir: Union[str, Path] = "processed/quality") -> QualityMetrics:
    """Convenience wrapper around :class:`ImageQualityAnalyzer`."""
    analyzer = ImageQualityAnalyzer(output_dir=output_dir)
    return analyzer.analyze(image_input)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("ImageQualityAnalyzer ready.")
