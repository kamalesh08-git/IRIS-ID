"""RetinaFace-based face detection for the Member 1 preprocessing pipeline.

This module provides a single-responsibility detector abstraction that loads an
InsightFace RetinaFace model, detects one or more faces, extracts five facial
landmarks, and returns structured face metadata for downstream alignment and
periocular cropping.

The class intentionally stays focused on detection only. It does not perform
feature extraction, embeddings, FAISS indexing, or recognition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except Exception as exc:  # pragma: no cover - runtime dependency is optional
    FaceAnalysis = None
    _FACEANALYSIS_IMPORT_ERROR = exc
else:
    _FACEANALYSIS_IMPORT_ERROR = None

logger = logging.getLogger(__name__)

ImageInput = Union[str, Path, np.ndarray]
BoundingBox = Tuple[float, float, float, float]
LandmarkPoint = Tuple[float, float]


@dataclass(frozen=True)
class FaceLandmarks:
    """Five key facial landmarks used by the alignment and cropping stages."""

    left_eye: LandmarkPoint
    right_eye: LandmarkPoint
    nose: LandmarkPoint
    left_mouth: LandmarkPoint
    right_mouth: LandmarkPoint


@dataclass(frozen=True)
class DetectedFace:
    """Container for one detected face and its geometry."""

    bbox: BoundingBox
    confidence: float
    landmarks: FaceLandmarks
    face_index: int


class RetinaFaceDetector:
    """Load and run RetinaFace to detect faces and five landmarks.

    The detector is intentionally decoupled from the rest of the preprocessing
    chain. It can consume either a filesystem path or a NumPy image array and
    always returns the original image with a list of structured detections.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        providers: Optional[Sequence[str]] = None,
        ctx_id: int = 0,
        det_size: Tuple[int, int] = (640, 640),
        output_dir: Union[str, Path] = "processed/detected",
        confidence_threshold: float = 0.5,
    ) -> None:
        """Initialize the detector and prepare the RetinaFace backend.

        Args:
            model_name: InsightFace model name such as ``buffalo_l``.
            providers: Optional backend provider order.
            ctx_id: CPU or GPU context id for the model.
            det_size: Detector input size used when preparing the model.
            output_dir: Directory for optional saved annotations.
            confidence_threshold: Minimum confidence to keep a detection.
        """
        if FaceAnalysis is None:
            raise RuntimeError(
                "InsightFace is required for RetinaFaceDetector. "
                f"Import failed with: {_FACEANALYSIS_IMPORT_ERROR}"
            )

        self.model_name = model_name
        self.providers = tuple(providers or ("CUDAExecutionProvider", "CPUExecutionProvider"))
        self.ctx_id = int(ctx_id)
        self.det_size = tuple(det_size)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.confidence_threshold = float(confidence_threshold)

        self.detector = FaceAnalysis(name=self.model_name, providers=list(self.providers))
        self.detector.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
        logger.info(
            "RetinaFaceDetector initialized with model=%s, providers=%s, det_size=%s",
            self.model_name,
            self.providers,
            self.det_size,
        )

    @staticmethod
    def _load_image(image_input: ImageInput) -> np.ndarray:
        """Load an image from a file path or a NumPy array."""
        if isinstance(image_input, np.ndarray):
            if image_input.ndim != 3:
                raise ValueError("Input NumPy image must be a 3-channel BGR/OpenCV array.")
            return image_input

        image_path = Path(image_input)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to decode image: {image_path}")

        return image

    @staticmethod
    def _validate_bbox(bbox: Sequence[float]) -> Optional[BoundingBox]:
        """Validate and normalize detector bounding-box output."""
        if len(bbox) < 4:
            return None

        try:
            x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
        except (TypeError, ValueError):
            return None

        if not all(np.isfinite([x1, y1, x2, y2])):
            return None

        if x2 <= x1 or y2 <= y1:
            return None

        return (x1, y1, x2, y2)

    @staticmethod
    def _extract_landmarks(face_data: dict) -> Optional[FaceLandmarks]:
        """Extract the required five landmarks from a detector result."""
        try:
            landmark_data = face_data.get("landmark_2d_106", face_data.get("kps"))
            if landmark_data is None:
                logger.warning("Detection response did not supply facial landmarks.")
                return None

            landmarks = np.asarray(landmark_data, dtype=np.float32)
            if landmarks.shape[0] < 5:
                logger.warning("RetinaFace returned fewer than five landmarks.")
                return None

            left_eye = tuple(map(float, landmarks[0].tolist()))
            right_eye = tuple(map(float, landmarks[1].tolist()))
            nose = tuple(map(float, landmarks[2].tolist()))
            left_mouth = tuple(map(float, landmarks[3].tolist()))
            right_mouth = tuple(map(float, landmarks[4].tolist()))

            return FaceLandmarks(
                left_eye=left_eye,
                right_eye=right_eye,
                nose=nose,
                left_mouth=left_mouth,
                right_mouth=right_mouth,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Unable to extract face landmarks: %s", exc)
            return None

    def _save_debug_annotation(
        self,
        image: np.ndarray,
        detected_faces: Sequence[DetectedFace],
        filename: str,
    ) -> None:
        """Save an optional annotated debug image for detection review."""
        annotated = image.copy()
        for detected in detected_faces:
            x1, y1, x2, y2 = detected.bbox
            cv2.rectangle(annotated, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{detected.confidence:.2f}",
                (int(round(x1)), max(0, int(round(y1)) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        output_path = self.output_dir / f"{filename}_detected.png"
        cv2.imwrite(str(output_path), annotated)
        logger.info("Saved detection annotation to %s", output_path.resolve())

    def detect(
        self,
        image_input: ImageInput,
        filename: Optional[str] = None,
        save_debug: bool = True,
    ) -> Tuple[Optional[np.ndarray], List[DetectedFace]]:
        """Detect all faces in an image and return structured results.

        Args:
            image_input: Image path or a NumPy image array.
            filename: Optional stem used when saving debug annotations.
            save_debug: Whether to save an annotated debug image.

        Returns:
            A tuple of ``(original_image, detected_faces)``. When no face is
            found, the list is empty and the original image is still returned.
        """
        logger.info("Starting RetinaFace detection.")
        image = self._load_image(image_input)
        if image is None or image.size == 0:
            logger.error("Input image is empty or unreadable.")
            return None, []

        try:
            face_results = self.detector.get(image)
        except Exception as exc:
            logger.error("RetinaFace inference failed: %s", exc)
            return image, []

        if not face_results:
            logger.info("No faces detected in the supplied image.")
            return image, []

        detected_faces: List[DetectedFace] = []
        for face_index, face_data in enumerate(face_results):
            try:
                bbox = self._validate_bbox(face_data.get("bbox", []))
                if bbox is None:
                    logger.warning("Face %s has an invalid bounding box; skipping.", face_index)
                    continue

                confidence = float(face_data.get("det_score", face_data.get("score", 0.0)))
                if confidence < self.confidence_threshold:
                    logger.info(
                        "Skipping face %s because confidence %.4f is below threshold %.4f",
                        face_index,
                        confidence,
                        self.confidence_threshold,
                    )
                    continue

                landmarks = self._extract_landmarks(face_data)
                if landmarks is None:
                    logger.warning("Face %s is missing usable landmarks; skipping.", face_index)
                    continue

                detected_faces.append(
                    DetectedFace(
                        bbox=bbox,
                        confidence=confidence,
                        landmarks=landmarks,
                        face_index=face_index,
                    )
                )
            except Exception as exc:
                logger.warning("Error processing face %s: %s", face_index, exc)
                continue

        logger.info("RetinaFace detected %s valid face(s).", len(detected_faces))

        if save_debug and detected_faces and filename is not None:
            self._save_debug_annotation(image, detected_faces, str(filename))

        return image, detected_faces


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.info("RetinaFaceDetector module ready.")
