"""
End-to-end detection pipeline for masked face recognition.

The pipeline processes one image at a time through the following stages:

DatasetLoader
    -> validate the file's dataset context and image readability.
RetinaFaceDetector
    -> detect faces and extract five facial landmarks.
FaceAligner
    -> rotate the face to make the eye line horizontal, crop, and resize to
       112x112.
MaskDetector
    -> classify the aligned face as ``Mask`` or ``No Mask``.
ImageQualityAnalyzer
    -> compute quality metrics and recommendations.
AdaptiveCropper
    -> crop the aligned face adaptively based on landmark geometry and the
       mask decision.

The pipeline does not perform recognition; it stays focused on preprocessing
and classification-only stages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from adaptive_crop import AdaptiveCropper
from alignment import FaceAligner
from dataset_loader import DatasetLoader, ImageMetadata
from image_quality import ImageQualityAnalyzer, QualityMetrics
from mask_detector import MaskDetector, MaskPrediction
from retinaface import DetectedFace, RetinaFaceDetector

logger = logging.getLogger(__name__)

ImageInput = Union[str, Path, np.ndarray]


@dataclass
class PipelineResult:
    """Structured result returned by the detection pipeline."""

    aligned_face: np.ndarray
    mask_status: MaskPrediction
    quality_metrics: QualityMetrics
    adaptive_crop: np.ndarray
    stage_paths: Dict[str, str]

    def as_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary for downstream integration."""
        return {
            "aligned_face": self.aligned_face,
            "mask_status": self.mask_status.as_dict(),
            "quality_metrics": self.quality_metrics.as_dict(),
            "adaptive_crop": self.adaptive_crop,
            "stage_paths": self.stage_paths,
        }


class DetectionPipeline:
    """
    One-image-at-a-time detection and preprocessing pipeline.

    This class coordinates the repository's detection and preprocessing stages
    around a single input image and preserves all intermediate artifacts in the
    repository's ``processed`` hierarchy.
    """

    def __init__(
        self,
        dataset_loader: Optional[DatasetLoader] = None,
        detector: Optional[RetinaFaceDetector] = None,
        aligner: Optional[FaceAligner] = None,
        mask_detector: Optional[MaskDetector] = None,
        quality_analyzer: Optional[ImageQualityAnalyzer] = None,
        cropper: Optional[AdaptiveCropper] = None,
    ) -> None:
        self.dataset_loader = dataset_loader or DatasetLoader()
        self.detector = detector or RetinaFaceDetector(output_dir="processed/detected")
        self.aligner = aligner or FaceAligner(output_dir="processed/aligned", target_size=(112, 112))
        self.mask_detector = mask_detector or MaskDetector(output_dir="processed/mask")
        self.quality_analyzer = quality_analyzer or ImageQualityAnalyzer(output_dir="processed/quality")
        self.cropper = cropper or AdaptiveCropper(output_dir="processed/cropped")

        self.stage_paths: Dict[str, str] = {}
        self._log_pipeline_start()

    @staticmethod
    def _log_pipeline_start() -> None:
        logger.info("Starting DetectionPipeline orchestration...")

    @staticmethod
    def _load_image(image_input: ImageInput) -> np.ndarray:
        """Load image input for a pipeline run."""
        if isinstance(image_input, np.ndarray):
            return image_input

        image_path = Path(image_input)
        if not image_path.exists():
            raise FileNotFoundError(f"Image path not found: {image_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to decode image: {image_path}")
        return image

    def _dataset_context(self, image_path: Path) -> Optional[str]:
        """Map the file to one of the known dataset roots when possible."""
        resolved = image_path.resolve()
        for dataset_name, dataset_path in self.dataset_loader.DATASET_PATHS.items():
            if resolved.is_relative_to(dataset_path.resolve()):
                return dataset_name
        return None

    def _extract_landmarks(self, detected_face: DetectedFace) -> List[Tuple[float, float]]:
        """Convert a detected face object's landmark structure into the required list."""
        landmarks = detected_face.landmarks
        if landmarks is None:
            raise ValueError("RetinaFace did not return usable landmarks for the detected face.")

        return [
            tuple(landmarks.left_eye),
            tuple(landmarks.right_eye),
            tuple(landmarks.nose),
            tuple(landmarks.left_mouth),
            tuple(landmarks.right_mouth),
        ]

    def process_image(
        self,
        image_input: ImageInput,
        filename: Optional[str] = None,
    ) -> PipelineResult:
        """
        Process a single image through the complete preprocessing pipeline.

        Args:
            image_input: Input image path or NumPy array.
            filename: Optional base filename for staged artifacts.

        Returns:
            PipelineResult: Final pipeline artifacts and structured metrics.
        """
        image_path = Path(image_input) if isinstance(image_input, (str, Path)) else None
        if image_path is not None:
            logger.info("Processing image: %s", image_path)
            dataset_name = self._dataset_context(image_path)
            if dataset_name:
                logger.info("Dataset context inferred: %s", dataset_name)
            else:
                logger.info("Image is outside the known dataset roots; continuing in single-image mode.")

        image = self._load_image(image_input)
        logger.info("Stage 1/6: DatasetLoader validation")
        if image_path is not None:
            metadata = self.dataset_loader._extract_image_metadata(image_path, dataset_name or "Unknown")
            if metadata is None:
                raise ValueError(f"DatasetLoader could not validate the image: {image_path}")
            self.stage_paths["dataset_loader"] = str(metadata.image_path)
        else:
            self.stage_paths["dataset_loader"] = "numpy-array-input"

        logger.info("Stage 2/6: RetinaFaceDetector detection")
        original_image, detected_faces = self.detector.detect(image)
        if original_image is None:
            raise RuntimeError("RetinaFaceDetector failed to load or decode the input image.")
        if not detected_faces:
            raise RuntimeError("RetinaFaceDetector did not detect any faces in the input image.")

        first_face = detected_faces[0]
        landmarks = self._extract_landmarks(first_face)
        logger.info(
            "Detected face with confidence %.4f and bbox=%s",
            first_face.confidence,
            first_face.bbox,
        )

        logger.info("Stage 3/6: FaceAligner")
        aligned_filename = filename or (image_path.stem if image_path is not None else "pipeline_image")
        aligned_face = self.aligner.align(
            image_input=original_image,
            landmarks=landmarks,
            filename=aligned_filename,
            save=True,
            visualize=True,
        )
        self.stage_paths["aligned_face"] = str(
            self.aligner.output_dir / f"{Path(aligned_filename).stem}.png"
        )

        logger.info("Stage 4/6: MaskDetector")
        mask_prediction = self.mask_detector.predict(
            image_input=aligned_face,
            save_annotated=True,
        )
        self.stage_paths["mask_prediction"] = str(
            self.mask_detector.output_dir / f"{Path(aligned_filename).stem}_mask_prediction.png"
        )

        logger.info("Stage 5/6: ImageQualityAnalyzer")
        quality_metrics = self.quality_analyzer.analyze(aligned_face)
        self.stage_paths["quality_metrics"] = str(
            self.quality_analyzer.output_dir / f"{Path(aligned_filename).stem}_quality.json"
        )

        logger.info("Stage 6/6: AdaptiveCropper")
        cropped = self.cropper.crop(
            image_input=aligned_face,
            mask_result=mask_prediction,
            landmarks=landmarks,
            save_visualization=True,
            filename=aligned_filename,
        )
        self.stage_paths["adaptive_crop"] = str(
            self.cropper.output_dir / f"{Path(aligned_filename).stem}_adaptive_crop.png"
        )

        logger.info("Pipeline complete. Returning aligned face, mask status, quality score, and adaptive crop.")
        return PipelineResult(
            aligned_face=aligned_face,
            mask_status=mask_prediction,
            quality_metrics=quality_metrics,
            adaptive_crop=cropped,
            stage_paths=self.stage_paths,
        )


def run_detection_pipeline(
    image_input: ImageInput,
    filename: Optional[str] = None,
) -> PipelineResult:
    """Convenience wrapper for pipeline execution."""
    pipeline = DetectionPipeline()
    return pipeline.process_image(image_input=image_input, filename=filename)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("DetectionPipeline module ready.")
