"""
Face alignment utilities for a masked face recognition pipeline.

This module provides a reusable ``FaceAligner`` class that accepts an input
image and five facial landmarks, aligns the face so the eyes become
horizontal, crops the face region, resizes it to a canonical 112x112 shape,
and saves the aligned output inside ``processed/aligned/``.

Production-quality design notes:
- Robust input validation.
- Clean separation between image loading, geometric alignment, and saving.
- Graceful handling of invalid or malformed landmark input.
- Visualization support for before/after debugging and presentation.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ImageInput = Union[str, Path, np.ndarray]
LandmarkPoint = Tuple[float, float]
Landmarks5 = Sequence[LandmarkPoint]


class FaceAligner:
    """
    Align a face image using five key landmarks.

    The class assumes the landmark order is:
    1. left eye
    2. right eye
    3. nose
    4. left mouth corner
    5. right mouth corner

    The alignment pipeline:
    1. Validates the input image and landmark coordinates.
    2. Computes the eye angle and rotates the face so the eyes become
       horizontal.
    3. Crops the aligned face region using the transformed landmarks.
    4. Resizes to a 112x112 output image.
    5. Saves the aligned image to ``processed/aligned/``.
    6. Optionally writes visualization artifacts.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "processed/aligned",
        target_size: Tuple[int, int] = (112, 112),
        save_visualizations: bool = True,
    ) -> None:
        """
        Initialize the face aligner.

        Args:
            output_dir: Directory used for saving aligned images.
            target_size: Output image size in pixels.
            save_visualizations: If True, save before/after visualization.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_size = tuple(target_size)
        self.save_visualizations = save_visualizations

    @staticmethod
    def _to_numpy_points(landmarks: Landmarks5) -> np.ndarray:
        """Convert a sequence of landmarks to a shape- (5,2) float32 array."""
        if len(landmarks) != 5:
            raise ValueError("Exactly 5 landmarks are required.")

        points = []
        for point in landmarks:
            if len(point) != 2:
                raise ValueError("Each landmark must be a 2D coordinate pair.")

            x, y = float(point[0]), float(point[1])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("Landmark coordinates must be finite numbers.")
            points.append((x, y))

        return np.array(points, dtype=np.float32)

    @staticmethod
    def _load_image(image_input: ImageInput) -> np.ndarray:
        """Load an image from a file path or numpy array."""
        if isinstance(image_input, np.ndarray):
            if image_input.ndim != 3:
                raise ValueError("A provided NumPy image must be RGB/BGR with 3 channels.")
            return image_input

        image_path = Path(image_input)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to decode image: {image_path}")

        return image

    @staticmethod
    def _draw_landmarks(
        image: np.ndarray,
        landmarks: np.ndarray,
        *,
        color: Tuple[int, int, int] = (0, 255, 0),
        radius: int = 3,
    ) -> np.ndarray:
        """Return an image with landmarks drawn directly on it."""
        annotated = image.copy()
        for x, y in landmarks:
            cv2.circle(annotated, (int(round(x)), int(round(y))), radius, color, -1)
        return annotated

    def _visualize(
        self,
        source_image: np.ndarray,
        landmarks: np.ndarray,
        aligned_image: np.ndarray,
        rotated_image: Optional[np.ndarray],
        transformed_landmarks: Optional[np.ndarray],
        crop_box: Optional[Tuple[int, int, int, int]],
        filename: str,
    ) -> None:
        """Save before/after visualization artifacts for debugging."""
        if not self.save_visualizations:
            return

        before = self._draw_landmarks(source_image, landmarks, color=(0, 255, 0))
        before_path = self.output_dir / f"{filename}_before_alignment.png"
        cv2.imwrite(str(before_path), before)

        if rotated_image is not None and transformed_landmarks is not None:
            rotated_vis = self._draw_landmarks(
                rotated_image,
                transformed_landmarks,
                color=(0, 0, 255),
                radius=2,
            )
            if crop_box is not None:
                x1, y1, x2, y2 = crop_box
                cv2.rectangle(rotated_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            after_path = self.output_dir / f"{filename}_after_alignment.png"
            cv2.imwrite(str(after_path), rotated_vis)

        final_path = self.output_dir / f"{filename}_aligned_112x112.png"
        cv2.imwrite(str(final_path), aligned_image)

    def _compute_rotation_matrix(
        self,
        landmarks: np.ndarray,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """Compute a rotation matrix that makes the eye line horizontal."""
        left_eye = landmarks[0]
        right_eye = landmarks[1]

        eye_center = (left_eye + right_eye) * 0.5
        dx = float(right_eye[0] - left_eye[0])
        dy = float(right_eye[1] - left_eye[1])
        angle = math.degrees(math.atan2(dy, dx))

        rotation_matrix = cv2.getRotationMatrix2D(
            tuple(np.round(eye_center).astype(np.float32)),
            angle,
            1.0,
        )

        return rotation_matrix, angle, eye_center

    def align(
        self,
        image_input: ImageInput,
        landmarks: Landmarks5,
        filename: Optional[str] = None,
        save: bool = True,
        visualize: bool = True,
    ) -> np.ndarray:
        """
        Align a face image using five landmarks.

        Args:
            image_input: Original image path or image array.
            landmarks: Five-landmark sequence in the expected order.
            filename: Optional image stem used to save output artifacts.
            save: If True, save the aligned image.
            visualize: If True, save before/after visualization.

        Returns:
            np.ndarray: Aligned face image resized to 112x112.
        """
        source_image = self._load_image(image_input)
        landmark_array = self._to_numpy_points(landmarks)

        if source_image is None or source_image.size == 0:
            raise ValueError("Source image is empty or unreadable.")

        if landmark_array.shape != (5, 2):
            raise ValueError("Expected 5 landmark points in a (5, 2) shape.")

        height, width = source_image.shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError("Source image dimensions must be valid positive values.")

        rotation_matrix, angle, eye_center = self._compute_rotation_matrix(landmark_array)
        rotated = cv2.warpAffine(
            source_image,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        transformed_landmarks = cv2.transform(
            landmark_array.reshape(1, -1, 2),
            rotation_matrix,
        ).reshape(-1, 2)

        # Compute face crop region from transformed landmarks.
        min_x = int(np.floor(np.min(transformed_landmarks[:, 0])))
        max_x = int(np.ceil(np.max(transformed_landmarks[:, 0])))
        min_y = int(np.floor(np.min(transformed_landmarks[:, 1])))
        max_y = int(np.ceil(np.max(transformed_landmarks[:, 1])))

        left_eye = transformed_landmarks[0]
        right_eye = transformed_landmarks[1]
        eye_distance = float(np.linalg.norm(right_eye - left_eye))
        margin = max(20, int(round(eye_distance * 0.45)))

        x1 = max(0, min_x - margin)
        y1 = max(0, min_y - margin)
        x2 = min(rotated.shape[1], max_x + margin)
        y2 = min(rotated.shape[0], max_y + margin)

        crop_box = (x1, y1, x2, y2)
        face_crop = rotated[y1:y2, x1:x2]

        if face_crop.size == 0:
            raise ValueError("Computed crop region is empty after alignment.")

        aligned = cv2.resize(face_crop, self.target_size, interpolation=cv2.INTER_LINEAR)

        if save:
            stem = filename or Path(str(image_input)).stem if isinstance(image_input, (str, Path)) else "aligned_face"
            save_name = f"{Path(stem).stem}.png"
            output_path = self.output_dir / save_name
            cv2.imwrite(str(output_path), aligned)
            logger.info("Saved aligned face to %s", output_path.resolve())

        if visualize and self.save_visualizations:
            self._visualize(
                source_image,
                landmark_array,
                aligned,
                rotated,
                transformed_landmarks,
                crop_box,
                filename=Path(str(filename or save_name)).stem if save else "alignment",
            )

        logger.info(
            "Aligned face using landmarks with eye angle %.2f degrees. "
            "Output resized to %sx%s.",
            angle,
            self.target_size[0],
            self.target_size[1],
        )
        return aligned


def align_face(
    image_input: ImageInput,
    landmarks: Landmarks5,
    output_dir: Union[str, Path] = "processed/aligned",
    target_size: Tuple[int, int] = (112, 112),
    filename: Optional[str] = None,
    save: bool = True,
    visualize: bool = True,
) -> np.ndarray:
    """Convenience wrapper around :class:`FaceAligner`."""
    aligner = FaceAligner(output_dir=output_dir, target_size=target_size)
    return aligner.align(
        image_input=image_input,
        landmarks=landmarks,
        filename=filename,
        save=save,
        visualize=visualize,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("FaceAligner module initialized. Example usage: align_face(...)")
