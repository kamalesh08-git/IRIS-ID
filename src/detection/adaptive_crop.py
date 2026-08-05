"""
Adaptive face cropping utilities for masked face recognition.

This module provides an ``AdaptiveCropper`` class that accepts:
- a face image,
- a mask detection result,
- five facial landmarks,

and returns a cropped face image. When a mask is detected, the crop is
restricted to the upper facial region (forehead, eyebrows, eyes). When no mask
is detected, the class returns a full-face crop.

The cropping logic is landmark-driven rather than fixed-coordinate based, so
it remains robust across different faces and face sizes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ImageInput = Union[str, Path, np.ndarray]
LandmarkPoint = Tuple[float, float]
Landmarks5 = Sequence[LandmarkPoint]
CropBox = Tuple[int, int, int, int]


class AdaptiveCropper:
    """
    Landmark-driven adaptive cropper for masked and non-masked face images.

    The module expects the landmark order to be:
    1. left_eye
    2. right_eye
    3. nose
    4. left_mouth
    5. right_mouth
    """

    def __init__(self, output_dir: Union[str, Path] = "processed/cropped") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_image(image_input: ImageInput) -> np.ndarray:
        """Load an image from a path or a numpy array."""
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
    def _normalize_landmarks(landmarks: Landmarks5) -> np.ndarray:
        """Convert 5 landmarks to a floating-point (5, 2) numpy array."""
        if len(landmarks) != 5:
            raise ValueError("Exactly 5 landmarks are required for adaptive crop.")

        points = []
        for point in landmarks:
            if len(point) != 2:
                raise ValueError("Each landmark must be a (x, y) pair.")
            x, y = float(point[0]), float(point[1])
            if not (np.isfinite(x) and np.isfinite(y)):
                raise ValueError("Landmarks must contain finite coordinate values.")
            points.append((x, y))

        return np.array(points, dtype=np.float32)

    @staticmethod
    def _get_mask_detection_flag(mask_result: Union[bool, object]) -> bool:
        """Support a boolean or object-style mask detection result."""
        if isinstance(mask_result, bool):
            return mask_result
        if hasattr(mask_result, "mask_detected"):
            return bool(getattr(mask_result, "mask_detected"))
        if isinstance(mask_result, dict):
            return bool(mask_result.get("mask_detected", False))
        return False

    @staticmethod
    def _draw_crop_box(image: np.ndarray, crop_box: CropBox) -> np.ndarray:
        """Draw a crop rectangle on a copy of the image."""
        x1, y1, x2, y2 = crop_box
        annotated = image.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return annotated

    def _compute_landmark_bounds(self, landmarks: np.ndarray) -> CropBox:
        """Compute a conservative crop box from landmark extrema."""
        min_x = int(np.floor(np.min(landmarks[:, 0])))
        max_x = int(np.ceil(np.max(landmarks[:, 0])))
        min_y = int(np.floor(np.min(landmarks[:, 1])))
        max_y = int(np.ceil(np.max(landmarks[:, 1])))

        eye_left = landmarks[0]
        eye_right = landmarks[1]
        eye_distance = float(np.linalg.norm(eye_right - eye_left))
        margin = max(12, int(round(eye_distance * 0.35)))

        x1 = max(0, min_x - margin)
        y1 = max(0, min_y - margin)
        x2 = min(landmarks.shape[0] if False else 0, max_x + margin)
        y2 = max_y + margin
        return x1, y1, x2, y2

    def _mask_crop_box(self, image: np.ndarray, landmarks: np.ndarray) -> CropBox:
        """Create an upper-facial crop box for masked-face scenarios."""
        height, width = image.shape[:2]
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]

        eye_distance = float(np.linalg.norm(right_eye - left_eye))
        eye_y = (left_eye[1] + right_eye[1]) / 2.0
        nose_y = nose[1]

        # Use the eye row and nose position to keep the crop aligned with the
        # upper facial region while avoiding the lower mouth area.
        top_y = max(0, int(round(min(eye_y, nose_y) - eye_distance * 1.25)))
        bottom_y = min(height - 1, int(round(max(eye_y, nose_y) + eye_distance * 0.8)))

        left_x = max(0, int(round(min(left_eye[0], right_eye[0]) - eye_distance * 0.8)))
        right_x = min(width - 1, int(round(max(left_eye[0], right_eye[0]) + eye_distance * 0.8)))

        # Expand slightly to include forehead and eyebrow structure.
        top_y = max(0, top_y - int(round(eye_distance * 0.35)))
        left_x = max(0, left_x - int(round(eye_distance * 0.25)))
        right_x = min(width - 1, right_x + int(round(eye_distance * 0.25)))

        return left_x, top_y, right_x, bottom_y

    def _full_face_crop_box(self, image: np.ndarray, landmarks: np.ndarray) -> CropBox:
        """Compute a full-face crop box using landmarks and face geometry."""
        height, width = image.shape[:2]
        eye_left = landmarks[0]
        eye_right = landmarks[1]
        nose = landmarks[2]
        left_mouth = landmarks[3]
        right_mouth = landmarks[4]

        eye_distance = float(np.linalg.norm(eye_right - eye_left))
        margin = max(15, int(round(eye_distance * 0.8)))

        left_x = int(round(min(eye_left[0], eye_right[0], left_mouth[0], right_mouth[0], nose[0]) - margin))
        right_x = int(round(max(eye_left[0], eye_right[0], left_mouth[0], right_mouth[0], nose[0]) + margin))
        top_y = int(round(min(eye_left[1], eye_right[1], nose[1], left_mouth[1], right_mouth[1]) - margin))
        bottom_y = int(round(max(eye_left[1], eye_right[1], nose[1], left_mouth[1], right_mouth[1]) + margin))

        left_x = max(0, left_x)
        top_y = max(0, top_y)
        right_x = min(width - 1, right_x)
        bottom_y = min(height - 1, bottom_y)
        return left_x, top_y, right_x, bottom_y

    def crop(
        self,
        image_input: ImageInput,
        mask_result: Union[bool, object],
        landmarks: Landmarks5,
        save_visualization: bool = True,
        filename: Optional[str] = None,
    ) -> np.ndarray:
        """
        Perform adaptive cropping on the given face image.

        Args:
            image_input: A path to an image or a numpy array.
            mask_result: Either a boolean or an object providing a
                ``mask_detected`` field.
            landmarks: Five facial landmarks in the expected order.
            save_visualization: If True, save original and cropped visual output.
            filename: Optional base filename for saved artifacts.

        Returns:
            np.ndarray: Cropped face image.
        """
        image = self._load_image(image_input)
        landmark_array = self._normalize_landmarks(landmarks)
        mask_detected = self._get_mask_detection_flag(mask_result)

        if mask_detected:
            crop_box = self._mask_crop_box(image, landmark_array)
            logger.info("Mask detected. Applying upper-face crop.")
        else:
            crop_box = self._full_face_crop_box(image, landmark_array)
            logger.info("No mask detected. Applying full-face crop.")

        x1, y1, x2, y2 = crop_box
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1] - 1, x2)
        y2 = min(image.shape[0] - 1, y2)

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid crop box computed from landmarks.")

        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            raise ValueError("Computed crop region is empty.")

        if save_visualization:
            stem = filename or (Path(str(image_input)).stem if isinstance(image_input, (str, Path)) else "adaptive_crop")
            original_path = self.output_dir / f"{Path(stem).stem}_original.png"
            crop_path = self.output_dir / f"{Path(stem).stem}_adaptive_crop.png"
            visual_path = self.output_dir / f"{Path(stem).stem}_original_to_crop.png"

            cv2.imwrite(str(original_path), image)
            cv2.imwrite(str(crop_path), cropped)

            original_with_box = self._draw_crop_box(image, crop_box)
            cropped_placeholder = self._draw_crop_box(
                cropped,
                (0, 0, cropped.shape[1] - 1, cropped.shape[0] - 1),
            )
            target_height = max(original_with_box.shape[0], cropped_placeholder.shape[0])
            target_width = original_with_box.shape[1] + cropped_placeholder.shape[1]

            original_canvas = cv2.resize(
                original_with_box,
                (target_width // 2, target_height),
                interpolation=cv2.INTER_LINEAR,
            )
            cropped_canvas = cv2.resize(
                cropped_placeholder,
                (target_width // 2, target_height),
                interpolation=cv2.INTER_LINEAR,
            )
            visual = np.concatenate([original_canvas, cropped_canvas], axis=1)
            cv2.imwrite(str(visual_path), visual)

            logger.info("Saved visualization and crop artifacts in %s", self.output_dir.resolve())

        logger.info("Adaptive crop complete. Output shape: %sx%s", cropped.shape[1], cropped.shape[0])
        return cropped


def adaptive_crop(
    image_input: ImageInput,
    mask_result: Union[bool, object],
    landmarks: Landmarks5,
    output_dir: Union[str, Path] = "processed/cropped",
    filename: Optional[str] = None,
) -> np.ndarray:
    """Convenience wrapper around :class:`AdaptiveCropper`."""
    cropper = AdaptiveCropper(output_dir=output_dir)
    return cropper.crop(
        image_input=image_input,
        mask_result=mask_result,
        landmarks=landmarks,
        save_visualization=True,
        filename=filename,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("AdaptiveCropper ready for landmark-based face cropping.")
