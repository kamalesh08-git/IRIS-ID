"""
Mask detection utilities for a masked face recognition pipeline.

This module exposes a ``MaskDetector`` class that:
- accepts a face image,
- classifies it as ``Mask`` or ``No Mask``,
- returns the binary decision, confidence score, and bounding box,
- saves an annotated visualization, and
- provides an ONNX-ready backend interface for a future pretrained model.

The current implementation intentionally prefers a clean interface-first design
so the repository can plug in a real ONNX checkpoint later without changing
calling code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None

logger = logging.getLogger(__name__)

ImageInput = Union[str, Path, np.ndarray]
BoundingBox = Tuple[int, int, int, int]


@dataclass
class MaskPrediction:
    """Structured output returned by the mask detector."""

    mask_detected: bool
    confidence: float
    bounding_box: BoundingBox
    label: str

    def as_dict(self) -> dict:
        """Serialize the prediction result to a plain dictionary."""
        return {
            "mask_detected": self.mask_detected,
            "confidence": float(self.confidence),
            "bounding_box": list(self.bounding_box),
            "label": self.label,
        }


class MaskDetector:
    """
    Mask detector wrapper with an optional ONNX inference backend.

    The detector is intentionally designed to classify only two classes:
    - ``Mask``
    - ``No Mask``

    If a real ONNX model path is not supplied, the class keeps a clean,
    loadable interface and gracefully reports a low-confidence placeholder
    prediction while still producing annotated output.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        output_dir: Union[str, Path] = "processed/mask",
        conf_threshold: float = 0.5,
    ) -> None:
        """
        Initialize the detector.

        Args:
            model_path: Optional path to a ONNX classification model.
            output_dir: Directory for saving annotated images.
            conf_threshold: Decision threshold for mask detection confidence.
        """
        self.model_path = Path(model_path) if model_path else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.conf_threshold = float(conf_threshold)
        self.session = None
        self.input_name = None
        self.input_shape = None

        if self.model_path is not None:
            self._load_model(self.model_path)
        else:
            logger.warning(
                "No pretrained ONNX model path was provided. "
                "MaskDetector is running in interface mode."
            )

    def _load_model(self, model_path: Union[str, Path]) -> None:
        """Attempt to load an ONNX runtime session for the provided model."""
        if ort is None:
            logger.warning(
                "onnxruntime is not installed. Falling back to interface mode."
            )
            return

        try:
            session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            self.session = session
            self.input_name = session.get_inputs()[0].name
            self.input_shape = tuple(session.get_inputs()[0].shape)
            logger.info("Loaded ONNX mask detector from %s", model_path)
        except Exception as exc:
            logger.warning("Unable to load ONNX mask detector: %s", exc)
            self.session = None

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

        if image.ndim != 3:
            raise ValueError("Input image must be a 3-channel BGR/OpenCV image.")

        return image

    @staticmethod
    def _annotate_result(
        image: np.ndarray,
        bounding_box: BoundingBox,
        label: str,
        confidence: float,
    ) -> np.ndarray:
        """Draw a bounding box and label on the input image."""
        annotated = image.copy()
        x1, y1, x2, y2 = bounding_box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{label} {confidence:.2f}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        return annotated

    def _normalize_model_outputs(self, outputs) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize outputs from different ONNX backends to a probability vector."""
        if isinstance(outputs, (list, tuple)):
            raw_output = outputs[0]
        else:
            raw_output = outputs

        array = np.asarray(raw_output)
        array = np.squeeze(array)

        if array.ndim == 0:
            array = np.array([1.0 - float(array), float(array)])

        if array.ndim == 1 and array.size >= 2:
            probabilities = array.astype(np.float32)
            if probabilities.ndim == 1 and probabilities.shape[0] > 1:
                total = probabilities.sum()
                if total > 0:
                    probabilities = probabilities / total
                else:
                    probabilities = np.array([0.5, 0.5], dtype=np.float32)
            return probabilities, np.array([0, 1], dtype=np.int32)

        if array.ndim == 2 and array.shape[-1] >= 2:
            probabilities = array[0].astype(np.float32)
            total = probabilities.sum()
            if total > 0:
                probabilities = probabilities / total
            return probabilities, np.array([0, 1], dtype=np.int32)

        raise ValueError("Unsupported ONNX output format for mask classification.")

    def _predict_with_onnx(self, image: np.ndarray) -> MaskPrediction:
        """Run a best-effort ONNX inference using the current model session."""
        if self.session is None:
            return self._predict_interface(image)

        try:
            height, width = image.shape[:2]
            bbox = (0, 0, width - 1, height - 1)

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb = rgb.astype(np.float32) / 255.0

            input_tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
            if self.input_shape is not None:
                dims = [int(d) for d in self.input_shape if d not in (None, -1)]
                if len(dims) == 4:
                    input_tensor = cv2.resize(rgb, (dims[-1], dims[-2]))
                    input_tensor = np.transpose(input_tensor, (2, 0, 1))[None, ...]

            outputs = self.session.run(None, {self.input_name: input_tensor})
            probabilities, _ = self._normalize_model_outputs(outputs)

            class_index = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities))
            mask_label = "Mask" if class_index == 0 else "No Mask"
            mask_detected = class_index == 0 and confidence >= self.conf_threshold

            return MaskPrediction(
                mask_detected=mask_detected,
                confidence=confidence,
                bounding_box=bbox,
                label=mask_label,
            )
        except Exception as exc:
            logger.warning("ONNX inference failed: %s", exc)
            return self._predict_interface(image)

    def _predict_interface(self, image: np.ndarray) -> MaskPrediction:
        """Return a graceful interface-mode prediction when no ONNX model is loaded."""
        height, width = image.shape[:2]
        bbox = (0, 0, width - 1, height - 1)
        return MaskPrediction(
            mask_detected=False,
            confidence=0.0,
            bounding_box=bbox,
            label="No Mask",
        )

    def predict(
        self,
        image_input: ImageInput,
        save_annotated: bool = True,
    ) -> MaskPrediction:
        """
        Classify a face image as ``Mask`` or ``No Mask``.

        Args:
            image_input: File path or NumPy image array.
            save_annotated: Whether to save a visualization image.

        Returns:
            MaskPrediction: Container with classification result and geometry.
        """
        image = self._load_image(image_input)
        prediction = self._predict_with_onnx(image)

        if save_annotated:
            annotated = self._annotate_result(
                image,
                prediction.bounding_box,
                prediction.label,
                prediction.confidence,
            )

            output_name = "mask_prediction.png"
            if isinstance(image_input, (str, Path)):
                output_name = f"{Path(image_input).stem}_mask_prediction.png"

            out_path = self.output_dir / output_name
            cv2.imwrite(str(out_path), annotated)
            logger.info("Saved annotated mask result to %s", out_path.resolve())

        return prediction


def detect_mask(
    image_input: ImageInput,
    model_path: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "processed/mask",
) -> MaskPrediction:
    """Convenience helper to run the mask detector on an image."""
    detector = MaskDetector(model_path=model_path, output_dir=output_dir)
    return detector.predict(image_input=image_input, save_annotated=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("MaskDetector ready. Supply an ONNX model path to enable real inference.")
