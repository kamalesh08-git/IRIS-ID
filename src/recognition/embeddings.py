"""Embedding generation for IRIS-ID recognition using InsightFace ArcFace weights."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .utils import config

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class EmbeddingGenerator:
    """Generate face embeddings from aligned or periocular images.

    This module accepts preprocessed images produced by Member 1 and
    produces normalized 512-dimensional embeddings using an InsightFace
    ONNX ArcFace IR-50 model.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        batch_size: int = config.BATCH_SIZE,
        image_size: Tuple[int, int] = config.IMAGE_SIZE,
        provider: str = config.ONNX_PROVIDER,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else config.MODEL_PATH
        self.batch_size = batch_size
        self.image_size = image_size
        self.provider = provider

        config.ensure_database_dirs()
        self._session = self._load_session()

    def _load_session(self) -> ort.InferenceSession:
        if not self.model_path.exists():
            logger.error("ONNX model not found at %s", self.model_path)
            raise FileNotFoundError(f"ONNX model missing: {self.model_path}")

        session_options = ort.SessionOptions()
        try:
            session = ort.InferenceSession(
                str(self.model_path),
                sess_options=session_options,
                providers=[self.provider],
            )
        except Exception as exc:
            logger.exception("Failed to initialize ONNX runtime session")
            raise RuntimeError(f"Unable to load ONNX model: {exc}") from exc

        logger.info("Loaded ArcFace ONNX model from %s", self.model_path)
        return session

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("Input image is None")

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image - np.array(config.PIXEL_MEAN, dtype=np.float32)) / np.array(
            config.PIXEL_STD, dtype=np.float32
        )
        image = np.transpose(image, (2, 0, 1))
        return image

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        return embeddings / norms

    def _batch(self, items: List[np.ndarray]) -> Iterable[List[np.ndarray]]:
        for start in range(0, len(items), self.batch_size):
            yield items[start : start + self.batch_size]

    def generate_embedding(self, image: np.ndarray) -> np.ndarray:
        """Generate a single normalized embedding for one image."""
        embeddings = self.generate_embeddings([image])
        return embeddings[0]

    def generate_embeddings(self, images: List[np.ndarray]) -> np.ndarray:
        """Generate normalized embeddings for a batch of images."""
        if not images:
            raise ValueError("No images provided for embedding generation")

        input_name = self._session.get_inputs()[0].name
        output_name = self._session.get_outputs()[0].name
        embedding_batches: List[np.ndarray] = []

        for batch_images in self._batch(images):
            batch_tensor = np.stack([self._preprocess(img) for img in batch_images], axis=0)
            try:
                raw_embeddings = self._session.run({output_name: None}, {input_name: batch_tensor})[0]
            except Exception as exc:
                logger.exception("ONNX runtime failed during inference")
                raise RuntimeError(f"Embedding inference failed: {exc}") from exc

            if raw_embeddings.ndim != 2 or raw_embeddings.shape[1] != config.EMBEDDING_DIMENSION:
                raise ValueError(
                    "Unexpected embedding output shape: %s" % (raw_embeddings.shape,)
                )

            embedding_batches.append(self._normalize_embeddings(raw_embeddings.astype(np.float32)))

        return np.vstack(embedding_batches)

    def _is_image_file(self, path: Path) -> bool:
        return path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS

    def _load_images_from_folder(self, folder_path: Path) -> Tuple[List[np.ndarray], List[Path]]:
        if not folder_path.exists() or not folder_path.is_dir():
            raise FileNotFoundError(f"Folder does not exist: {folder_path}")

        image_paths = sorted(
            [path for path in folder_path.iterdir() if path.is_file() and self._is_image_file(path)]
        )
        if not image_paths:
            raise ValueError(f"No supported images found in folder: {folder_path}")

        images = []
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning("Unable to read image: %s", image_path)
                continue
            images.append(image)

        if not images:
            raise ValueError(f"Failed to load any valid images from: {folder_path}")

        return images, image_paths

    def save_embeddings(self, embeddings: np.ndarray, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            np.save(str(output_path), embeddings)
        except Exception as exc:
            logger.exception("Failed to save embeddings to %s", output_path)
            raise IOError(f"Unable to save embeddings: {exc}") from exc

        logger.info("Saved embeddings to %s", output_path)

    def generate_embeddings_from_folder(
        self,
        folder_path: Path,
        save_output: bool = True,
        output_path: Optional[Path] = None,
    ) -> np.ndarray:
        """Generate embeddings for all images in a folder and optionally save them."""
        folder_path = Path(folder_path)
        images, _ = self._load_images_from_folder(folder_path)
        embeddings = self.generate_embeddings(images)

        if save_output:
            destination = output_path or (config.EMBEDDINGS_DIR / f"{folder_path.name}.npy")
            self.save_embeddings(embeddings, destination)

        return embeddings

    def generate_embeddings_from_aligned_folder(
        self,
        aligned_folder: Path,
        save_output: bool = True,
        output_path: Optional[Path] = None,
    ) -> np.ndarray:
        """Generate embeddings from aligned face images."""
        return self.generate_embeddings_from_folder(aligned_folder, save_output, output_path)

    def generate_embeddings_from_periocular_folder(
        self,
        periocular_folder: Path,
        save_output: bool = True,
        output_path: Optional[Path] = None,
    ) -> np.ndarray:
        """Generate embeddings from periocular-cropped face images."""
        return self.generate_embeddings_from_folder(periocular_folder, save_output, output_path)


__all__ = ["EmbeddingGenerator"]
