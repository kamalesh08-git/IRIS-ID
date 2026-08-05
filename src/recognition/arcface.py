"""High-level ArcFace recognition orchestration for IRIS-ID."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .embeddings import EmbeddingGenerator
from .faiss_manager import FaissManager
from .matcher import Matcher
from .utils import config

logger = logging.getLogger(__name__)


class ArcFaceRecognition:
    """ArcFace recognition workflow for aligned and periocular images."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        faiss_index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        batch_size: int = config.BATCH_SIZE,
        similarity_threshold: float = config.SIMILARITY_THRESHOLD,
    ) -> None:
        self.embedding_generator = EmbeddingGenerator(
            model_path=model_path,
            batch_size=batch_size,
        )
        self.faiss_manager = FaissManager(
            index_path=faiss_index_path,
            metadata_path=metadata_path,
        )
        self.matcher = Matcher(self.faiss_manager, similarity_threshold=similarity_threshold)

    def register_person(
        self,
        person_name: str,
        image_folder: Path,
        save_embeddings: bool = True,
    ) -> np.ndarray:
        """Register a new person by generating embeddings from a folder of images."""
        embeddings = self.embedding_generator.generate_embeddings_from_folder(
            image_folder,
            save_output=save_embeddings,
            output_path=(config.EMBEDDINGS_DIR / f"{person_name}.npy") if save_embeddings else None,
        )
        self.faiss_manager.add_embeddings(embeddings, person_name, source_path=image_folder)
        return embeddings

    def build_database_from_embeddings(self, embedding_files: List[Path]) -> None:
        """Build the FAISS database from existing embedding files."""
        self.faiss_manager.create_index()
        self.faiss_manager.add_embedding_files(embedding_files)
        self.faiss_manager.save_index()

    def build_database_from_aligned_folders(
        self,
        aligned_folders: List[Path],
        save_embeddings: bool = True,
    ) -> None:
        """Create embeddings and build database from aligned face folders."""
        self.faiss_manager.create_index()
        for folder in aligned_folders:
            person_name = folder.name
            embeddings = self.embedding_generator.generate_embeddings_from_folder(
                folder,
                save_output=save_embeddings,
                output_path=(config.EMBEDDINGS_DIR / f"{person_name}.npy") if save_embeddings else None,
            )
            self.faiss_manager.add_embeddings(embeddings, person_name, source_path=folder)
        self.faiss_manager.save_index()

    def recognize_aligned_face(self, image: np.ndarray) -> Dict[str, object]:
        """Recognize a single aligned face image."""
        embedding = self.embedding_generator.generate_embedding(image)
        return self.matcher.match(embedding)

    def recognize_periocular_face(self, image: np.ndarray) -> Dict[str, object]:
        """Recognize a single periocular-cropped image."""
        embedding = self.embedding_generator.generate_embedding(image)
        return self.matcher.match(embedding)

    def load_database(self) -> None:
        """Load an existing FAISS database and its metadata."""
        self.faiss_manager.load_index()

    def recognize_from_folder(
        self,
        image_folder: Path,
        top_k: int = 5,
    ) -> List[Dict[str, object]]:
        """Recognize all images in a folder and return identity predictions."""
        image_folder = Path(image_folder)
        if not image_folder.exists() or not image_folder.is_dir():
            raise FileNotFoundError(f"Image folder does not exist: {image_folder}")

        results: List[Dict[str, object]] = []
        for image_path in sorted(image_folder.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}:
                continue
            image = self._read_image(image_path)
            embedding = self.embedding_generator.generate_embedding(image)
            result = self.matcher.match(embedding, top_k=top_k)
            result["image_path"] = str(image_path)
            results.append(result)

        return results

    def _read_image(self, image_path: Path) -> np.ndarray:
        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        return image


__all__ = ["ArcFaceRecognition"]
