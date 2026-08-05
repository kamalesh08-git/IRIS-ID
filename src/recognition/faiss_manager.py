"""FAISS index management for IRIS-ID embedding search."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import faiss
import numpy as np

from .utils import config

logger = logging.getLogger(__name__)


class FaissManager:
    """Manage FAISS index creation, persistence, and search operations."""

    def __init__(
        self,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        embedding_dimension: int = config.EMBEDDING_DIMENSION,
    ) -> None:
        self.index_path = Path(index_path) if index_path is not None else config.FAISS_INDEX_PATH
        self.metadata_path = Path(metadata_path) if metadata_path is not None else config.METADATA_PATH
        self.embedding_dimension = embedding_dimension

        config.ensure_database_dirs()
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, Any]] = []

    def create_index(self) -> None:
        """Create a new FAISS index for inner product similarity."""
        self.index = faiss.IndexFlatIP(self.embedding_dimension)
        logger.info("Created new FAISS index with dimension %d", self.embedding_dimension)

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        person_name: str,
        source_path: Optional[Path] = None,
    ) -> None:
        """Add normalized embeddings and associated metadata to the index."""
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(
                "Embeddings must be a 2D array with shape (N, %d)" % self.embedding_dimension
            )

        if self.index is None:
            self.create_index()

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        self.index.add(embeddings)
        start_id = len(self.metadata)
        for offset in range(embeddings.shape[0]):
            entry: Dict[str, Any] = {
                "person_name": person_name,
                "source": str(source_path) if source_path is not None else None,
                "index": start_id + offset,
            }
            self.metadata.append(entry)

        logger.info(
            "Added %d embeddings for person '%s' to FAISS index",
            embeddings.shape[0],
            person_name,
        )

    def add_embedding_files(self, embedding_files: Iterable[Path]) -> None:
        """Load embeddings from .npy files and add them to the index."""
        for embedding_file in embedding_files:
            if not embedding_file.exists():
                logger.warning("Embedding file not found: %s", embedding_file)
                continue

            embeddings = np.load(str(embedding_file))
            person_name = embedding_file.stem
            self.add_embeddings(embeddings, person_name, embedding_file)

    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search the FAISS index for the top-k nearest neighbors."""
        if self.index is None:
            raise RuntimeError("FAISS index has not been created or loaded")

        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings[np.newaxis, :]

        if query_embeddings.ndim != 2 or query_embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(
                "Query embeddings must have shape (N, %d)" % self.embedding_dimension
            )

        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)

        distances, indices = self.index.search(query_embeddings, top_k)
        return distances, indices

    def get_metadata(self, index: int) -> Dict[str, Any]:
        """Retrieve metadata for a specific vector index."""
        try:
            return self.metadata[index]
        except IndexError as exc:
            raise IndexError(f"Metadata index out of range: {index}") from exc

    def save_index(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        if self.index is None:
            raise RuntimeError("No FAISS index to save")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            faiss.write_index(self.index, str(self.index_path))
            logger.info("Saved FAISS index to %s", self.index_path)
        except Exception as exc:
            logger.exception("Failed to save FAISS index")
            raise IOError(f"Unable to save FAISS index: {exc}") from exc

        self.save_metadata()

    def load_index(self) -> None:
        """Load the FAISS index and metadata from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {self.index_path}")

        try:
            self.index = faiss.read_index(str(self.index_path))
            logger.info("Loaded FAISS index from %s", self.index_path)
        except Exception as exc:
            logger.exception("Failed to load FAISS index")
            raise IOError(f"Unable to load FAISS index: {exc}") from exc

        self.load_metadata()

    def save_metadata(self) -> None:
        """Persist metadata alongside the FAISS index."""
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.metadata_path.open("w", encoding="utf-8") as metadata_file:
                json.dump(self.metadata, metadata_file, indent=2)
            logger.info("Saved metadata to %s", self.metadata_path)
        except Exception as exc:
            logger.exception("Failed to save metadata")
            raise IOError(f"Unable to save metadata: {exc}") from exc

    def load_metadata(self) -> None:
        """Load metadata from disk."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        try:
            with self.metadata_path.open("r", encoding="utf-8") as metadata_file:
                self.metadata = json.load(metadata_file)
            logger.info("Loaded metadata from %s", self.metadata_path)
        except Exception as exc:
            logger.exception("Failed to load metadata")
            raise IOError(f"Unable to load metadata: {exc}") from exc

    def has_index(self) -> bool:
        """Return whether a FAISS index has been loaded or created."""
        return self.index is not None


__all__ = ["FaissManager"]
