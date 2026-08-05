"""Matching logic for IRIS-ID recognition using FAISS search results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .faiss_manager import FaissManager
from .utils import config

logger = logging.getLogger(__name__)


class Matcher:
    """Match query embeddings against a FAISS database and return identity predictions."""

    def __init__(
        self,
        faiss_manager: FaissManager,
        similarity_threshold: float = config.SIMILARITY_THRESHOLD,
    ) -> None:
        self.faiss_manager = faiss_manager
        self.similarity_threshold = similarity_threshold

        if not self.faiss_manager.has_index():
            logger.warning("Matcher initialized without loaded FAISS index")

    def _resolve_candidate(self, index: int, similarity: float) -> Dict[str, Any]:
        metadata = self.faiss_manager.get_metadata(index)
        return {
            "person_name": metadata.get("person_name"),
            "source": metadata.get("source"),
            "index": metadata.get("index"),
            "similarity": float(similarity),
        }

    def match(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Match a single query embedding and return the best identity prediction."""
        if query_embedding.ndim != 1 or query_embedding.shape[0] != config.EMBEDDING_DIMENSION:
            raise ValueError(
                f"Query embedding must be a 1D numpy array of length {config.EMBEDDING_DIMENSION}"
            )

        if self.faiss_manager.index is None:
            raise RuntimeError("FAISS index is not available for matching")

        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        distances, indices = self.faiss_manager.search(query_embedding, top_k=top_k)
        best_similarity = float(distances[0, 0])
        best_index = int(indices[0, 0])

        if best_similarity < self.similarity_threshold:
            logger.info(
                "Query embedding did not meet similarity threshold: %.4f < %.4f",
                best_similarity,
                self.similarity_threshold,
            )
            return {
                "person_name": "Unknown",
                "similarity": best_similarity,
                "top_k": self._build_top_k(distances[0], indices[0]),
            }

        candidate = self._resolve_candidate(best_index, best_similarity)
        candidate["top_k"] = self._build_top_k(distances[0], indices[0])
        logger.info(
            "Matched query to '%s' with similarity %.4f",
            candidate["person_name"],
            best_similarity,
        )
        return candidate

    def _build_top_k(self, similarities: np.ndarray, indices: np.ndarray) -> List[Dict[str, Any]]:
        top_k_results: List[Dict[str, Any]] = []
        for similarity, index in zip(similarities, indices):
            if int(index) < 0:
                continue
            top_k_results.append(self._resolve_candidate(int(index), float(similarity)))
        return top_k_results

    def match_batch(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Match a batch of query embeddings and return predictions for each."""
        if query_embeddings.ndim != 2 or query_embeddings.shape[1] != config.EMBEDDING_DIMENSION:
            raise ValueError(
                f"Query embeddings must be a 2D array with shape (N, {config.EMBEDDING_DIMENSION})"
            )

        if self.faiss_manager.index is None:
            raise RuntimeError("FAISS index is not available for matching")

        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)

        distances, indices = self.faiss_manager.search(query_embeddings, top_k=top_k)
        results: List[Dict[str, Any]] = []

        for row_similarities, row_indices in zip(distances, indices):
            best_similarity = float(row_similarities[0])
            best_index = int(row_indices[0])
            if best_similarity < self.similarity_threshold:
                results.append(
                    {
                        "person_name": "Unknown",
                        "similarity": best_similarity,
                        "top_k": self._build_top_k(row_similarities, row_indices),
                    }
                )
            else:
                candidate = self._resolve_candidate(best_index, best_similarity)
                candidate["top_k"] = self._build_top_k(row_similarities, row_indices)
                results.append(candidate)

        return results


__all__ = ["Matcher"]
