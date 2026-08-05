"""Benchmarking utilities for the recognition pipeline.

Provides `Benchmark` which measures inference time, average latency,
FPS, and memory usage for a pipeline over a set of images. Results can
be exported to CSV.
"""
from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

try:
    import psutil
except Exception:
    psutil = None

logger = logging.getLogger(__name__)


@dataclass
class InferenceRecord:
    image_path: str
    status: str
    similarity: Optional[float]
    pipeline_elapsed_ms: Optional[float]
    measured_elapsed_ms: float
    memory_rss_bytes: Optional[int]


class Benchmark:
    """Measure inference performance for a recognition pipeline.

    Usage:
        bench = Benchmark(results_dir=Path("results"))
        records = bench.run(pipeline.run_batch, image_paths)
        bench.save_csv(records, bench.results_dir / "benchmark.csv")
    """

    def __init__(self, results_dir: Optional[Path] = None) -> None:
        self.results_dir = Path(results_dir or Path.cwd() / "results")
        self.reports_dir = self.results_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _memory_rss(self) -> Optional[int]:
        if psutil is None:
            return None
        try:
            p = psutil.Process()
            return int(p.memory_info().rss)
        except Exception:
            logger.exception("Failed to read process memory usage")
            return None

    def run(self, run_fn: Callable[[Iterable[Path]], List[Any]], image_paths: Iterable[Path], warmup: int = 0) -> List[InferenceRecord]:
        """Run benchmarking by invoking a provided run function.

        Args:
            run_fn: Callable that accepts an iterable of Paths and returns a
                list of prediction-like objects. For example,
                `RecognitionPipeline.run_batch`.
            image_paths: Iterable of image Paths to process.
            warmup: Number of warmup runs (ignored outputs) to stabilise caches.

        Returns:
            List of `InferenceRecord` for each image.
        """

        image_list = list(image_paths)
        records: List[InferenceRecord] = []

        # Warmup
        for _ in range(warmup):
            try:
                _ = run_fn(image_list)
            except Exception:
                logger.exception("Warmup run failed; continuing to timed run")

        # Timed run: measure wall-clock time and memory per-image
        start_total = time.time()
        mem_before = self._memory_rss()

        # We'll run per-image to measure each call separately for latency
        for img in image_list:
            t0 = time.time()
            mem_before_img = self._memory_rss()
            try:
                # If run_fn expects batch, wrap single element
                result = run_fn([img])
                # If run_fn returns list, take first
                res = result[0] if isinstance(result, (list, tuple)) and result else result
                status = getattr(res, "status", None) or (res.get("status") if isinstance(res, dict) else "unknown")
                similarity = getattr(res, "similarity", None) or (res.get("similarity") if isinstance(res, dict) else None)
                pipeline_elapsed = getattr(res, "elapsed_ms", None) or (res.get("elapsed_ms") if isinstance(res, dict) else None)
            except Exception as exc:
                logger.exception("Inference failed for %s", img)
                status = "error"
                similarity = None
                pipeline_elapsed = None
            t1 = time.time()
            mem_after_img = self._memory_rss()

            measured_ms = (t1 - t0) * 1000.0
            mem_rss = None
            if mem_before_img is not None and mem_after_img is not None:
                mem_rss = max(mem_before_img, mem_after_img)

            records.append(InferenceRecord(
                image_path=str(img),
                status=str(status),
                similarity=float(similarity) if similarity is not None else None,
                pipeline_elapsed_ms=float(pipeline_elapsed) if pipeline_elapsed is not None else None,
                measured_elapsed_ms=measured_ms,
                memory_rss_bytes=mem_rss,
            ))

        total_time = time.time() - start_total
        logger.info("Processed %d images in %.3fs (%.2f FPS)", len(image_list), total_time, len(image_list) / total_time if total_time > 0 else 0.0)

        return records

    def summarize(self, records: List[InferenceRecord]) -> dict:
        """Return basic summary statistics for the benchmark run."""

        times = [r.measured_elapsed_ms for r in records if r.measured_elapsed_ms is not None]
        pipeline_times = [r.pipeline_elapsed_ms for r in records if r.pipeline_elapsed_ms is not None]
        mems = [r.memory_rss_bytes for r in records if r.memory_rss_bytes is not None]

        summary = {
            "num_images": len(records),
            "total_time_s": sum(times) / 1000.0 if times else 0.0,
            "avg_latency_ms": float(sum(times) / len(times)) if times else 0.0,
            "median_latency_ms": float(np.median(times)) if times else 0.0,
            "avg_pipeline_ms": float(sum(pipeline_times) / len(pipeline_times)) if pipeline_times else None,
            "max_memory_rss_bytes": int(max(mems)) if mems else None,
            "fps": (len(records) / (sum(times) / 1000.0)) if times and sum(times) > 0 else 0.0,
        }
        return summary

    def save_csv(self, records: List[InferenceRecord], out_file: Optional[Path] = None) -> Path:
        """Save per-image benchmark records and summary to a CSV file."""

        out_file = Path(out_file or (self.reports_dir / "benchmark.csv"))
        out_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with out_file.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["image_path", "status", "similarity", "pipeline_elapsed_ms", "measured_elapsed_ms", "memory_rss_bytes"])
                for r in records:
                    writer.writerow([r.image_path, r.status, r.similarity, r.pipeline_elapsed_ms, r.measured_elapsed_ms, r.memory_rss_bytes])
            logger.info("Saved benchmark CSV to %s", out_file)
        except Exception:
            logger.exception("Failed to save benchmark CSV to %s", out_file)
            raise

        return out_file


__all__ = ["Benchmark", "InferenceRecord"]
