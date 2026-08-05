"""Main entry point for integration and evaluation.

This script wires together the `RecognitionPipeline`, `Evaluator`,
`ResultVisualizer`, and `Benchmark` classes. It attempts to discover
preprocessing and recognition adapter functions from the workspace and
falls back to descriptive runtime errors if they are not available.

Usage: run as a module or script. See CLI `--help` for options.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Tuple

from .benchmark import Benchmark
from .evaluator import Evaluator
from .pipeline import RecognitionPipeline
from .visualizer import ResultVisualizer

logger = logging.getLogger(__name__)


def _try_import_module(name: str):
    """Try to import a module by name, returning the module or None."""

    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _discover_callable(mod_names: Iterable[str], candidates: Iterable[str]):
    """Search a list of module names for the first matching callable.

    Args:
        mod_names: module import paths to try (e.g., 'detection.detection_pipeline').
        candidates: attribute names to look for (e.g., 'preprocess', 'run').

    Returns:
        The found callable, or None if none found.
    """

    for mname in mod_names:
        mod = _try_import_module(mname)
        if mod is None:
            continue
        for cand in candidates:
            fn = getattr(mod, cand, None)
            if callable(fn):
                logger.info("Discovered callable %s in %s", cand, mname)
                return fn
    return None


def _load_preprocess_fn() -> Callable[[Path], Any]:
    """Attempt to load a preprocessing function from likely modules.

    The function should accept a `Path` and return an artifact consumable
    by the recognition function (e.g., a cropped image or an array).
    """

    mod_candidates = [
        "detection.detection_pipeline",
        "detection.adaptive_crop",
        "detection.retinaface",
        "detection.retinaface_detector",
        "detection.retinaface_detector",
        "src.detection.detection_pipeline",
    ]
    fn_names = ["preprocess", "process", "run", "detect_and_crop", "crop_periocular", "preprocess_image"]
    fn = _discover_callable(mod_candidates, fn_names)
    if fn is not None:
        return fn

    def _missing_preprocess(path: Path) -> Any:  # pragma: no cover - guidance path
        raise RuntimeError(
            "Preprocessing adapter not found. Provide a callable named 'preprocess' or 'process' in detection modules."
        )

    return _missing_preprocess


def _load_recognize_fn() -> Callable[[Any], dict]:
    """Attempt to load a recognition function from likely modules.

    The function must accept the output of `preprocess_fn` and return a
    dict with keys like `name`, `similarity`, `status`, and optionally
    `bounding_boxes`, `elapsed_ms`.
    """

    mod_candidates = [
        "recognition.matcher",
        "recognition.faiss_manager",
        "recognition.arcface",
        "recognition.embeddings",
        "src.recognition.matcher",
    ]
    fn_names = ["recognize", "match", "infer", "run", "predict"]
    fn = _discover_callable(mod_candidates, fn_names)
    if fn is not None:
        return fn

    def _missing_recognize(preprocessed: Any) -> dict:  # pragma: no cover - guidance path
        raise RuntimeError(
            "Recognition adapter not found. Provide a callable named 'recognize' or 'match' in recognition modules."
        )

    return _missing_recognize


def _read_dataset_csv(path: Path) -> List[Tuple[Path, str]]:
    """Read a CSV with `image_path,true_label` rows into a list."""

    import csv

    items: List[Tuple[Path, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            if len(row) < 2:
                continue
            img = Path(row[0])
            label = row[1]
            items.append((img, label))
    return items


def configure_logging(level: str = "INFO") -> None:
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Integration/evaluation runner")
    parser.add_argument("--dataset-csv", type=Path, help="CSV file with image_path,true_label rows", required=False)
    parser.add_argument("--results-dir", type=Path, default=Path.cwd() / "results", help="Directory to write results")
    parser.add_argument("--annotate", action="store_true", help="Save annotated images (requires OpenCV)")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark during evaluation")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup runs for benchmark")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    results_dir = Path(args.results_dir)

    # Discover adapters
    preprocess_fn = _load_preprocess_fn()
    recognize_fn = _load_recognize_fn()

    pipeline = RecognitionPipeline(preprocess_fn=preprocess_fn, recognize_fn=recognize_fn, results_dir=results_dir)

    visualizer = None
    if args.annotate:
        try:
            visualizer = ResultVisualizer()
        except Exception:
            logger.exception("Failed to initialize ResultVisualizer; annotation disabled")
            visualizer = None

    evaluator = Evaluator(pipeline=pipeline, results_dir=results_dir, visualizer=visualizer)

    # Run dataset evaluation if provided
    if args.dataset_csv:
        dataset = _read_dataset_csv(args.dataset_csv)
        logger.info("Loaded dataset with %d samples", len(dataset))

        results = evaluator.evaluate(dataset, annotate=args.annotate)
        preds_csv = evaluator.export_predictions_csv(results)
        metrics = evaluator.generate_reports(results)
        logger.info("Evaluation complete. Metrics: %s", metrics)

        if args.benchmark:
            bench = Benchmark(results_dir=results_dir)
            # For benchmarking, create a simple run function wrapper
            def run_fn(batch: Iterable[Path]):
                return pipeline.run_batch(batch)

            records = bench.run(run_fn, [p for p, _ in dataset], warmup=args.warmup)
            bench.save_csv(records)
            logger.info("Benchmark complete")

    else:
        logger.info("No dataset provided. Exiting.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
