"""Integration package exports.

This module exposes the main classes provided by the `integration`
package for convenient imports, e.g. `from integration import RecognitionPipeline`.
"""

from .pipeline import RecognitionPipeline, Prediction
from .visualizer import ResultVisualizer
from .metrics import Metrics
from .benchmark import Benchmark, InferenceRecord
from .evaluator import Evaluator

__all__ = [
    "RecognitionPipeline",
    "Prediction",
    "ResultVisualizer",
    "Metrics",
    "Benchmark",
    "InferenceRecord",
    "Evaluator",
]
