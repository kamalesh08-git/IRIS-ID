"""
trust_score_calculator.py

A modular, production-quality utility for computing a unified "trust score"
for a masked face recognition pipeline.

The final trust score blends several independent signals that a masked-face
recognition system typically produces:

    1. Face recognition similarity score  - how closely the detected face
       matches the claimed identity/enrolled embedding.
    2. Liveness confidence                - how likely the input is a real,
       live person rather than a spoof (photo/video/mask attack).
    3. Image quality score                - sharpness/lighting/resolution
       quality of the captured frame.
    4. Eye visibility score                - how visible/usable the ocular
       region is (critical when the lower face is occluded by a mask).
    5. Face pose quality                   - how frontal/well-aligned the
       face is (extreme yaw/pitch/roll hurts reliability).

All inputs are normalized to a common [0, 1] scale before weighting, so the
calculator is robust to components that arrive in different native ranges
(e.g. cosine similarity in [-1, 1] vs. a quality score already in [0, 1]).

Author: (Hackathon submission)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from enum import Enum


class ConfidenceLevel(str, Enum):
    """Discrete confidence bucket derived from the final trust score."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


@dataclass
class TrustScoreWeights:
    """
    Configurable weights for each component of the trust score.

    Weights do not need to sum to 1.0 up front -- they are normalized
    internally at calculation time, so relative magnitudes are what matter.
    """
    similarity: float = 0.35
    liveness: float = 0.25
    image_quality: float = 0.15
    eye_visibility: float = 0.15
    pose_quality: float = 0.10

    def as_dict(self) -> Dict[str, float]:
        return {
            "similarity": self.similarity,
            "liveness": self.liveness,
            "image_quality": self.image_quality,
            "eye_visibility": self.eye_visibility,
            "pose_quality": self.pose_quality,
        }


@dataclass
class TrustScoreResult:
    """Structured result returned by TrustScoreCalculator.calculate()."""
    trust_score: float
    confidence_level: str
    explanation: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convenience serializer, e.g. for JSON responses / logging."""
        return {
            "trust_score": self.trust_score,
            "confidence_level": self.confidence_level,
            "explanation": self.explanation,
        }


class TrustScoreCalculator:
    """
    Computes a unified 0-100 trust score for a masked face recognition
    decision, combining recognition similarity, liveness, image quality,
    eye visibility, and pose quality into a single interpretable metric.

    Example
    -------
    >>> calc = TrustScoreCalculator()
    >>> result = calc.calculate(
    ...     similarity_score=0.82,      # e.g. cosine similarity in [-1, 1]
    ...     liveness_confidence=0.91,   # already in [0, 1]
    ...     image_quality_score=70,     # e.g. on a [0, 100] scale
    ...     eye_visibility_score=0.95,  # in [0, 1]
    ...     pose_quality_score=0.6,     # in [0, 1]
    ... )
    >>> result.trust_score
    78.4
    >>> result.confidence_level
    'High'
    """

    # Score thresholds (on the final 0-100 scale) used to bucket confidence.
    HIGH_THRESHOLD: float = 75.0
    MEDIUM_THRESHOLD: float = 50.0

    def __init__(
        self,
        weights: Optional[TrustScoreWeights] = None,
        similarity_range: tuple = (-1.0, 1.0),
        image_quality_range: tuple = (0.0, 100.0),
    ) -> None:
        """
        Args:
            weights: Custom TrustScoreWeights. Falls back to sensible
                defaults tuned for masked-face recognition (similarity and
                liveness weighted most heavily, since the mask degrades the
                reliability of pose/eye cues somewhat).
            similarity_range: The native (min, max) range the caller's face
                recognition model emits similarity scores in. Commonly
                cosine similarity (-1, 1) or a normalized distance-based
                score already in (0, 1).
            image_quality_range: The native (min, max) range of the image
                quality score, e.g. (0, 100) for a percentage-style metric.
        """
        self.weights = weights or TrustScoreWeights()
        self._similarity_range = similarity_range
        self._image_quality_range = image_quality_range

        # Validate weights are non-negative and not all zero up front so
        # failures surface at construction time, not deep in a demo run.
        if any(w < 0 for w in self.weights.as_dict().values()):
            raise ValueError("Weights must be non-negative.")
        if sum(self.weights.as_dict().values()) == 0:
            raise ValueError("At least one weight must be greater than 0.")

    # ------------------------------------------------------------------ #
    # Normalization helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize(value: float, min_val: float, max_val: float) -> float:
        """
        Linearly rescale `value` from [min_val, max_val] to [0, 1], then
        clamp so noisy/out-of-range model outputs can't blow up the score.
        """
        if max_val == min_val:
            raise ValueError("min_val and max_val must differ for normalization.")
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))

    def _normalize_inputs(
        self,
        similarity_score: float,
        liveness_confidence: float,
        image_quality_score: float,
        eye_visibility_score: float,
        pose_quality_score: float,
    ) -> Dict[str, float]:
        """Normalize every raw input to a common [0, 1] scale."""
        return {
            "similarity": self._normalize(similarity_score, *self._similarity_range),
            # Liveness/eye-visibility/pose are assumed to already be
            # probability-like scores in [0, 1]; clamp defensively anyway.
            "liveness": self._normalize(liveness_confidence, 0.0, 1.0),
            "image_quality": self._normalize(image_quality_score, *self._image_quality_range),
            "eye_visibility": self._normalize(eye_visibility_score, 0.0, 1.0),
            "pose_quality": self._normalize(pose_quality_score, 0.0, 1.0),
        }

    # ------------------------------------------------------------------ #
    # Core calculation
    # ------------------------------------------------------------------ #

    def calculate(
        self,
        similarity_score: float,
        liveness_confidence: float,
        image_quality_score: float,
        eye_visibility_score: float,
        pose_quality_score: float,
    ) -> TrustScoreResult:
        """
        Compute the final trust score and a breakdown of each component's
        contribution.

        Args:
            similarity_score: Raw face recognition similarity, in the range
                given by `similarity_range` at construction time.
            liveness_confidence: Anti-spoofing/liveness confidence in [0, 1].
            image_quality_score: Raw image quality metric, in the range
                given by `image_quality_range` at construction time.
            eye_visibility_score: How visible/usable the eye region is,
                in [0, 1]. Important since the mask occludes the mouth/nose.
            pose_quality_score: Frontal-ness/alignment quality in [0, 1],
                where 1.0 is a perfectly frontal, well-aligned face.

        Returns:
            TrustScoreResult with trust_score (0-100), confidence_level,
            and a per-component explanation dictionary.
        """
        normalized = self._normalize_inputs(
            similarity_score,
            liveness_confidence,
            image_quality_score,
            eye_visibility_score,
            pose_quality_score,
        )

        weights_dict = self.weights.as_dict()
        total_weight = sum(weights_dict.values())

        # Weighted sum over normalized [0,1] values, then scale to 0-100.
        weighted_sum = 0.0
        explanation: Dict[str, Dict[str, float]] = {}

        for component, norm_value in normalized.items():
            weight = weights_dict[component]
            effective_weight = weight / total_weight  # renormalize weights to sum to 1
            contribution = norm_value * effective_weight * 100.0
            weighted_sum += contribution

            explanation[component] = {
                "raw_weight": round(weight, 4),
                "effective_weight": round(effective_weight, 4),
                "normalized_value": round(norm_value, 4),
                "contribution_to_score": round(contribution, 2),
            }

        trust_score = round(weighted_sum, 2)
        confidence_level = self._determine_confidence_level(trust_score)

        return TrustScoreResult(
            trust_score=trust_score,
            confidence_level=confidence_level.value,
            explanation=explanation,
        )

    def _determine_confidence_level(self, trust_score: float) -> ConfidenceLevel:
        """Bucket the final numeric score into a Low/Medium/High label."""
        if trust_score >= self.HIGH_THRESHOLD:
            return ConfidenceLevel.HIGH
        if trust_score >= self.MEDIUM_THRESHOLD:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW


# ---------------------------------------------------------------------- #
# Demo / manual test when run directly
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    # Default weights, cosine similarity in [-1, 1], quality on 0-100 scale
    calculator = TrustScoreCalculator()

    result = calculator.calculate(
        similarity_score=0.82,
        liveness_confidence=0.91,
        image_quality_score=70,
        eye_visibility_score=0.95,
        pose_quality_score=0.60,
    )

    print("Trust Score:", result.trust_score)
    print("Confidence Level:", result.confidence_level)
    print("Explanation:")
    for component, details in result.explanation.items():
        print(f"  {component}: {details}")

    # Example with custom weights emphasizing liveness (anti-spoof heavy)
    custom_weights = TrustScoreWeights(
        similarity=0.30,
        liveness=0.35,
        image_quality=0.10,
        eye_visibility=0.15,
        pose_quality=0.10,
    )
    calculator_custom = TrustScoreCalculator(weights=custom_weights)
    result_custom = calculator_custom.calculate(
        similarity_score=0.55,
        liveness_confidence=0.40,
        image_quality_score=50,
        eye_visibility_score=0.70,
        pose_quality_score=0.45,
    )
    print("\n--- Custom Weights Example ---")
    print("Trust Score:", result_custom.trust_score)
    print("Confidence Level:", result_custom.confidence_level)