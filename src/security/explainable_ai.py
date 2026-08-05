"""
explanation_generator.py

Rule-based (no LLM) module that converts raw masked-face-recognition
metrics into a short, professional, human-readable explanation suitable
for display on a security dashboard.

Given:
    - identity                (str)   claimed/matched identity, or None
    - similarity_score        (float) raw similarity, native range configurable
    - liveness_confidence     (float) [0, 1]
    - image_quality_score     (float) native range configurable (default 0-100)
    - eye_visibility_score    (float) [0, 1]
    - pose_quality_score      (float) [0, 1]
    - trust_score             (float) [0, 100] - typically from TrustScoreCalculator

This module produces a multi-line, plain-English explanation like:

    Authentication Successful.
    Identity Verified: John Doe.
    Matched primarily using eye geometry and eyebrow features.
    Liveness check passed with high confidence.
    Image quality is excellent.
    Trust Score: 94/100.

It is entirely deterministic (if/else + threshold tables) -- no model
calls, no randomness -- so results are reproducible and auditable, which
matters for a security-facing explanation.
"""

from dataclasses import dataclass
from typing import List, Optional


# ------------------------------------------------------------------------ #
# Threshold configuration
# ------------------------------------------------------------------------ #
# Centralizing thresholds here means the wording logic below never hardcodes
# a magic number -- tune the system's "personality" by editing this class.
# ------------------------------------------------------------------------ #

@dataclass
class ExplanationThresholds:
    """Cutoffs used to translate raw metrics into qualitative language."""

    # Trust score (0-100) -> overall pass/fail decision
    trust_pass_threshold: float = 60.0

    # Similarity (assumed normalized to [0, 1] before reaching this module)
    similarity_strong: float = 0.85
    similarity_moderate: float = 0.65

    # Liveness confidence [0, 1]
    liveness_high: float = 0.85
    liveness_moderate: float = 0.60

    # Image quality (assumed normalized to [0, 1] before reaching this module)
    image_quality_excellent: float = 0.85
    image_quality_acceptable: float = 0.60
    image_quality_poor: float = 0.40

    # Eye visibility [0, 1] -- critical for a masked face, since it's the
    # main visible biometric region
    eye_visibility_high: float = 0.85
    eye_visibility_moderate: float = 0.60

    # Pose quality [0, 1] -- 1.0 = perfectly frontal
    pose_frontal: float = 0.85
    pose_slight_angle: float = 0.60


# ------------------------------------------------------------------------ #
# Input container
# ------------------------------------------------------------------------ #

@dataclass
class AuthenticationMetrics:
    """
    Raw metrics feeding the explanation generator.

    Normalization contract: similarity_score, image_quality_score,
    liveness_confidence, eye_visibility_score, and pose_quality_score are
    all expected in [0, 1]. If your upstream scores use a different native
    range (e.g. cosine similarity in [-1, 1], or quality on 0-100), normalize
    them before constructing this object -- e.g. reuse the normalization
    already done inside your TrustScoreCalculator.
    """
    identity: Optional[str]
    similarity_score: float
    liveness_confidence: float
    image_quality_score: float
    eye_visibility_score: float
    pose_quality_score: float
    trust_score: float  # 0-100


# ------------------------------------------------------------------------ #
# Explanation generator
# ------------------------------------------------------------------------ #

class ExplanationGenerator:
    """
    Converts AuthenticationMetrics into a professional, human-readable,
    multi-line explanation string for a security dashboard.

    Entirely rule-based: every sentence is produced by comparing a metric
    against configurable thresholds and selecting from a fixed set of
    professionally-worded templates. No external API or LLM is used.

    Example
    -------
    >>> metrics = AuthenticationMetrics(
    ...     identity="John Doe",
    ...     similarity_score=0.91,
    ...     liveness_confidence=0.93,
    ...     image_quality_score=0.88,
    ...     eye_visibility_score=0.95,
    ...     pose_quality_score=0.80,
    ...     trust_score=94,
    ... )
    >>> print(ExplanationGenerator().generate(metrics))
    Authentication Successful.
    Identity Verified: John Doe.
    Matched primarily using eye geometry and eyebrow features.
    Liveness check passed with high confidence.
    Image quality is excellent.
    Trust Score: 94/100.
    """

    def __init__(self, thresholds: Optional[ExplanationThresholds] = None) -> None:
        self.t = thresholds or ExplanationThresholds()

    # -------------------------------------------------------------- #
    # Public API
    # -------------------------------------------------------------- #

    def generate(self, metrics: AuthenticationMetrics) -> str:
        """Build the full multi-line explanation string."""
        lines: List[str] = []

        lines.append(self._decision_line(metrics.trust_score))
        lines.append(self._identity_line(metrics))
        lines.append(self._match_basis_line(metrics))
        lines.append(self._liveness_line(metrics.liveness_confidence))
        lines.append(self._image_quality_line(metrics.image_quality_score))
        lines.append(self._pose_line(metrics.pose_quality_score))
        lines.append(self._trust_score_line(metrics.trust_score))

        # Drop any line a helper chose to omit (returned as None/empty)
        return "\n".join(line for line in lines if line)

    def generate_summary(self, metrics: AuthenticationMetrics) -> str:
        """
        A compact single-line variant for table rows / log entries, e.g.:
        'PASS (94/100) - John Doe - eye geometry match - excellent image quality'
        """
        decision = "PASS" if metrics.trust_score >= self.t.trust_pass_threshold else "FAIL"
        identity_part = metrics.identity if metrics.identity else "Unverified"
        basis = self._match_basis_phrase(metrics)
        quality_phrase = self._image_quality_phrase(metrics.image_quality_score)
        return (
            f"{decision} ({metrics.trust_score:.0f}/100) - {identity_part} - "
            f"{basis} - {quality_phrase} image quality"
        )

    # -------------------------------------------------------------- #
    # Line builders (each returns one sentence, or "" to omit it)
    # -------------------------------------------------------------- #

    def _decision_line(self, trust_score: float) -> str:
        if trust_score >= self.t.trust_pass_threshold:
            return "Authentication Successful."
        return "Authentication Failed."

    def _identity_line(self, metrics: AuthenticationMetrics) -> str:
        if not metrics.identity:
            return "Identity Not Verified."
        if metrics.trust_score >= self.t.trust_pass_threshold:
            return f"Identity Verified: {metrics.identity}."
        return f"Identity Not Confirmed (closest match: {metrics.identity})."

    def _match_basis_line(self, metrics: AuthenticationMetrics) -> str:
        """
        Explains *which visible facial region* drove the match. Since the
        face is masked, this reasons primarily off eye visibility and pose
        quality -- the periocular region and brow area are what remains
        reliably visible.
        """
        phrase = self._match_basis_phrase(metrics)
        if metrics.similarity_score < self.t.similarity_moderate:
            return "Insufficient facial similarity to confirm a reliable match."
        return f"Matched primarily using {phrase}."

    def _match_basis_phrase(self, metrics: AuthenticationMetrics) -> str:
        """Returns just the noun phrase describing the match basis (no verb)."""
        eye_ok = metrics.eye_visibility_score >= self.t.eye_visibility_moderate
        frontal = metrics.pose_quality_score >= self.t.pose_slight_angle

        if metrics.similarity_score < self.t.similarity_moderate:
            return "limited visible facial features"

        if eye_ok and frontal:
            return "eye geometry and eyebrow features"
        if eye_ok and not frontal:
            return "eye geometry, captured at a non-frontal angle"
        if not eye_ok:
            return "periocular region features, with reduced eye visibility"
        return "visible periocular features"

    def _liveness_line(self, liveness_confidence: float) -> str:
        if liveness_confidence >= self.t.liveness_high:
            return "Liveness check passed with high confidence."
        if liveness_confidence >= self.t.liveness_moderate:
            return "Liveness check passed with moderate confidence."
        return "Liveness check did not reach a confident pass; possible spoof indicators detected."

    def _image_quality_line(self, image_quality_score: float) -> str:
        phrase = self._image_quality_phrase(image_quality_score)
        return f"Image quality is {phrase}."

    def _image_quality_phrase(self, image_quality_score: float) -> str:
        if image_quality_score >= self.t.image_quality_excellent:
            return "excellent"
        if image_quality_score >= self.t.image_quality_acceptable:
            return "acceptable"
        if image_quality_score >= self.t.image_quality_poor:
            return "below average"
        return "poor"

    def _pose_line(self, pose_quality_score: float) -> str:
        if pose_quality_score >= self.t.pose_frontal:
            return "Face angle was frontal and well-aligned."
        if pose_quality_score >= self.t.pose_slight_angle:
            return "Face angle showed a slight deviation from frontal."
        return "Face angle was significantly off-frontal, which may have reduced match reliability."

    def _trust_score_line(self, trust_score: float) -> str:
        return f"Trust Score: {trust_score:.0f}/100."


# ------------------------------------------------------------------------ #
# Demo when run directly
# ------------------------------------------------------------------------ #
if __name__ == "__main__":
    generator = ExplanationGenerator()

    # High-confidence successful match
    success_case = AuthenticationMetrics(
        identity="John Doe",
        similarity_score=0.91,
        liveness_confidence=0.93,
        image_quality_score=0.88,
        eye_visibility_score=0.95,
        pose_quality_score=0.80,
        trust_score=94,
    )
    print(generator.generate(success_case))
    print()
    print("Summary line:", generator.generate_summary(success_case))

    print("\n" + "=" * 60 + "\n")

    # Borderline / failed case: poor pose, moderate liveness, low quality
    fail_case = AuthenticationMetrics(
        identity="Jane Smith",
        similarity_score=0.58,
        liveness_confidence=0.55,
        image_quality_score=0.45,
        eye_visibility_score=0.50,
        pose_quality_score=0.35,
        trust_score=42,
    )
    print(generator.generate(fail_case))
    print()
    print("Summary line:", generator.generate_summary(fail_case))