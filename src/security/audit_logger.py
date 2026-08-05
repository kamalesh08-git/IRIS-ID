"""
audit_logger.py

Production-quality CSV audit logger for a masked face recognition
authentication pipeline.

Every authentication attempt (successful or not) is appended as one row to
a CSV file, with columns:

    timestamp, identity, similarity, liveness, image_quality,
    trust_score, decision, latency

The CSV file (and its header) is created automatically on first use if it
does not already exist, and the logger is safe to reuse across many
attempts within a process (single open file handle, flushed per write).
"""

import csv
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional, Union


# ------------------------------------------------------------------------ #
# Log entry container
# ------------------------------------------------------------------------ #

@dataclass
class AuthAttempt:
    """
    A single authentication attempt to be recorded.

    Args:
        identity: Claimed or matched identity. Use None / "" for an
            unmatched attempt -- it will be logged as "Unknown".
        similarity: Face recognition similarity score.
        liveness: Liveness confidence score.
        image_quality: Image quality score.
        trust_score: Final trust score (e.g. 0-100) from TrustScoreCalculator.
        decision: Outcome label, e.g. "ACCESS_GRANTED" / "ACCESS_DENIED".
        latency: End-to-end processing time in seconds for this attempt.
        timestamp: When the attempt occurred. Defaults to "now" (UTC) if
            not provided.
    """
    identity: Optional[str]
    similarity: float
    liveness: float
    image_quality: float
    trust_score: float
    decision: str
    latency: float
    timestamp: Optional[datetime] = None


# ------------------------------------------------------------------------ #
# Audit logger
# ------------------------------------------------------------------------ #

class AuditLogger:
    """
    Appends authentication attempts to a CSV audit log.

    Thread-safe for concurrent calls to `log()` within a single process
    (guarded by an internal lock); creates the CSV file and header row
    automatically if the target file does not yet exist.

    Example
    -------
    >>> logger = AuditLogger("audit_log.csv")
    >>> logger.log(AuthAttempt(
    ...     identity="John Doe",
    ...     similarity=0.91,
    ...     liveness=0.93,
    ...     image_quality=0.88,
    ...     trust_score=94,
    ...     decision="ACCESS_GRANTED",
    ...     latency=0.284,
    ... ))
    """

    # Fixed column order -- also serves as the CSV header.
    FIELDNAMES = [
        "timestamp",
        "identity",
        "similarity",
        "liveness",
        "image_quality",
        "trust_score",
        "decision",
        "latency",
    ]

    def __init__(self, log_path: Union[str, Path] = "audit_log.csv") -> None:
        """
        Args:
            log_path: Path to the CSV file. Parent directories are created
                automatically if they don't exist. If the file itself does
                not exist, it is created with a header row on first log().
        """
        self.log_path = Path(log_path)
        self._lock = Lock()
        self._ensure_file_ready()

    # -------------------------------------------------------------- #
    # Setup
    # -------------------------------------------------------------- #

    def _ensure_file_ready(self) -> None:
        """
        Create parent directories and the CSV file (with header) if they
        don't already exist. Safe to call repeatedly -- a no-op once the
        file is present.
        """
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        file_exists = self.log_path.exists()
        file_is_empty = file_exists and os.path.getsize(self.log_path) == 0

        if not file_exists or file_is_empty:
            with open(self.log_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    # -------------------------------------------------------------- #
    # Public API
    # -------------------------------------------------------------- #

    def log(self, attempt: AuthAttempt) -> None:
        """
        Append one authentication attempt as a new row in the CSV file.

        Thread-safe: multiple callers may invoke this concurrently without
        interleaving or corrupting rows.
        """
        timestamp = attempt.timestamp or datetime.now(timezone.utc)

        row = {
            "timestamp": timestamp.isoformat(),
            "identity": attempt.identity if attempt.identity else "Unknown",
            "similarity": round(attempt.similarity, 4),
            "liveness": round(attempt.liveness, 4),
            "image_quality": round(attempt.image_quality, 4),
            "trust_score": round(attempt.trust_score, 2),
            "decision": attempt.decision,
            "latency": round(attempt.latency, 4),
        }

        with self._lock:
            # Re-check in case the file was deleted/moved externally between
            # construction and this call (e.g. log rotation by another tool).
            self._ensure_file_ready()
            with open(self.log_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writerow(row)

    def log_values(
        self,
        identity: Optional[str],
        similarity: float,
        liveness: float,
        image_quality: float,
        trust_score: float,
        decision: str,
        latency: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Convenience wrapper around `log()` for callers who'd rather pass
        plain keyword arguments than construct an AuthAttempt directly.
        """
        self.log(AuthAttempt(
            identity=identity,
            similarity=similarity,
            liveness=liveness,
            image_quality=image_quality,
            trust_score=trust_score,
            decision=decision,
            latency=latency,
            timestamp=timestamp,
        ))

    def read_all(self) -> list:
        """
        Read back all logged rows as a list of dicts. Useful for feeding
        the dashboard UI or for quick debugging during the hackathon demo.
        """
        if not self.log_path.exists():
            return []
        with open(self.log_path, mode="r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def count(self) -> int:
        """Return the number of logged attempts (excludes the header row)."""
        return len(self.read_all())


# ------------------------------------------------------------------------ #
# Demo when run directly
# ------------------------------------------------------------------------ #
if __name__ == "__main__":
    demo_path = "demo_audit_log.csv"
    logger = AuditLogger(demo_path)

    logger.log(AuthAttempt(
        identity="John Doe",
        similarity=0.91,
        liveness=0.93,
        image_quality=0.88,
        trust_score=94,
        decision="ACCESS_GRANTED",
        latency=0.284,
    ))

    logger.log_values(
        identity=None,
        similarity=0.42,
        liveness=0.55,
        image_quality=0.60,
        trust_score=38,
        decision="ACCESS_DENIED",
        latency=0.311,
    )

    print(f"Total logged attempts: {logger.count()}")
    for row in logger.read_all():
        print(row)