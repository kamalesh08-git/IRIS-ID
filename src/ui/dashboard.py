from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import streamlit as st

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PersonRecord:
    """Represents a registered person in the system."""

    person_id: str
    name: str
    enrollment_date: str
    metadata: Dict[str, Any]
    status: str = "Registered"


@dataclass(frozen=True)
class RecognitionHistoryEntry:
    """Represents a single recognition event."""

    timestamp: str
    person_id: str
    name: str
    confidence: float
    status: str
    notes: Optional[str] = None


class Dashboard:
    """User interface for displaying recognition results and system status."""

    def __init__(self, title: str = "Adaptive Periocular Recognition") -> None:
        self.title = title
        logger.debug("Initializing Dashboard with title: %s", title)

    def render_header(self) -> None:
        """Render the dashboard header."""
        st.title(self.title)
        st.markdown("### Adaptive periocular-based masked face recognition system")
        st.divider()

    def display_live_recognition(self, results: Iterable[Dict[str, Any]]) -> None:
        """Display live recognition results from the recognition service."""
        st.subheader("Live Recognition Results")

        if not results:
            st.info("No live recognition results available.")
            logger.debug("display_live_recognition called with no results")
            return

        st.dataframe(
            [
                {
                    "Time": item.get("timestamp", "-") ,
                    "Name": item.get("name", "Unknown"),
                    "Confidence": f"{item.get('confidence', 0.0):.2%}",
                    "Status": item.get("status", "Pending"),
                }
                for item in results
            ],
            use_container_width=True,
        )

    def display_registered_users(self, users: Iterable[PersonRecord]) -> None:
        """Display a list of registered users and metadata."""
        st.subheader("Registered Users")
        user_list = list(users)

        if not user_list:
            st.warning("No registered users found.")
            logger.debug("display_registered_users called with empty list")
            return

        st.dataframe(
            [
                {
                    "Person ID": user.person_id,
                    "Name": user.name,
                    "Enrolled": user.enrollment_date,
                    "Status": user.status,
                    "Metadata": user.metadata,
                }
                for user in user_list
            ],
            use_container_width=True,
        )

    def display_confidence_score(self, score: float, label: str = "Current Confidence") -> None:
        """Display the confidence score of the last recognition attempt."""
        st.subheader(label)
        st.metric(label="Confidence", value=f"{score:.2%}")
        logger.debug("display_confidence_score: %f", score)

    def display_recognition_history(self, history: Iterable[RecognitionHistoryEntry]) -> None:
        """Display historical recognition events."""
        st.subheader("Recognition History")
        history_list = list(history)

        if not history_list:
            st.info("Recognition history is empty.")
            logger.debug("display_recognition_history called with empty history")
            return

        st.dataframe(
            [
                {
                    "Time": entry.timestamp,
                    "Person ID": entry.person_id,
                    "Name": entry.name,
                    "Confidence": f"{entry.confidence:.2%}",
                    "Status": entry.status,
                    "Notes": entry.notes or "",
                }
                for entry in history_list
            ],
            use_container_width=True,
        )

    def display_system_status(self, status: str, details: Optional[Dict[str, str]] = None) -> None:
        """Display the current application status and system metrics."""
        st.subheader("System Status")
        st.write(status)

        if details:
            status_columns = st.columns(2)
            for index, (key, value) in enumerate(details.items()):
                status_columns[index % 2].metric(label=key, value=value)

        logger.debug("display_system_status: %s %s", status, details)

    def render_summary(
        self,
        users: Iterable[PersonRecord],
        history: Iterable[RecognitionHistoryEntry],
        live_results: Iterable[Dict[str, Any]],
        status: str,
        status_details: Optional[Dict[str, str]] = None,
    ) -> None:
        """Render a compact dashboard summary view."""
        self.render_header()

        self.display_system_status(status, status_details)
        st.divider()

        self.display_live_recognition(live_results)
        st.divider()

        self.display_confidence_score(
            score=float(status_details.get("Confidence", "0").strip("%")) / 100
            if status_details and status_details.get("Confidence")
            else 0.0,
        )
        st.divider()

        self.display_registered_users(users)
        st.divider()

        self.display_recognition_history(history)


def load_dashboard_assets(asset_folder: Path) -> None:
    """Load optional assets for the dashboard if available."""
    if not asset_folder.exists():
        logger.debug("Dashboard asset folder does not exist: %s", asset_folder)
        return

    for asset_path in asset_folder.glob("*.css"):
        with asset_path.open(encoding="utf-8") as asset_file:
            st.markdown(f"<style>{asset_file.read()}</style>", unsafe_allow_html=True)
            logger.debug("Loaded dashboard asset: %s", asset_path)
