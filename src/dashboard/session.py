from __future__ import annotations

from datetime import datetime
from typing import Any

import streamlit as st

DEFAULT_STATE: dict[str, Any] = {
    "page": "Dashboard",
    "theme": "dark",
    "camera_running": False,
    "camera_status": "Offline",
    "recognition_result": None,
    "recognition_history": [],
    "metrics_cache": {},
    "analytics_cache": {},
    "logs_cache": [],
    "last_capture": None,
    "system_status": "Running",
    "model_name": "Vision-MASK v1.0",
    "notifications": [],
}


def initialize_session() -> None:
    """Initialize all required session state variables."""

    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_current_page(page: str) -> None:
    """Set the current dashboard page."""

    st.session_state.page = page


def get_current_page() -> str:
    """Return the current page."""

    return st.session_state.page


def start_camera() -> None:
    """Enable camera."""

    st.session_state.camera_running = True
    st.session_state.camera_status = "Online"


def stop_camera() -> None:
    """Disable camera."""

    st.session_state.camera_running = False
    st.session_state.camera_status = "Offline"


def is_camera_running() -> bool:
    """Return camera running status."""

    return st.session_state.camera_running


def update_recognition(result: dict[str, Any]) -> None:
    """Update recognition result and history."""

    st.session_state.recognition_result = result
    st.session_state.last_capture = datetime.now()

    history = st.session_state.recognition_history
    history.insert(0, result)

    st.session_state.recognition_history = history[:100]


def get_recognition_history() -> list[dict]:
    """Return recognition history."""

    return st.session_state.recognition_history


def update_metrics(metrics: dict[str, Any]) -> None:
    """Update metrics cache."""

    st.session_state.metrics_cache = metrics


def get_metrics() -> dict:
    """Placeholder for future metrics API."""

    return st.session_state.metrics_cache


def update_analytics(data: dict[str, Any]) -> None:
    """Update analytics cache."""

    st.session_state.analytics_cache = data


def get_analytics() -> dict:
    """Placeholder for future analytics API."""

    return st.session_state.analytics_cache


def update_logs(logs: list[dict]) -> None:
    """Update logs cache."""

    st.session_state.logs_cache = logs


def get_logs() -> list[dict]:
    """Placeholder for future logs API."""

    return st.session_state.logs_cache


def push_notification(message: str) -> None:
    """Store notification."""

    st.session_state.notifications.append(
        {
            "message": message,
            "time": datetime.now(),
        }
    )


def clear_notifications() -> None:
    """Clear notifications."""

    st.session_state.notifications.clear()


def reset_dashboard() -> None:
    """Reset dashboard state."""

    for key, value in DEFAULT_STATE.items():
        st.session_state[key] = value


def get_camera_status() -> str:
    """Placeholder for camera status API."""

    return st.session_state.camera_status


def recognize_face() -> dict:
    """Placeholder for recognition API."""

    return {}
