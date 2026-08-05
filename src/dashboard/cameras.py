from __future__ import annotations

import random
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.session import (
    is_camera_running,
    start_camera,
    stop_camera,
)


def get_camera_status() -> str:
    return "Online" if is_camera_running() else "Offline"


def camera_controls() -> None:
    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "▶️ Start Camera",
            use_container_width=True,
        ):
            start_camera()

    with col2:
        if st.button(
            "⏹ Stop Camera",
            use_container_width=True,
        ):
            stop_camera()


def camera_status() -> None:
    status = get_camera_status()

    if status == "Online":
        st.success(f"Camera Status : {status}")
    else:
        st.error(f"Camera Status : {status}")


def placeholder_frame() -> np.ndarray:
    frame = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    cv2.putText(
        frame,
        "Camera Preview",
        (170, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    return frame


def webcam_preview() -> None:
    frame = placeholder_frame()

    st.image(
        frame,
        channels="BGR",
        use_container_width=True,
    )


def recognize_face() -> dict:
    names = [
        "John Doe",
        "Alice",
        "Michael",
        "Emma",
        "Unknown",
    ]

    name = random.choice(names)

    return {
        "name": name,
        "confidence": round(random.uniform(91, 99.9), 2),
        "inference_time": round(random.uniform(18, 35), 2),
        "status": "Recognized" if name != "Unknown" else "Unknown",
        "time": datetime.now().strftime("%H:%M:%S"),
    }


def capture_image() -> None:
    if st.button(
        "📸 Capture Image",
        use_container_width=True,
    ):
        result = recognize_face()

        st.session_state["recognition_result"] = result

        history = st.session_state.get(
            "recognition_history",
            [],
        )

        history.insert(0, result)

        st.session_state["recognition_history"] = history[:20]


def recognition_result() -> None:
    result = st.session_state.get("recognition_result")

    if result is None:
        st.info("No recognition performed.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Person",
            result["name"],
        )

        st.metric(
            "Confidence",
            f"{result['confidence']}%",
        )

    with col2:
        st.metric(
            "Inference",
            f"{result['inference_time']} ms",
        )

        st.metric(
            "Status",
            result["status"],
        )


def recognition_history() -> None:
    history = st.session_state.get(
        "recognition_history",
        [],
    )

    if not history:
        st.info("Recognition history is empty.")
        return

    st.subheader("Recent Recognitions")

    st.dataframe(
        pd.DataFrame(history),
        use_container_width=True,
        hide_index=True,
    )


def render_camera() -> None:
    st.subheader("📷 Live Camera")

    camera_controls()

    camera_status()

    webcam_preview()

    capture_image()

    st.divider()

    recognition_result()

    st.divider()

    recognition_history()
