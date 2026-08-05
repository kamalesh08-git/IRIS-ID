from __future__ import annotations

import random

import streamlit as st

from dashboard.widgets import metric_card


def get_metrics() -> dict:
    """
    Placeholder for future backend API.
    """

    return {
        "accuracy": round(random.uniform(97.5, 99.9), 2),
        "confidence": round(random.uniform(90.0, 99.8), 2),
        "fps": random.randint(24, 32),
        "inference": round(random.uniform(18, 35), 2),
        "gpu": "Available",
        "cpu": random.randint(20, 55),
        "memory": random.randint(35, 75),
        "recognized": random.randint(120, 250),
        "unknown": random.randint(1, 25),
        "model": "Vision-MASK v1.0",
        "status": "Running",
    }


def accuracy_card(value: float) -> None:
    metric_card(
        "Accuracy",
        f"{value}%",
        "+0.8%",
        "🎯",
    )


def confidence_card(value: float) -> None:
    metric_card(
        "Confidence",
        f"{value}%",
        "+1.4%",
        "✅",
    )


def inference_card(value: float) -> None:
    metric_card(
        "Inference Time",
        f"{value} ms",
        "-2 ms",
        "⚡",
    )


def fps_card(value: int) -> None:
    metric_card(
        "FPS",
        str(value),
        "+3 FPS",
        "🎥",
    )


def gpu_card(value: str) -> None:
    metric_card(
        "GPU Status",
        value,
        "CUDA",
        "🖥️",
    )


def cpu_card(value: int) -> None:
    metric_card(
        "CPU Usage",
        f"{value}%",
        "",
        "💻",
    )


def memory_card(value: int) -> None:
    metric_card(
        "Memory Usage",
        f"{value}%",
        "",
        "🧠",
    )


def recognized_card(value: int) -> None:
    metric_card(
        "Recognized Today",
        str(value),
        "+15",
        "👤",
    )


def unknown_card(value: int) -> None:
    metric_card(
        "Unknown Faces",
        str(value),
        "-3",
        "❓",
    )


def model_card(value: str) -> None:
    metric_card(
        "Model",
        value,
        "Latest",
        "🤖",
    )


def status_card(value: str) -> None:
    metric_card(
        "System Status",
        value,
        "Healthy",
        "🟢",
    )


def render_metrics() -> None:
    """
    Render all dashboard KPI cards.
    """

    metrics = get_metrics()

    st.subheader("📊 System Metrics")

    row1 = st.columns(4)

    with row1[0]:
        accuracy_card(metrics["accuracy"])

    with row1[1]:
        confidence_card(metrics["confidence"])

    with row1[2]:
        inference_card(metrics["inference"])

    with row1[3]:
        fps_card(metrics["fps"])

    st.markdown("<br>", unsafe_allow_html=True)

    row2 = st.columns(4)

    with row2[0]:
        gpu_card(metrics["gpu"])

    with row2[1]:
        cpu_card(metrics["cpu"])

    with row2[2]:
        memory_card(metrics["memory"])

    with row2[3]:
        recognized_card(metrics["recognized"])

    st.markdown("<br>", unsafe_allow_html=True)

    row3 = st.columns(3)

    with row3[0]:
        unknown_card(metrics["unknown"])

    with row3[1]:
        model_card(metrics["model"])

    with row3[2]:
        status_card(metrics["status"])
