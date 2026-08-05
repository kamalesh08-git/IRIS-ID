from __future__ import annotations

import random

import pandas as pd
import plotly.express as px
import streamlit as st


def get_analytics() -> dict:
    return {}


def recognition_trend_chart() -> None:
    df = pd.DataFrame(
        {
            "Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            "Recognitions": [
                random.randint(80, 180)
                for _ in range(7)
            ],
        }
    )

    fig = px.line(
        df,
        x="Day",
        y="Recognitions",
        markers=True,
        title="Recognition Trend",
    )

    fig.update_layout(
        template="plotly_dark",
        height=350,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def recognition_by_hour_chart() -> None:
    df = pd.DataFrame(
        {
            "Hour": list(range(24)),
            "Recognitions": [
                random.randint(0, 40)
                for _ in range(24)
            ],
        }
    )

    fig = px.bar(
        df,
        x="Hour",
        y="Recognitions",
        title="Recognition by Hour",
    )

    fig.update_layout(
        template="plotly_dark",
        height=350,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def confidence_distribution_chart() -> None:
    df = pd.DataFrame(
        {
            "Confidence": [
                random.uniform(80, 100)
                for _ in range(300)
            ]
        }
    )

    fig = px.histogram(
        df,
        x="Confidence",
        nbins=20,
        title="Confidence Distribution",
    )

    fig.update_layout(
        template="plotly_dark",
        height=350,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def masked_vs_unmasked_chart() -> None:
    df = pd.DataFrame(
        {
            "Category": ["Masked", "Unmasked"],
            "Count": [
                random.randint(80, 180),
                random.randint(120, 220),
            ],
        }
    )

    fig = px.pie(
        df,
        names="Category",
        values="Count",
        hole=0.45,
        title="Masked vs Unmasked",
    )

    fig.update_layout(
        template="plotly_dark",
        height=350,
        showlegend=True,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def recognition_success_rate_chart() -> None:
    success = random.randint(92, 99)
    failed = 100 - success

    df = pd.DataFrame(
        {
            "Result": ["Success", "Failed"],
            "Percentage": [success, failed],
        }
    )

    fig = px.bar(
        df,
        x="Result",
        y="Percentage",
        color="Result",
        text="Percentage",
        title="Recognition Success Rate",
    )

    fig.update_layout(
        template="plotly_dark",
        height=350,
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def inference_time_trend_chart() -> None:
    df = pd.DataFrame(
        {
            "Frame": list(range(1, 31)),
            "Inference Time (ms)": [
                round(random.uniform(18, 35), 2)
                for _ in range(30)
            ],
        }
    )

    fig = px.line(
        df,
        x="Frame",
        y="Inference Time (ms)",
        markers=True,
        title="Inference Time Trend",
    )

    fig.update_layout(
        template="plotly_dark",
        height=350,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def render_charts() -> None:
    st.subheader("📈 Analytics Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        recognition_trend_chart()

    with col2:
        recognition_by_hour_chart()

    col3, col4 = st.columns(2)

    with col3:
        confidence_distribution_chart()

    with col4:
        masked_vs_unmasked_chart()

    col5, col6 = st.columns(2)

    with col5:
        recognition_success_rate_chart()

    with col6:
        inference_time_trend_chart()
