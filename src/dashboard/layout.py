from __future__ import annotations

from datetime import datetime

import streamlit as st


def render_header() -> None:
    """Render the dashboard header."""

    st.markdown(
        """
        <div style="
            background:linear-gradient(90deg,#2563EB,#7C3AED);
            padding:20px;
            border-radius:18px;
            margin-bottom:20px;
            color:white;
        ">
            <h1 style="margin:0;">
                🤖 AI Masked Face Recognition Dashboard
            </h1>

            <p style="margin-top:8px;font-size:16px;">
                Real-Time Monitoring & Analytics
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render dashboard footer."""

    year = datetime.now().year

    st.markdown("---")

    st.markdown(
        f"""
        <div style="
            text-align:center;
            color:#94A3B8;
            padding:15px;
        ">
            © {year} AI Masked Face Recognition System |
            Built with Streamlit ❤️
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(
    title: str,
    subtitle: str = "",
) -> None:
    """Display section heading."""

    st.markdown(
        f"""
        <h2 style="
            color:white;
            margin-bottom:5px;
        ">
            {title}
        </h2>

        <p style="
            color:#94A3B8;
            margin-top:0;
        ">
            {subtitle}
        </p>
        """,
        unsafe_allow_html=True,
    )


def create_columns(
    count: int,
    gap: str = "medium",
):
    """Return responsive Streamlit columns."""

    return st.columns(count, gap=gap)


def card_container():
    """Reusable bordered container."""

    return st.container(border=True)


def page_container():
    """Main page container."""

    return st.container()


def metric_grid():
    """Return a four-column KPI layout."""

    return st.columns(4)


def chart_grid():
    """Return two-column chart layout."""

    return st.columns(2)


def split_layout(
    left_ratio: int = 2,
    right_ratio: int = 1,
):
    """Return two responsive columns."""

    return st.columns([left_ratio, right_ratio])


def camera_layout():
    """Layout for camera page."""

    left, right = st.columns([3, 2])

    return left, right


def analytics_layout():
    """Layout for analytics."""

    top = st.container()

    bottom_left, bottom_right = st.columns(2)

    return top, bottom_left, bottom_right


def empty_space(height: int = 20) -> None:
    """Vertical spacing."""

    st.markdown(
        f"<div style='height:{height}px'></div>",
        unsafe_allow_html=True,
    )


def horizontal_line() -> None:
    """Display divider."""

    st.divider()


def page_title(
    title: str,
    icon: str = "📊",
) -> None:
    """Display page title."""

    st.title(f"{icon} {title}")


def subheading(text: str) -> None:
    """Display subsection heading."""

    st.subheader(text)


def info_banner(message: str) -> None:
    """Display information banner."""

    st.info(message)


def warning_banner(message: str) -> None:
    """Display warning banner."""

    st.warning(message)


def success_banner(message: str) -> None:
    """Display success banner."""

    st.success(message)


def error_banner(message: str) -> None:
    """Display error banner."""

    st.error(message)
