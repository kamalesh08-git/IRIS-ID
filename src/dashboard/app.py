from __future__ import annotations

from pathlib import Path

import streamlit as st

from dashboard.cameras import render_camera
from dashboard.charts import render_charts
from dashboard.layout import render_footer, render_header
from dashboard.metrices import render_metrics
from dashboard.session import initialize_session
from dashboard.sidebar import render_sidebar


st.set_page_config(
    page_title="Vision-MASK Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    css_path = Path("assets/styles.css")

    if css_path.exists():
        with css_path.open(encoding="utf-8") as css:
            st.markdown(
                f"<style>{css.read()}</style>",
                unsafe_allow_html=True,
            )


def dashboard_page() -> None:
    render_header()
    render_metrics()
    st.divider()
    render_charts()
    render_footer()


def live_recognition_page() -> None:
    render_header()
    render_camera()
    render_footer()


def enrollment_page() -> None:
    render_header()
    st.title("👤 Enrollment")
    st.info("Enrollment module will be integrated with backend APIs.")
    render_footer()


def analytics_page() -> None:
    render_header()
    render_charts()
    render_footer()


def logs_page() -> None:
    render_header()

    st.title("📜 System Logs")

    logs = [
        {
            "Time": "10:30:21",
            "Event": "Recognition Completed",
            "Status": "Success",
        },
        {
            "Time": "10:32:10",
            "Event": "Unknown Face Detected",
            "Status": "Warning",
        },
        {
            "Time": "10:35:42",
            "Event": "Camera Started",
            "Status": "Info",
        },
    ]

    st.dataframe(
        logs,
        use_container_width=True,
        hide_index=True,
    )

    render_footer()


def settings_page() -> None:
    render_header()

    st.title("⚙️ Settings")

    st.selectbox(
        "Theme",
        ["Dark", "Light"],
        index=0,
        disabled=True,
    )

    st.toggle(
        "Enable Notifications",
        value=True,
        disabled=True,
    )

    st.toggle(
        "Auto Refresh Dashboard",
        value=True,
        disabled=True,
    )

    st.info("Settings are placeholders for future integration.")

    render_footer()


def about_page() -> None:
    render_header()

    st.title("ℹ️ About")

    st.markdown(
        """
### Vision-MASK

AI-powered Masked Face Recognition Dashboard.

**Features**

- Live Camera Preview
- AI Recognition
- Real-Time Analytics
- Interactive Charts
- KPI Dashboard
- Recognition History

**Version**

1.0.0

Built with ❤️ using Streamlit.
"""
    )

    render_footer()


PAGES = {
    "Dashboard": dashboard_page,
    "Live Recognition": live_recognition_page,
    "Enrollment": enrollment_page,
    "Analytics": analytics_page,
    "Logs": logs_page,
    "Settings": settings_page,
    "About": about_page,
}


def main() -> None:
    try:
        initialize_session()
        load_css()

        page = render_sidebar()

        PAGES.get(
            page,
            lambda: st.error("Page not found."),
        )()

    except Exception as error:
        st.error("An unexpected error occurred.")
        st.exception(error)


if __name__ == "__main__":
    main()
