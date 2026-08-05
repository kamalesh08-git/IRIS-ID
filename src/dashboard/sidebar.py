from __future__ import annotations

import streamlit as st

from dashboard.session import (
    get_current_page,
    set_current_page,
)


MENU_ITEMS = {
    "🏠 Dashboard": "Dashboard",
    "📷 Live Recognition": "Live Recognition",
    "👤 Enrollment": "Enrollment",
    "📊 Analytics": "Analytics",
    "📜 Logs": "Logs",
    "⚙️ Settings": "Settings",
    "ℹ️ About": "About",
}


def render_sidebar() -> str:
    """
    Render the dashboard sidebar.

    Returns
    -------
    str
        Selected page.
    """

    with st.sidebar:

        st.markdown("# 🤖 Vision-MASK")

        st.caption("AI Powered Face Recognition")

        st.divider()

        current_page = get_current_page()

        menu = list(MENU_ITEMS.keys())
        values = list(MENU_ITEMS.values())

        current_index = (
            values.index(current_page)
            if current_page in values
            else 0
        )

        selected = st.radio(
            "Navigation",
            options=menu,
            index=current_index,
            label_visibility="collapsed",
        )

        page = MENU_ITEMS[selected]
        set_current_page(page)

        st.divider()

        st.subheader("📡 System Status")

        st.success("System Online")

        st.metric(
            "Model",
            "Vision-MASK v1.0",
        )

        st.metric(
            "Camera",
            "Ready",
        )

        st.metric(
            "GPU",
            "Available",
        )

        st.divider()

        st.subheader("🎨 Theme")

        st.selectbox(
            "Mode",
            ["Dark", "Light"],
            index=0,
            disabled=True,
            label_visibility="collapsed",
        )

        st.divider()

        st.caption("Version 1.0.0")

    return page
