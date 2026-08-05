from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import pandas as pd
import streamlit as st


def page_header(
    title: str,
    subtitle: str = "",
    icon: str = "📊",
) -> None:
    st.title(f"{icon} {title}")

    if subtitle:
        st.caption(subtitle)


def section_title(
    title: str,
    subtitle: str = "",
) -> None:
    st.markdown(
        f"""
        <h2 style="color:white;margin-bottom:0;">
            {title}
        </h2>

        <p style="
            color:#94A3B8;
            margin-top:5px;
        ">
            {subtitle}
        </p>
        """,
        unsafe_allow_html=True,
    )


def divider() -> None:
    st.markdown("<hr>", unsafe_allow_html=True)


def spacer(height: int = 20) -> None:
    st.markdown(
        f"<div style='height:{height}px'></div>",
        unsafe_allow_html=True,
    )


def progress_indicator(
    value: int,
    text: str = "Processing...",
) -> None:
    value = max(0, min(100, value))

    st.progress(value)

    st.caption(f"{text} • {value}%")


@contextmanager
def loading_spinner(text: str = "Loading...") -> Generator[None, None, None]:
    with st.spinner(text):
        yield


def primary_button(
    label: str,
    key: str | None = None,
    disabled: bool = False,
    use_container_width: bool = True,
) -> bool:
    return st.button(
        label,
        key=key,
        type="primary",
        disabled=disabled,
        use_container_width=use_container_width,
    )


def secondary_button(
    label: str,
    key: str | None = None,
    disabled: bool = False,
    use_container_width: bool = True,
) -> bool:
    return st.button(
        label,
        key=key,
        disabled=disabled,
        use_container_width=use_container_width,
    )


def icon_button(
    icon: str,
    label: str,
    key: str | None = None,
    disabled: bool = False,
) -> bool:
    return st.button(
        f"{icon} {label}",
        key=key,
        disabled=disabled,
        use_container_width=True,
    )


def two_column_buttons(
    left_label: str,
    right_label: str,
) -> tuple[bool, bool]:
    col1, col2 = st.columns(2)

    with col1:
        left = st.button(
            left_label,
            use_container_width=True,
        )

    with col2:
        right = st.button(
            right_label,
            use_container_width=True,
        )

    return left, right


def search_box(
    placeholder: str = "Search...",
    key: str | None = None,
) -> str:
    return st.text_input(
        label="Search",
        placeholder=placeholder,
        key=key,
        label_visibility="collapsed",
    )


def upload_widget(
    label: str = "Upload Image",
    file_types: list[str] | None = None,
):
    return st.file_uploader(
        label,
        type=file_types or ["jpg", "jpeg", "png"],
    )


def confirmation_box(
    message: str = "Confirm action",
) -> bool:
    return st.checkbox(message)


def metric_card(
    title: str,
    value: str,
    delta: str = "",
    icon: str = "📊",
) -> None:
    st.markdown(
        f"""
        <div style="
            background:#1E293B;
            padding:18px;
            border-radius:16px;
            border:1px solid #334155;
            box-shadow:0 4px 10px rgba(0,0,0,0.25);
        ">
            <div style="font-size:14px;color:#94A3B8;">
                {icon} {title}
            </div>

            <div style="
                font-size:30px;
                font-weight:bold;
                color:white;
                margin-top:6px;
            ">
                {value}
            </div>

            <div style="
                color:#22C55E;
                font-size:14px;
                margin-top:5px;
            ">
                {delta}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(
    label: str,
    status: str,
) -> None:
    colors = {
        "Online": "#22C55E",
        "Offline": "#EF4444",
        "Running": "#3B82F6",
        "Stopped": "#F59E0B",
        "Warning": "#F59E0B",
    }

    color = colors.get(status, "#64748B")

    st.markdown(
        f"""
        <span style="
            background:{color};
            color:white;
            padding:8px 14px;
            border-radius:20px;
            font-size:13px;
            font-weight:600;
        ">
            {label}: {status}
        </span>
        """,
        unsafe_allow_html=True,
    )


def alert_box(message: str) -> None:
    st.warning(message)


def success_message(message: str) -> None:
    st.success(message)


def error_message(message: str) -> None:
    st.error(message)


def info_card(
    title: str,
    description: str,
    icon: str = "ℹ️",
) -> None:
    st.markdown(
        f"""
        <div style="
            background:#172033;
            border-left:5px solid #2563EB;
            padding:18px;
            border-radius:14px;
            margin-bottom:10px;
        ">
            <h4 style="margin:0;color:white;">
                {icon} {title}
            </h4>

            <p style="
                margin-top:10px;
                color:#CBD5E1;
                line-height:1.6;
            ">
                {description}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def toast_success(message: str) -> None:
    st.toast(message, icon="✅")


def toast_error(message: str) -> None:
    st.toast(message, icon="❌")


def toast_info(message: str) -> None:
    st.toast(message, icon="ℹ️")


def dataframe_table(
    data: pd.DataFrame,
) -> None:
    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True,
    )


def empty_state(
    title: str = "No Data Available",
    description: str = "Nothing to display.",
    icon: str = "📂",
) -> None:
    st.markdown(
        f"""
        <div style="
            text-align:center;
            padding:50px;
            border-radius:16px;
            border:2px dashed #475569;
            background:#111827;
        ">
            <div style="font-size:48px;">
                {icon}
            </div>

            <h3 style="color:white;">
                {title}
            </h3>

            <p style="color:#94A3B8;">
                {description}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def download_button(
    label: str,
    data,
    file_name: str,
    mime: str = "text/plain",
) -> None:
    st.download_button(
        label=label,
        data=data,
        file_name=file_name,
        mime=mime,
        use_container_width=True,
    )
