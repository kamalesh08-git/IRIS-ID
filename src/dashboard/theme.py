from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Final
import streamlit as st
@dataclass(frozen=True)
class Colors:
    """Application color configuration."""

    PRIMARY: str = "#2563EB"
    PRIMARY_LIGHT: str = "#3B82F6"

    SECONDARY: str = "#7C3AED"

    SUCCESS: str = "#22C55E"
    WARNING: str = "#F59E0B"
    ERROR: str = "#EF4444"
    INFO: str = "#06B6D4"

    BACKGROUND: str = "#0F172A"
    SURFACE: str = "#1E293B"
    CARD: str = "#172033"

    BORDER: str = "#334155"

    TEXT: str = "#FFFFFF"
    SUBTEXT: str = "#CBD5E1"
    MUTED: str = "#94A3B8"

    GRID: str = "#293548"

@dataclass(frozen=True)
class Typography:
    """Typography settings."""

    FONT: str = "Inter, sans-serif"

    TITLE_SIZE: int = 32
    SUBTITLE_SIZE: int = 22
    BODY_SIZE: int = 16
    SMALL_SIZE: int = 13
@dataclass(frozen=True)
class Spacing:
    """Spacing constants."""

    XS: int = 4
    SM: int = 8
    MD: int = 16
    LG: int = 24
    XL: int = 32

@dataclass(frozen=True)
class Radius:
    """Border radius values."""

    SMALL: int = 8
    MEDIUM: int = 14
    LARGE: int = 20
    EXTRA_LARGE: int = 28

ICONS: Final = {
    "dashboard": "🏠",
    "camera": "📷",
    "analytics": "📊",
    "logs": "📜",
    "settings": "⚙️",
    "about": "ℹ️",
    "success": "✅",
    "warning": "⚠️",
    "error": "❌",
    "gpu": "🖥️",
    "cpu": "💻",
    "memory": "🧠",
    "model": "🤖",
    "face": "👤",
    "masked": "😷",
}

colors = Colors()
typography = Typography()
spacing = Spacing()
radius = Radius()

PLOTLY_LAYOUT = {
    "paper_bgcolor": colors.BACKGROUND,
    "plot_bgcolor": colors.BACKGROUND,
    "font": {
        "family": typography.FONT,
        "color": colors.TEXT,
    },
    "xaxis": {
        "gridcolor": colors.GRID,
        "zeroline": False,
    },
    "yaxis": {
        "gridcolor": colors.GRID,
        "zeroline": False,
    },
    "margin": {
        "l": 20,
        "r": 20,
        "t": 40,
        "b": 20,
    },
}


def load_css(css_file: str | Path) -> None:
    """
    Load external CSS into Streamlit.

    Parameters
    ----------
    css_file : str | Path
        Path to CSS file.
    """

    path = Path(css_file)

    if not path.exists():
        st.warning(f"CSS file not found: {path}")
        return

    with path.open(encoding="utf-8") as file:
        st.markdown(
            f"<style>{file.read()}</style>",
            unsafe_allow_html=True,
        )

def get_status_color(status: str) -> str:
    """
    Return color for a given status.

    Parameters
    ----------
    status : str

    Returns
    -------
    str
    """

    mapping = {
        "online": colors.SUCCESS,
        "offline": colors.ERROR,
        "running": colors.SUCCESS,
        "warning": colors.WARNING,
        "idle": colors.INFO,
    }

    return mapping.get(status.lower(), colors.MUTED)

def card_style() -> str:
    """
    Return inline CSS for dashboard cards.
    """

    return f"""
    background:{colors.CARD};
    border:1px solid {colors.BORDER};
    border-radius:{radius.LARGE}px;
    padding:{spacing.LG}px;
    """

def page_style() -> str:
    """
    Return page background CSS.
    """

    return f"""
    background:{colors.BACKGROUND};
    color:{colors.TEXT};
    """

def gradient() -> str:
    """
    Gradient used for headers.
    """

    return (
        f"linear-gradient(90deg,"
        f"{colors.PRIMARY},"
        f"{colors.SECONDARY})"
    )
THEME = {
    "colors": colors,
    "typography": typography,
    "spacing": spacing,
    "radius": radius,
    "icons": ICONS,
    "plotly": PLOTLY_LAYOUT,
}

__all__ = [
    "colors",
    "typography",
    "spacing",
    "radius",
    "ICONS",
    "PLOTLY_LAYOUT",
    "THEME",
    "load_css",
    "get_status_color",
    "card_style",
    "page_style",
    "gradient",
]
