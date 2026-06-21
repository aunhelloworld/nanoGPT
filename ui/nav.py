"""Navigation — pills (compact) with single-page render."""

import streamlit as st

from language import t
from ui.state import NAV_PAGES, get_lang

_ICONS = {
    "guide": "📚",
    "import": "📥",
    "data": "📊",
    "train": "📈",
    "analysis": "📉",
    "chat": "💬",
    "tools": "🛠️",
}


def _label(key: str, lang: str) -> str:
    return f"{_ICONS[key]} {t(f'tabs.{key}', lang)}"


def render_nav() -> str:
    lang = get_lang()
    current = st.session_state.get("nav", "guide")

    if hasattr(st, "pills"):
        picked = st.pills(
            "Navigation",
            options=list(NAV_PAGES),
            default=current,
            format_func=lambda k: _label(k, lang),
            label_visibility="collapsed",
            key="main_nav_pills",
        )
        if picked and picked != current:
            st.session_state.nav = picked
            st.rerun()
        elif picked:
            st.session_state.nav = picked
    else:
        cols = st.columns(len(NAV_PAGES))
        for col, key in zip(cols, NAV_PAGES):
            if col.button(
                _label(key, lang),
                key=f"nav_{key}",
                use_container_width=True,
                type="primary" if key == current else "secondary",
            ):
                st.session_state.nav = key
                st.rerun()

    return st.session_state.nav
