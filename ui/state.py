"""Streamlit session state initialization."""

import streamlit as st

from language import DEFAULT_LANG


def init_session_state():
    defaults = {
        "msgs": [],
        "train_run_id": None,
        "train_proc": None,
        "show_reset_confirm": False,
        "lang": DEFAULT_LANG,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_lang() -> str:
    return st.session_state.get("lang", DEFAULT_LANG)
