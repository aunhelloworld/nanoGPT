"""Unified training-active detection for the UI."""

import streamlit as st

from lib import training


def is_training_live() -> bool:
    proc = st.session_state.get("train_proc")
    if proc is not None and proc.poll() is None:
        return True
    return training.is_training()
