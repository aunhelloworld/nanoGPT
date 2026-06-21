"""App workflow state and step indicator."""

import streamlit as st

from language import t
from lib import registry, training
from ui.state import get_lang
from ui.training_state import is_training_live

STEPS = ("import", "train", "chat")


def get_workflow_state() -> str:
    if is_training_live():
        return "training"
    if training.has_checkpoint():
        return "trained"
    if registry.list_txt_files():
        return "has_data"
    return "empty"


def render_workflow_banner():
    lang = get_lang()
    state = get_workflow_state()

    if state == "training":
        st.info(t("workflow.status_training", lang))
    elif state == "trained":
        st.success(t("workflow.status_trained", lang))
    elif state == "has_data":
        st.info(t("workflow.status_has_data", lang))
    else:
        st.info(t("workflow.status_empty", lang))

    cols = st.columns(3)
    current_idx = {
        "empty": 0,
        "has_data": 1,
        "training": 1,
        "trained": 2,
    }[state]

    labels = [t(f"workflow.step_{s}", lang) for s in STEPS]
    for i, (col, label) in enumerate(zip(cols, labels)):
        if i == current_idx:
            col.markdown(f"**→ {i + 1}. {label}**")
        elif i < current_idx:
            col.markdown(f"✓ {i + 1}. {label}")
        else:
            col.markdown(f"{i + 1}. {label}")

    st.divider()
