"""Live training: autorefresh only (no run_every fragments — avoids stale fragment errors)."""

import streamlit as st

from language import t
from lib import training
from ui.state import get_lang
from ui.training_state import is_training_live


def fragment_rerun():
    import inspect
    try:
        if "scope" in inspect.signature(st.rerun).parameters:
            st.rerun(scope="fragment")
            return
    except (ValueError, TypeError):
        pass
    st.rerun()


def poll_and_autorefresh(poll_fn, nav: str) -> bool:
    if not is_training_live() or nav == "chat":
        return False
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=2000, limit=None, key="training_live")
    except ImportError:
        return False
    poll_fn()
    return True


def render_status_bar():
    if not is_training_live():
        return
    lang = get_lang()
    latest = training.get_latest_train_iter()
    best = training.get_best_val_loss()
    best_s = f"{best:.4f}" if best is not None else "—"
    prog = training.get_progress_info()
    pct = prog["percent"]
    eta = prog["eta"]
    st.markdown(
        f'<div class="live-bar">🔄 {t("sidebar.training", lang)} &nbsp;·&nbsp; '
        f'{pct}% &nbsp;·&nbsp; '
        f'{t("analysis.latest_iter", lang)}: <b>{latest}</b> / {prog["max_iters"]} &nbsp;·&nbsp; '
        f'{t("analysis.eta", lang)}: <b>{eta}</b> &nbsp;·&nbsp; '
        f'{t("analysis.best_val", lang)}: <b>{best_s}</b></div>',
        unsafe_allow_html=True,
    )


def show_completion_toast():
    msg = st.session_state.pop("_train_toast", None)
    if msg:
        st.toast(msg, icon="✅")
