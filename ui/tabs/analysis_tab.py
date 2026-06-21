"""Analysis tab — updated via app-level autorefresh while training."""

import streamlit as st

from language import t
from lib import registry, training
from ui.state import get_lang
from ui.training_state import is_training_live

_STATUS = {"completed": "✅", "running": "🔄", "failed": "❌", "stopped": "⏹️"}


def _render_dashboard(lang: str, *, tail: int | None = 200):
    train_loss, val_loss = training.read_loss_data()
    latest = training.get_latest_train_iter()
    best = training.get_best_val_loss()

    c1, c2, c3 = st.columns(3)
    c1.metric(t("analysis.latest_iter", lang), latest)
    c2.metric(t("analysis.best_val", lang), f"{best:.4f}" if best is not None else "—")
    if is_training_live():
        c3.metric(t("analysis.status", lang), f"🔄 {t('analysis.status_training', lang)}")

    if train_loss:
        data = train_loss[-tail:] if tail else train_loss
        st.line_chart({"train": {d["iter"]: d["loss"] for d in data}}, height=200)
    if val_loss:
        st.line_chart({"val": {d["iter"]: d["loss"] for d in val_loss}}, height=200)

    st.code(training.read_log_tail(60) or t("analysis.waiting_log", lang), language="text")


def render(poll_training_fn):
    lang = get_lang()

    if is_training_live():
        poll_training_fn()
        _render_dashboard(lang)
    else:
        train_loss, val_loss = training.read_loss_data()
        if train_loss or val_loss:
            _render_dashboard(lang, tail=None)
        else:
            st.info(t("analysis.no_loss", lang))

    with st.expander(t("analysis.history_title", lang), expanded=False):
        runs = registry.load_registry().get("runs", [])
        if not runs:
            st.caption(t("analysis.no_history", lang))
        else:
            for run in reversed(runs[-8:]):
                status = run.get("status", "?")
                icon = _STATUS.get(status, "?")
                st.caption(t(
                    "analysis.run_line", lang,
                    icon=icon, id=run["id"], iters=run.get("iters"),
                    init_from=run.get("init_from"),
                    val_loss=run.get("final_val_loss", "—"),
                    status=t(f"status.{status}", lang) if status in _STATUS else status,
                ))
