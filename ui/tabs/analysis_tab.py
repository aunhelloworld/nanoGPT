"""Analysis tab: loss charts, live log, training history."""

import streamlit as st

from language import t
from lib import registry, training
from ui.state import get_lang

STATUS_ICONS = {"completed": "✅", "running": "🔄", "failed": "❌", "stopped": "⏹️"}


def _estimate_eta(current_iter, max_iters, train_loss):
    """Rough ETA from recent iteration pace."""
    if len(train_loss) < 2 or current_iter <= 0:
        return None
    recent = train_loss[-20:]
    if len(recent) < 2:
        return None
    iter_span = recent[-1]["iter"] - recent[0]["iter"]
    if iter_span <= 0:
        return None
    remaining = max_iters - current_iter
    if remaining <= 0:
        return "0 min"
    # Assume ~10 iters per log line interval; rough minutes estimate
    steps_per_iter = iter_span / max(len(recent) - 1, 1)
    est_iters_left = remaining
    est_minutes = int(est_iters_left / max(steps_per_iter, 1) * 0.5)
    if est_minutes < 1:
        return "< 1 min"
    if est_minutes > 120:
        return f"~{est_minutes // 60}h {est_minutes % 60}m"
    return f"~{est_minutes} min"


def _render_charts(lang, live=False):
    train_loss, val_loss = training.read_loss_data()
    latest = training.get_latest_train_iter()
    best = training.get_best_val_loss()

    c1, c2, c3 = st.columns(3)
    c1.metric(t("analysis.latest_iter", lang), latest)
    c2.metric(t("analysis.best_val", lang), f"{best:.4f}" if best else "—")
    if live:
        c3.metric(t("analysis.status", lang), f"🔄 {t('analysis.status_training', lang)}")
        active_run = registry.get_active_run()
        if active_run and train_loss:
            eta = _estimate_eta(latest, active_run.get("iters", 0), train_loss)
            if eta:
                st.caption(f"{t('analysis.eta', lang)}: {eta}")

    if train_loss:
        data = train_loss[-200:] if live else train_loss
        st.line_chart({"train": {d["iter"]: d["loss"] for d in data}})
    if val_loss:
        st.line_chart({"val": {d["iter"]: d["loss"] for d in val_loss}})

    log = training.read_log_tail(60 if live else 100)
    placeholder = t("analysis.waiting_log", lang) if live else t("analysis.log_empty", lang)
    if live:
        st.code(log or placeholder, language="text")
    else:
        with st.expander(t("analysis.log_terminal", lang)):
            st.code(log or placeholder, language="text")


def render(poll_training_fn):
    lang = get_lang()
    st.subheader(t("analysis.title", lang))
    training_active = training.is_training()

    if training_active:
        def _live():
            poll_training_fn()
            _render_charts(lang, live=True)

        if hasattr(st, "fragment"):
            @st.fragment(run_every=2)
            def live_monitor():
                _live()
            live_monitor()
        else:
            _live()
            st.caption(t("analysis.refresh_hint", lang))
            if st.button(f"🔄 {t('analysis.refresh_btn', lang)}", key="refresh_analysis"):
                st.rerun()
    else:
        train_loss, val_loss = training.read_loss_data()
        if train_loss or val_loss:
            _render_charts(lang, live=False)
        else:
            st.info(t("analysis.no_loss", lang))

    st.divider()
    st.subheader(t("analysis.history_title", lang))
    runs = registry.load_registry().get("runs", [])
    if runs:
        for run in reversed(runs[-10:]):
            status = run.get("status", "?")
            icon = STATUS_ICONS.get(status, "?")
            st.markdown(t(
                "analysis.run_line", lang,
                icon=icon, id=run["id"], iters=run.get("iters"),
                init_from=run.get("init_from"),
                val_loss=run.get("final_val_loss", "—"),
                status=t(f"status.{status}", lang) if status in STATUS_ICONS else status,
            ))
    else:
        st.caption(t("analysis.no_history", lang))
