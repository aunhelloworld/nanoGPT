"""Analysis tab — progress, ETA, loss charts, smart failure recovery."""

import streamlit as st

from language import t
from lib import cleanup, registry, training
from ui.state import get_lang
from ui.training_state import is_training_live

_STATUS = {"completed": "✅", "running": "🔄", "failed": "❌", "stopped": "⏹️"}


def _apply_suggested(analysis: dict):
    st.session_state.suggested_train_config = analysis["suggested_config"]
    st.session_state.suggested_train_analysis = analysis
    st.session_state.nav = "train"
    st.rerun()


def _render_progress(lang: str):
    if not is_training_live():
        return
    prog = training.get_progress_info()
    if prog["max_iters"] <= 0:
        return
    st.progress(prog["fraction"], text=t(
        "analysis.progress", lang,
        percent=prog["percent"],
        latest=prog["latest"],
        max_iters=prog["max_iters"],
        eta=prog["eta"],
    ))


def _render_failed_banner(lang: str):
    info = st.session_state.get("_train_failed_info")
    if not info:
        return

    analysis = info.get("analysis") or {}
    category = analysis.get("category", "unknown")
    exc = analysis.get("exception", "")

    st.error(t("train.failed_banner", lang))
    st.markdown(f"**{t(f'fail.cat.{category}', lang, exc=exc[:120] if exc else '—')}**")
    if exc:
        st.caption(t("fail.exception", lang, exc=exc))

    ctx = analysis.get("context") or {}
    if ctx:
        ctx_parts = []
        for k in ("device", "init_from", "batch_size", "vocab_size", "params", "tokens_per_iter"):
            if k in ctx:
                ctx_parts.append(f"{k}={ctx[k]}")
        if ctx_parts:
            st.caption(t("fail.log_context", lang, ctx=", ".join(ctx_parts)))

    st.markdown(f"**{t('fail.recommendations_title', lang)}**")
    for i, rec in enumerate(analysis.get("recommendations", []), 1):
        st.markdown(f"{i}. {t(rec['key'], lang, **rec.get('kwargs', {}))}")

    changes = analysis.get("changes") or []
    if changes:
        st.markdown(f"**{t('fail.suggested_changes', lang)}**")
        for ch in changes:
            st.markdown(f"- `{ch['param']}`: **{ch['from']}** → **{ch['to']}**")

    with st.expander(t("fail.log_expand", lang), expanded=False):
        st.code(info.get("log", "")[:4000] or t("analysis.log_empty", lang), language="text")

    bc1, bc2, bc3 = st.columns(3)
    if bc1.button(t("fail.apply_suggested", lang), type="primary", key="fail_apply"):
        _apply_suggested(analysis)
    if analysis.get("delete_checkpoint") and bc2.button(t("fail.delete_ckpt", lang), key="fail_del_ckpt"):
        try:
            cleanup.delete_checkpoints()
            st.success(t("fail.delete_ckpt_done", lang))
            suggested = dict(analysis.get("suggested_config") or {})
            suggested["init_from"] = "scratch"
            analysis = dict(analysis)
            analysis["suggested_config"] = suggested
            st.session_state._train_failed_info["analysis"] = analysis
            st.rerun()
        except RuntimeError:
            st.error(t("error.training_in_progress", lang))
    if bc3.button(t("train.dismiss_failed", lang), key="dismiss_failed"):
        st.session_state.pop("_train_failed_info", None)
        st.rerun()


def _render_complete_sample(lang: str):
    sample = st.session_state.get("_train_complete_sample")
    if not sample:
        return
    st.success(t("train.complete_banner", lang))
    st.text_area(t("train.sample_preview", lang), sample[:800], height=120, disabled=True)
    if st.button(t("train.dismiss_sample", lang), key="dismiss_sample"):
        st.session_state.pop("_train_complete_sample", None)
        st.rerun()


def _render_dashboard(lang: str, *, tail: int | None = 200):
    train_loss, val_loss = training.read_loss_data()
    latest = training.get_latest_train_iter()
    best = training.get_best_val_loss()

    c1, c2, c3 = st.columns(3)
    c1.metric(t("analysis.latest_iter", lang), latest)
    c2.metric(t("analysis.best_val", lang), f"{best:.4f}" if best is not None else "—")
    if is_training_live():
        prog = training.get_progress_info()
        c3.metric(t("analysis.eta", lang), prog["eta"])

    _render_progress(lang)

    if train_loss:
        data = train_loss[-tail:] if tail else train_loss
        st.line_chart({"train": {d["iter"]: d["loss"] for d in data}}, height=200)
    if val_loss:
        st.line_chart({"val": {d["iter"]: d["loss"] for d in val_loss}}, height=200)

    st.code(training.read_log_tail(60) or t("analysis.waiting_log", lang), language="text")


def render(poll_training_fn):
    lang = get_lang()

    _render_complete_sample(lang)
    _render_failed_banner(lang)

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
                preset = run.get("preset") or "—"
                cfg = run.get("config") or {}
                cfg_s = ", ".join(f"{k}={v}" for k, v in list(cfg.items())[:8])
                st.caption(t(
                    "analysis.run_line", lang,
                    icon=icon, id=run["id"], iters=run.get("iters"),
                    init_from=run.get("init_from"),
                    val_loss=run.get("final_val_loss", "—"),
                    status=t(f"status.{status}", lang) if status in _STATUS else status,
                ))
                if preset != "—" or cfg_s:
                    st.caption(t("analysis.run_config", lang, preset=preset, config=cfg_s or "—"))
