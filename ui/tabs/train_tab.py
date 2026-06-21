"""Train tab — presets + compact advanced settings."""

import streamlit as st

from config import AI_NAME
from language import t
from lib import registry, training
from ui.state import get_lang
from ui.training_state import is_training_live

PRESET_KEYS = ["fast", "normal", "quality", "custom"]


def render():
    lang = get_lang()
    files = registry.list_txt_files()
    has_ckpt = training.has_checkpoint()
    live = is_training_live()

    if not files:
        st.warning(t("train.no_files", lang))
    else:
        st.caption(t("train.merge_info", lang, count=len(files), name=AI_NAME))

    c1, c2 = st.columns([2, 1])
    with c1:
        preset_key = st.selectbox(
            t("train.preset", lang),
            PRESET_KEYS,
            format_func=lambda k: t("preset.custom" if k == "custom" else f"preset.{k}", lang),
        )
    with c2:
        init_from = st.radio(
            t("train.mode", lang),
            ["scratch", "resume"],
            format_func=lambda x: t("train.mode_scratch" if x == "scratch" else "train.mode_resume", lang),
            index=1 if has_ckpt else 0,
            horizontal=True,
        )
    if init_from == "resume" and not has_ckpt:
        st.warning(t("train.no_checkpoint", lang))
        init_from = "scratch"

    preset_vals = training.PRESETS.get(preset_key, {}) if preset_key != "custom" else {}
    fresh_loss = st.checkbox(t("train.clear_loss", lang), value=(init_from == "scratch"))

    with st.expander("Advanced hyperparameters", expanded=(preset_key == "custom")):
        r1c1, r1c2, r1c3 = st.columns(3)
        max_iters = r1c1.number_input("max_iters", 100, 100000, preset_vals.get("max_iters", 5000), 100)
        n_layer = r1c2.slider("n_layer", 1, 8, preset_vals.get("n_layer", 3))
        n_head = r1c3.slider("n_head", 1, 12, preset_vals.get("n_head", 6))

        embd_opts = [128, 256, 384, 512, 768]
        block_opts = [16, 32, 64, 128, 256]
        embd_def = preset_vals.get("n_embd", 384)
        block_def = preset_vals.get("block_size", 32)

        r2c1, r2c2, r2c3 = st.columns(3)
        n_embd = r2c1.selectbox("n_embd", embd_opts, index=embd_opts.index(embd_def) if embd_def in embd_opts else 2)
        block_size = r2c2.selectbox("block_size", block_opts, index=block_opts.index(block_def) if block_def in block_opts else 1)
        batch_size = r2c3.number_input("batch_size", 1, 64, 8)

        r3c1, r3c2, r3c3 = st.columns(3)
        learning_rate = r3c1.number_input("learning_rate", 1e-5, 1e-2, 1e-3, format="%.5f")
        dropout = r3c2.slider("dropout", 0.0, 0.5, 0.2)
        device = r3c3.selectbox("device", ["auto", "cpu", "cuda"], index=0 if training.default_device() == "cuda" else 1)

    resolved_device = training.default_device() if device == "auto" else device
    config = {
        "max_iters": max_iters,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "block_size": block_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "init_from": init_from,
        "device": resolved_device,
        "compile": resolved_device != "cpu",
        "min_lr": learning_rate / 10,
    }

    if st.button(
        f"🚀 {t('train.start', lang)}",
        use_container_width=True,
        type="primary",
        disabled=(not files or live),
    ):
        proc, result = training.start_training(config, fresh_loss_log=fresh_loss)
        if proc is None:
            if isinstance(result, tuple):
                st.error(t(result[0], lang, detail=result[1]))
            elif isinstance(result, str) and result.startswith("error."):
                st.error(t(result, lang))
            else:
                st.error(str(result))
        else:
            run_id = registry.start_run(
                iters=max_iters,
                init_from=init_from,
                files_used=result["files"],
                config={k: config[k] for k in ["max_iters", "n_layer", "n_head", "n_embd", "block_size"]},
            )
            st.session_state.train_run_id = run_id
            st.session_state.train_proc = proc
            st.session_state.nav = "analysis"
            st.rerun()

    if live:
        st.info(t("train.already_running", lang))
