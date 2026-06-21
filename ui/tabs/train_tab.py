"""Train tab: presets, hyperparameters, start training."""

import time

import streamlit as st

from config import AI_NAME
from language import t
from lib import registry, training
from ui.state import get_lang

PRESET_KEYS = ["fast", "normal", "quality", "custom"]


def _preset_label(key: str, lang: str) -> str:
    if key == "custom":
        return t("preset.custom", lang)
    return t(f"preset.{key}", lang)


def render():
    lang = get_lang()
    st.subheader(t("train.title", lang))
    files = registry.list_txt_files()
    has_ckpt = training.has_checkpoint()
    training_active = training.is_training()

    if not files:
        st.warning(t("train.no_files", lang))
    else:
        st.info(t("train.merge_info", lang, count=len(files), name=AI_NAME))

    preset_key = st.selectbox(
        t("train.preset", lang),
        PRESET_KEYS,
        format_func=lambda k: _preset_label(k, lang),
    )
    preset_vals = training.PRESETS.get(preset_key, {}) if preset_key != "custom" else {}

    c1, c2, c3 = st.columns(3)
    max_iters = c1.number_input("max_iters", 100, 100000, preset_vals.get("max_iters", 5000), 100)
    n_layer = c2.slider("n_layer", 1, 8, preset_vals.get("n_layer", 3))
    n_head = c3.slider("n_head", 1, 12, preset_vals.get("n_head", 6))

    embd_options = [128, 256, 384, 512, 768]
    block_options = [16, 32, 64, 128, 256]
    embd_default = preset_vals.get("n_embd", 384)
    block_default = preset_vals.get("block_size", 32)

    c4, c5, c6 = st.columns(3)
    n_embd = c4.selectbox(
        "n_embd", embd_options,
        index=embd_options.index(embd_default) if embd_default in embd_options else 2,
    )
    block_size = c5.selectbox(
        "block_size", block_options,
        index=block_options.index(block_default) if block_default in block_options else 1,
    )
    batch_size = c6.number_input("batch_size", 1, 64, 8)

    c7, c8, c9 = st.columns(3)
    learning_rate = c7.number_input("learning_rate", 1e-5, 1e-2, 1e-3, format="%.5f")
    dropout = c8.slider("dropout", 0.0, 0.5, 0.2)
    device = c9.selectbox(
        "device", ["auto", "cpu", "cuda"],
        index=0 if training.default_device() == "cuda" else 1,
    )

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

    fresh_loss = st.checkbox(t("train.clear_loss", lang), value=(init_from == "scratch"))

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
        disabled=(not files or training_active),
    ):
        proc, result = training.start_training(config, fresh_loss_log=fresh_loss)
        if proc is None:
            if isinstance(result, tuple):
                err_key, detail = result
                st.error(t(err_key, lang, detail=detail))
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
            st.success(t("train.started", lang, pid=result["pid"]))
            time.sleep(1)
            st.rerun()

    if training_active:
        st.warning(t("train.already_running", lang))
