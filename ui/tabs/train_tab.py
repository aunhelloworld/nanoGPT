"""Train tab — presets + auto settings + advanced overrides."""

import time

import streamlit as st

from config import AI_NAME
from language import t
from lib import registry, training
from lib.auto_settings import compute_auto_settings
from lib.error_hints import hints_for_result
from ui.state import get_lang
from ui.training_state import is_training_live

PRESET_KEYS = ["auto", "fast", "normal", "quality", "custom"]


def _render_auto_panel(result: dict, lang: str):
    stats = result["stats"]
    st.info(t("auto.summary", lang, **{
        "files": stats["file_count"],
        "chars": f"{stats['total_chars']:,}",
        "vocab": stats["unique_chars"],
        "device": result["device"],
        "vram": result["vram_mb"] or "—",
    }))

    for w in result["warnings"]:
        st.warning(t(w["key"], lang, **w.get("kwargs", {})))

    st.markdown(f"**{t('auto.adjustments_title', lang)}**")
    for adj in result["adjustments"]:
        reason = t(adj["reason_key"], lang, **adj.get("kwargs", {}))
        st.markdown(f"- **{adj['param']}** → `{adj['value']}` — {reason}")


def render():
    lang = get_lang()
    files = registry.list_txt_files()
    has_ckpt = training.has_checkpoint()
    live = is_training_live()
    suggested = st.session_state.get("suggested_train_config")
    suggested_analysis = st.session_state.get("suggested_train_analysis")

    if suggested and suggested_analysis:
        cat = suggested_analysis.get("category", "unknown")
        st.warning(t("fail.suggested_banner", lang, category=t(f"fail.cat.{cat}", lang)))
        for ch in suggested_analysis.get("changes", []):
            st.markdown(f"- `{ch['param']}`: **{ch['from']}** → **{ch['to']}**")
        if st.button(t("fail.dismiss_suggested", lang), key="dismiss_suggested"):
            st.session_state.suggested_train_config = None
            st.session_state.suggested_train_analysis = None
            st.rerun()

    if not files:
        st.warning(t("train.no_files", lang))
    else:
        st.caption(t("train.merge_info", lang, count=len(files), name=AI_NAME))

    c1, c2 = st.columns([2, 1])
    preset_default = "custom" if suggested else "auto"
    with c1:
        preset_key = st.selectbox(
            t("train.preset", lang),
            PRESET_KEYS,
            index=PRESET_KEYS.index(preset_default) if preset_default in PRESET_KEYS else 0,
            format_func=lambda k: t(
                "preset.custom" if k == "custom" else f"preset.{k}", lang,
            ),
        )
        if suggested and preset_key != "custom":
            st.session_state.suggested_train_config = None
            st.session_state.suggested_train_analysis = None
    with c2:
        init_default = suggested.get("init_from", "scratch") if suggested else ("resume" if has_ckpt else "scratch")
        init_from = st.radio(
            t("train.mode", lang),
            ["scratch", "resume"],
            format_func=lambda x: t("train.mode_scratch" if x == "scratch" else "train.mode_resume", lang),
            index=0 if init_default == "scratch" else 1,
            horizontal=True,
        )
    if init_from == "resume" and not has_ckpt:
        st.warning(t("train.no_checkpoint", lang))
        init_from = "scratch"

    auto_result = None
    if preset_key == "custom":
        preset_vals = dict(suggested) if suggested else {}
    elif files:
        auto_result = compute_auto_settings(preset_key=preset_key, init_from=init_from)
        if auto_result and preset_key == "auto":
            _render_auto_panel(auto_result, lang)
        elif auto_result and preset_key != "auto":
            with st.expander(t("auto.hardware_title", lang), expanded=False):
                _render_auto_panel(auto_result, lang)
        preset_vals = auto_result["config"] if auto_result else training.PRESETS.get(preset_key, {})
    else:
        preset_vals = training.PRESETS.get(preset_key, {}) if preset_key != "custom" else {}

    fresh_loss = st.checkbox(t("train.clear_loss", lang), value=(init_from == "scratch"))

    with st.expander("Advanced hyperparameters", expanded=(preset_key == "custom" or bool(suggested))):
        if preset_key == "auto" and auto_result:
            st.caption(t("auto.advanced_hint", lang))
        if suggested and preset_key == "custom":
            st.caption(t("fail.suggested_edit_hint", lang))
        r1c1, r1c2, r1c3 = st.columns(3)
        max_iters = r1c1.number_input(
            "max_iters", 100, 100000,
            preset_vals.get("max_iters", 5000), 100,
            disabled=(preset_key == "auto"),
        )
        n_layer = r1c2.slider(
            "n_layer", 1, 8, preset_vals.get("n_layer", 3),
            disabled=(preset_key == "auto"),
        )
        n_head = r1c3.slider(
            "n_head", 1, 12, preset_vals.get("n_head", 6),
            disabled=(preset_key == "auto"),
        )

        embd_opts = [128, 256, 384, 512, 768]
        block_opts = [16, 32, 64, 128, 256]
        embd_def = preset_vals.get("n_embd", 384)
        block_def = preset_vals.get("block_size", 32)

        r2c1, r2c2, r2c3 = st.columns(3)
        n_embd = r2c1.selectbox(
            "n_embd", embd_opts,
            index=embd_opts.index(embd_def) if embd_def in embd_opts else 2,
            disabled=(preset_key == "auto"),
        )
        block_size = r2c2.selectbox(
            "block_size", block_opts,
            index=block_opts.index(block_def) if block_def in block_opts else 1,
            disabled=(preset_key == "auto"),
        )
        batch_size = r2c3.number_input(
            "batch_size", 1, 64, preset_vals.get("batch_size", 8),
            disabled=(preset_key == "auto"),
        )

        r3c1, r3c2, r3c3 = st.columns(3)
        lr_def = preset_vals.get("learning_rate", 1e-3)
        learning_rate = r3c1.number_input(
            "learning_rate", 1e-5, 1e-2, float(lr_def), format="%.5f",
            disabled=(preset_key == "auto"),
        )
        dropout = r3c2.slider(
            "dropout", 0.0, 0.5, preset_vals.get("dropout", 0.2),
            disabled=(preset_key == "auto"),
        )
        dev_opts = ["auto", "cpu", "cuda"]
        dev_def = "auto" if preset_vals.get("device", training.default_device()) == training.default_device() else preset_vals.get("device", "cpu")
        device = r3c3.selectbox(
            "device", dev_opts,
            index=dev_opts.index(dev_def) if dev_def in dev_opts else 0,
            disabled=(preset_key == "auto"),
        )
        r4c1, _ = st.columns(2)
        eval_iters = r4c1.number_input(
            "eval_iters",
            5, 500,
            int(preset_vals.get("eval_iters", 25 if preset_vals.get("device") == "cpu" else 200)),
            disabled=(preset_key == "auto"),
        )

    if preset_key == "auto" and auto_result:
        config = dict(auto_result["config"])
        config["init_from"] = init_from
    else:
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
            "compile": bool(preset_vals.get("compile", resolved_device != "cpu")) if suggested else resolved_device != "cpu",
            "min_lr": learning_rate / 10,
            "eval_iters": int(eval_iters),
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
            for hint in hints_for_result(result, config):
                st.info(t(hint, lang))
        else:
            auto_meta = None
            if auto_result:
                auto_meta = {
                    "tier": auto_result.get("tier"),
                    "adjustments": auto_result.get("adjustments"),
                    "stats": auto_result.get("stats"),
                }
            run_id = registry.start_run(
                iters=config["max_iters"],
                init_from=init_from,
                files_used=result["files"],
                config=config,
                preset=preset_key,
                auto_meta=auto_meta,
            )
            st.session_state.train_run_id = run_id
            st.session_state.train_proc = proc
            st.session_state.train_start_time = time.time()
            st.session_state.train_max_iters = config["max_iters"]
            st.session_state.suggested_train_config = None
            st.session_state.suggested_train_analysis = None
            st.session_state.nav = "analysis"
            st.rerun()

    if live:
        st.info(t("train.already_running", lang))
