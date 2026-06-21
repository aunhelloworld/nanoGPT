"""
NanoGPT Training Dashboard.

Run: streamlit run app.py
"""
import os
import sys
import time

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from language import t
from lib import diagnostics, registry, training
from lib.failure_analysis import analyze_failure
from ui import nav, sidebar
from ui.live import poll_and_autorefresh, render_status_bar, show_completion_toast
from ui.state import get_lang, init_session_state
from ui.styles import inject_styles
from ui.workflow import render_workflow_banner
from ui.tabs import analysis_tab, chat_tab, data_tab, guide, import_tab, tools_tab, train_tab

st.set_page_config(page_title="NanoGPT Training Dashboard", layout="wide", page_icon="🧠")
init_session_state()
inject_styles()


def poll_training():
    if st.session_state.train_run_id is None:
        return
    finished, ret = training.check_training_finished(st.session_state.train_proc)
    if not finished:
        return
    run_id = st.session_state.train_run_id
    lang = get_lang()
    run = registry.get_run(run_id)
    run_config = (run or {}).get("config") or {}
    if ret == 0:
        val = training.get_best_val_loss()
        registry.finish_run(run_id, final_val_loss=val, status="completed")
        run = registry.get_run(run_id) or run
        if run:
            registry.increment_train_count(run.get("files_used", []))
        loss_s = f"{val:.4f}" if val is not None else "—"
        st.session_state._train_toast = t("train.complete_toast", lang, loss=loss_s)
        sample, _err = diagnostics.generate_sample(max_new_tokens=120, temperature=0.8)
        if sample:
            st.session_state._train_complete_sample = sample
    else:
        registry.finish_run(run_id, status="failed")
        st.session_state._train_toast = t("train.failed_toast", lang)
        log = training.read_log_tail(120)
        analysis = analyze_failure(log, run_config)
        st.session_state._train_failed_info = {
            "log": log,
            "analysis": analysis,
            "hints": analysis["hints"],
            "run_config": run_config,
        }
    st.session_state.train_run_id = None
    st.session_state.train_proc = None
    st.session_state.train_start_time = None
    st.session_state.train_max_iters = None
    st.cache_resource.clear()


nav_page = st.session_state.get("nav", "guide")
poll_and_autorefresh(poll_training, nav_page)
poll_training()
show_completion_toast()

sidebar.render_sidebar()

lang = get_lang()
st.title(f"🧠 {t('app.title', lang)}")
render_workflow_banner()
render_status_bar()
active = nav.render_nav()

_RENDERERS = {
    "guide": guide.render,
    "import": import_tab.render,
    "data": data_tab.render,
    "train": train_tab.render,
    "analysis": lambda: analysis_tab.render(poll_training),
    "chat": chat_tab.render,
    "tools": tools_tab.render,
}
_RENDERERS[active]()
