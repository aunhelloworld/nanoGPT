"""
NanoGPT Training Dashboard — entry point.

Run: streamlit run app.py
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from language import t
from lib import registry, training
from ui import sidebar
from ui.state import get_lang, init_session_state
from ui.tabs import analysis_tab, chat_tab, data_tab, guide, import_tab, tools_tab, train_tab

st.set_page_config(page_title="NanoGPT Training Dashboard", layout="wide", page_icon="🧠")
init_session_state()


def poll_training():
    """Check if background training finished; update registry."""
    if st.session_state.train_run_id is None:
        return
    finished, ret = training.check_training_finished(st.session_state.train_proc)
    if finished:
        run_id = st.session_state.train_run_id
        if ret == 0:
            val = training.get_best_val_loss()
            registry.finish_run(run_id, final_val_loss=val, status="completed")
            run = next((r for r in registry.load_registry()["runs"] if r["id"] == run_id), None)
            if run:
                registry.increment_train_count(run.get("files_used", []))
        else:
            registry.finish_run(run_id, status="failed")
        st.session_state.train_run_id = None
        st.session_state.train_proc = None
        st.cache_resource.clear()


poll_training()
sidebar.render_sidebar()

lang = get_lang()
st.title(f"🧠 {t('app.title', lang)}")

tabs = st.tabs([
    f"📚 {t('tabs.guide', lang)}",
    f"📥 {t('tabs.import', lang)}",
    f"📊 {t('tabs.data', lang)}",
    f"📈 {t('tabs.train', lang)}",
    f"📉 {t('tabs.analysis', lang)}",
    f"💬 {t('tabs.chat', lang)}",
    f"🛠️ {t('tabs.tools', lang)}",
])

with tabs[0]:
    guide.render()
with tabs[1]:
    import_tab.render()
with tabs[2]:
    data_tab.render()
with tabs[3]:
    train_tab.render()
with tabs[4]:
    analysis_tab.render(poll_training)
with tabs[5]:
    chat_tab.render()
with tabs[6]:
    tools_tab.render()
