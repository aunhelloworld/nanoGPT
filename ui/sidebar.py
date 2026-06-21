"""Sidebar: language selector, training controls, reset."""

import streamlit as st

from language import available_languages, language_label, t
from lib import registry, training
from ui.state import get_lang


def render_sidebar():
    lang = get_lang()

    with st.sidebar:
        st.header(f"⚙️ {t('sidebar.title', lang)}")

        langs = available_languages()
        labels = {code: language_label(code, lang) for code in langs}
        selected = st.selectbox(
            t("lang.label", lang),
            options=langs,
            format_func=lambda c: labels.get(c, c),
            index=langs.index(lang) if lang in langs else 0,
            key="lang_selector",
        )
        if selected != lang:
            st.session_state.lang = selected
            st.rerun()

        lang = get_lang()
        training_active = training.is_training()

        if training_active:
            st.warning(f"🔄 {t('sidebar.training', lang)}")
            latest = training.get_latest_train_iter()
            st.caption(t("sidebar.latest_iter", lang, iter=latest))
        else:
            st.success(f"✅ {t('sidebar.ready', lang)}")

        if st.button(f"🔄 {t('sidebar.refresh', lang)}", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

        if st.button(
            f"🛑 {t('sidebar.stop_train', lang)}",
            use_container_width=True,
            disabled=not training_active,
        ):
            training.kill_train_process()
            if st.session_state.train_run_id:
                registry.finish_run(st.session_state.train_run_id, status="stopped")
                st.session_state.train_run_id = None
                st.session_state.train_proc = None
            st.rerun()

        st.divider()
        if st.button(f"🗑️ {t('sidebar.reset_all', lang)}", use_container_width=True, type="primary"):
            st.session_state.show_reset_confirm = True

        if st.session_state.show_reset_confirm:
            st.error(f"⚠️ {t('sidebar.reset_warning', lang)}")
            confirm = st.text_input(t("sidebar.reset_confirm", lang))
            c1, c2 = st.columns(2)
            with c1:
                if st.button(t("sidebar.reset_confirm_btn", lang), type="primary"):
                    if confirm == "RESET":
                        training.reset_all()
                        st.session_state.msgs = []
                        st.session_state.train_run_id = None
                        st.session_state.train_proc = None
                        st.session_state.show_reset_confirm = False
                        st.cache_resource.clear()
                        st.rerun()
                    else:
                        st.error(t("sidebar.reset_type_error", lang))
            with c2:
                if st.button(t("sidebar.reset_cancel", lang)):
                    st.session_state.show_reset_confirm = False
                    st.rerun()

        st.divider()
        device = training.default_device()
        st.caption(f"{t('sidebar.device', lang)}: **{device}**")
        ckpt_key = "sidebar.checkpoint_yes" if training.has_checkpoint() else "sidebar.checkpoint_no"
        st.caption(t(ckpt_key, lang))
