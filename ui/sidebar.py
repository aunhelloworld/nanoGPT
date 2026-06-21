"""Sidebar controls."""

import os

import streamlit as st

from language import available_languages, language_label, t
from lib import model_io, registry, training
from ui.state import get_lang
from ui.training_state import is_training_live
from ui.workflow import get_workflow_state


def render_sidebar():
    lang = get_lang()

    with st.sidebar:
        st.caption(f"⚙️ {t('sidebar.title', lang)}")

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
        live = is_training_live()
        state = get_workflow_state()

        if live:
            st.warning(f"🔄 {t('sidebar.training', lang)}")
            st.caption(t("sidebar.latest_iter", lang, iter=training.get_latest_train_iter()))
        elif state == "trained":
            st.success(t("sidebar.model_ready", lang))
        elif state == "has_data":
            st.info(t("sidebar.files_ready", lang))
        else:
            st.info(t("sidebar.get_started", lang))

        if st.button(f"🔄 {t('sidebar.refresh', lang)}", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

        if live:
            if st.button(
                f"🛑 {t('sidebar.stop_train', lang)}",
                use_container_width=True,
                type="primary",
            ):
                training.kill_train_process()
                if st.session_state.train_run_id:
                    registry.finish_run(st.session_state.train_run_id, status="stopped")
                    st.session_state.train_run_id = None
                    st.session_state.train_proc = None
                    st.session_state.train_start_time = None
                    st.session_state.train_max_iters = None
                st.rerun()

        st.divider()
        st.caption(t("sidebar.danger_zone", lang))

        has_anything = (
            state != "empty"
            or training.has_checkpoint()
            or registry.list_txt_files()
            or os.path.exists("data/registry.json")
        )

        if st.button(
            f"🗑️ {t('sidebar.reset_all', lang)}",
            use_container_width=True,
            disabled=not has_anything,
            help=t("sidebar.reset_hint", lang),
        ):
            st.session_state.show_reset_confirm = True

        if st.session_state.show_reset_confirm:
            st.error(t("sidebar.reset_warning", lang))
            st.markdown(t("sidebar.reset_deletes", lang))
            st.caption(t("sidebar.backup_before_reset", lang))
            bc1, bc2 = st.columns(2)
            with bc1:
                if model_io.can_export():
                    st.download_button(
                        t("manage.export_model", lang),
                        model_io.export_model_zip(),
                        file_name=model_io.ZIP_NAME,
                        mime="application/zip",
                        use_container_width=True,
                        key="sidebar_backup_model",
                    )
            with bc2:
                if os.path.exists("data/registry.json"):
                    st.download_button(
                        t("manage.export_registry", lang),
                        registry.export_registry_json(),
                        file_name="registry.json",
                        mime="application/json",
                        use_container_width=True,
                        key="sidebar_backup_registry",
                    )
            confirm = st.text_input(t("sidebar.reset_confirm", lang))
            c1, c2 = st.columns(2)
            if c1.button(t("sidebar.reset_confirm_btn", lang), type="primary"):
                if confirm == "RESET":
                    training.reset_all()
                    st.session_state.msgs = []
                    st.session_state.train_run_id = None
                    st.session_state.train_proc = None
                    st.session_state.train_start_time = None
                    st.session_state.train_max_iters = None
                    st.session_state.show_reset_confirm = False
                    st.session_state.nav = "guide"
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.error(t("sidebar.reset_type_error", lang))
            if c2.button(t("sidebar.reset_cancel", lang)):
                st.session_state.show_reset_confirm = False
                st.rerun()

        if not has_anything:
            st.caption(t("sidebar.reset_nothing", lang))
        else:
            st.caption(t("sidebar.manage_hint", lang))

        st.divider()
        st.caption(f"{t('sidebar.device', lang)}: `{training.default_device()}`")
        ck = t("sidebar.checkpoint_yes" if training.has_checkpoint() else "sidebar.checkpoint_no", lang)
        st.caption(ck)
        if state == "empty":
            st.caption(t("sidebar.hint_import", lang))
