"""Backup, export/import model, and selective cleanup."""

import streamlit as st

from language import t
from lib import cleanup, model_io, registry
from lib import training
from ui.state import get_lang


def render():
    lang = get_lang()

    st.markdown(t("manage.intro", lang))

    # --- 1. Backup & export ---
    st.markdown(f"#### 1. {t('manage.section_backup', lang)}")
    st.caption(t("manage.backup_hint", lang))

    c1, c2 = st.columns(2)
    with c1:
        if model_io.can_export():
            st.download_button(
                f"📦 {t('manage.export_model', lang)}",
                model_io.export_model_zip(),
                file_name=model_io.ZIP_NAME,
                mime="application/zip",
                use_container_width=True,
                key="manage_export_model",
            )
        else:
            st.caption(t("manage.no_checkpoint", lang))
    with c2:
        st.download_button(
            f"📥 {t('manage.export_registry', lang)}",
            registry.export_registry_json(),
            file_name="registry.json",
            mime="application/json",
            use_container_width=True,
            key="manage_export_registry",
        )

    # --- 2. Import ---
    st.markdown(f"#### 2. {t('manage.section_import', lang)}")
    ic1, ic2 = st.columns(2)
    with ic1:
        model_zip = st.file_uploader(t("manage.import_model_label", lang), type=["zip"], key="manage_import_model")
        if model_zip and st.button(t("manage.import_model_btn", lang), key="manage_import_model_go"):
            ok, msg = model_io.import_model_zip(model_zip.getvalue())
            if ok:
                st.success(t("manage.import_model_success", lang))
                st.cache_resource.clear()
                st.rerun()
            elif msg == "error.import_no_ckpt":
                st.error(t("error.import_no_ckpt", lang))
            elif msg == "error.import_bad_zip":
                st.error(t("error.import_bad_zip", lang))
            else:
                st.error(msg)
    with ic2:
        reg_json = st.file_uploader(t("manage.import_registry_label", lang), type=["json"], key="manage_import_reg")
        if reg_json and st.button(t("manage.import_registry_btn", lang), key="manage_import_reg_go"):
            try:
                registry.import_registry_json(reg_json.getvalue().decode("utf-8"))
                st.success(t("manage.import_registry_success", lang))
            except Exception as exc:
                msg = str(exc)
                if msg == "error.invalid_registry":
                    st.error(t("error.invalid_registry", lang))
                else:
                    st.error(msg)

    # --- 3. Selective delete ---
    st.markdown(f"#### 3. {t('manage.section_delete', lang)}")
    st.caption(t("manage.delete_hint", lang))

    options = []
    if cleanup.has_checkpoints():
        options.append("checkpoint")
    if cleanup.has_import_data():
        options.append("data")
    if cleanup.has_history():
        options.append("history")

    if not options:
        st.info(t("manage.nothing_to_delete", lang))
        return

    scope = st.selectbox(
        t("manage.delete_scope", lang),
        options,
        format_func=lambda s: t(f"manage.scope_{s}", lang),
        key="manage_delete_scope",
    )

    st.warning(t(f"manage.scope_warning_{scope}", lang))

    if st.button(t("manage.delete_btn", lang), key="manage_delete_go"):
        st.session_state.manage_delete_confirm = scope

    if st.session_state.get("manage_delete_confirm") == scope:
        confirm = st.text_input(t("manage.delete_confirm", lang))
        dc1, dc2 = st.columns(2)
        if dc1.button(t("manage.delete_confirm_btn", lang), type="primary", key="manage_delete_yes"):
            if confirm != "DELETE":
                st.error(t("manage.delete_type_error", lang))
            else:
                try:
                    if scope == "checkpoint":
                        cleanup.delete_checkpoints()
                    elif scope == "data":
                        cleanup.delete_import_data()
                    else:
                        cleanup.delete_history()
                    st.session_state.manage_delete_confirm = None
                    st.session_state.msgs = []
                    st.cache_resource.clear()
                    st.success(t("manage.delete_success", lang))
                    st.rerun()
                except RuntimeError as exc:
                    if str(exc) == "error.training_in_progress":
                        st.error(t("error.training_in_progress", lang))
                    else:
                        st.error(str(exc))
        if dc2.button(t("manage.delete_cancel", lang), key="manage_delete_no"):
            st.session_state.manage_delete_confirm = None
            st.rerun()
