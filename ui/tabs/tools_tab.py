"""Tools tab — each tool explained in detail."""

import streamlit as st

from config import AI_NAME, OUT_DIR
from language import t
from lib import converters, diagnostics, registry
from ui.state import get_lang


def _tool_block(title_key: str, desc_key: str, render_action):
    lang = get_lang()
    with st.expander(t(title_key, lang), expanded=False):
        st.markdown(t(desc_key, lang))
        render_action(lang)


def render():
    lang = get_lang()
    st.markdown(t("tools.intro", lang))

    def diagnostics_action(_lang):
        if st.button(f"🔍 {t('tools.diagnostics_btn', _lang)}", key="tool_diag"):
            for r in diagnostics.run_diagnostics():
                icon = "✅" if r["ok"] else "❌"
                st.write(f"{icon} **{r['category']}/{r['name']}** — {r['detail']}")

    _tool_block("tools.diagnostics_title", "tools.diagnostics_desc", diagnostics_action)

    def convert_action(_lang):
        test_file = st.file_uploader(t("tools.test_convert_label", _lang), key="test_convert")
        if test_file and st.button(t("tools.test_convert_btn", _lang), key="tool_convert"):
            text, err = converters.convert_for_preview(test_file)
            if err:
                st.error(err)
            else:
                st.success(t("tools.test_convert_success", _lang, chars=len(text)))
                st.text(text[:2000])

    _tool_block("tools.test_convert_title", "tools.test_convert_desc", convert_action)

    def validate_action(_lang):
        if st.button(t("tools.validate_btn", _lang), key="tool_validate"):
            ok, result = diagnostics.validate_dataset(AI_NAME)
            if ok:
                st.json(result)
            else:
                st.warning(result)

    _tool_block("tools.validate_title", "tools.validate_desc", validate_action)

    def sample_action(_lang):
        sample_prompt = st.text_input("Prompt", value="The ", key="tool_sample_prompt")
        sc1, sc2 = st.columns(2)
        s_temp = sc1.slider("temperature", 0.1, 2.0, 0.8, key="s_temp")
        s_tokens = sc2.number_input("max_tokens", 20, 500, 100, key="s_tokens")
        if st.button(f"🎲 {t('tools.sample_btn', _lang)}", key="tool_sample"):
            text, err = diagnostics.generate_sample(
                OUT_DIR, prompt=sample_prompt, max_new_tokens=s_tokens, temperature=s_temp,
            )
            if err:
                st.error(err)
            else:
                st.text(text)

    _tool_block("tools.sample_title", "tools.sample_desc", sample_action)

    def registry_action(_lang):
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                f"📥 {t('tools.export_registry', _lang)}",
                registry.export_registry_json(),
                file_name="registry.json",
                mime="application/json",
                key="tool_export_reg",
            )
        with c2:
            if st.button(f"🧹 {t('tools.clean_temp', _lang)}", key="tool_clean"):
                removed = diagnostics.clean_temp_files()
                items = ", ".join(removed) if removed else t("tools.clean_temp_none", _lang)
                st.success(t("tools.clean_temp_success", _lang, items=items))
        import_json = st.file_uploader(t("tools.import_registry_label", _lang), type=["json"], key="import_reg")
        if import_json and st.button(t("tools.import_registry_btn", _lang), key="tool_import_reg"):
            try:
                registry.import_registry_json(import_json.getvalue().decode("utf-8"))
                st.success(t("tools.import_registry_success", _lang))
            except Exception as e:
                msg = str(e)
                if msg == "error.invalid_registry":
                    st.error(t("error.invalid_registry", _lang))
                else:
                    st.error(msg)

    _tool_block("tools.registry_title", "tools.registry_desc", registry_action)
