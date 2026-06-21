"""Tools tab: diagnostics, test convert, validate, sample, registry."""

import streamlit as st

from config import AI_NAME, OUT_DIR
from language import t
from lib import converters, diagnostics, registry
from ui.state import get_lang


def render():
    lang = get_lang()

    st.subheader(t("tools.diagnostics_title", lang))
    if st.button(f"🔍 {t('tools.diagnostics_btn', lang)}"):
        for r in diagnostics.run_diagnostics():
            icon = "✅" if r["ok"] else "❌"
            st.write(f"{icon} **{r['category']}/{r['name']}** — {r['detail']}")

    st.divider()
    st.subheader(t("tools.test_convert_title", lang))
    test_file = st.file_uploader(t("tools.test_convert_label", lang), key="test_convert")
    if test_file and st.button(t("tools.test_convert_btn", lang)):
        text, err = converters.convert_for_preview(test_file)
        if err:
            st.error(err)
        else:
            st.success(t("tools.test_convert_success", lang, chars=len(text)))
            st.text(text[:2000])

    st.divider()
    st.subheader(t("tools.validate_title", lang))
    if st.button(t("tools.validate_btn", lang)):
        ok, result = diagnostics.validate_dataset(AI_NAME)
        if ok:
            st.json(result)
        else:
            st.warning(result)

    st.divider()
    st.subheader(t("tools.sample_title", lang))
    sample_prompt = st.text_input("Prompt", value="The ")
    sc1, sc2 = st.columns(2)
    s_temp = sc1.slider("temperature", 0.1, 2.0, 0.8, key="s_temp")
    s_tokens = sc2.number_input("max_tokens", 20, 500, 100, key="s_tokens")
    if st.button(f"🎲 {t('tools.sample_btn', lang)}"):
        text, err = diagnostics.generate_sample(
            OUT_DIR, prompt=sample_prompt, max_new_tokens=s_tokens, temperature=s_temp,
        )
        if err:
            st.error(err)
        else:
            st.text(text)

    st.divider()
    st.subheader(t("tools.registry_title", lang))
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            f"📥 {t('tools.export_registry', lang)}",
            registry.export_registry_json(),
            file_name="registry.json",
            mime="application/json",
        )
    with c2:
        if st.button(f"🧹 {t('tools.clean_temp', lang)}"):
            removed = diagnostics.clean_temp_files()
            items = ", ".join(removed) if removed else t("tools.clean_temp_none", lang)
            st.success(t("tools.clean_temp_success", lang, items=items))

    import_json = st.file_uploader(t("tools.import_registry_label", lang), type=["json"], key="import_reg")
    if import_json and st.button(t("tools.import_registry_btn", lang)):
        try:
            registry.import_registry_json(import_json.getvalue().decode("utf-8"))
            st.success(t("tools.import_registry_success", lang))
        except Exception as e:
            msg = str(e)
            if msg == "error.invalid_registry":
                st.error(t("error.invalid_registry", lang))
            else:
                st.error(msg)
