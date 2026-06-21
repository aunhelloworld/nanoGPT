"""Import tab: URL download and file upload with confirm-before-convert."""

import streamlit as st

from language import t
from lib import converters, registry
from lib.url_utils import format_urls_one_per_line, normalize_urls
from ui.state import get_lang


def _show_next_step(lang: str):
    st.info(t("import.next_step", lang))
    if st.button(t("import.go_data", lang), key="import_go_data", use_container_width=True):
        st.session_state.nav = "data"
        st.rerun()


def render():
    lang = get_lang()

    st.caption(t("import.flow_hint", lang))

    # Mutate widget-backed state before the text_area is instantiated.
    if st.session_state.pop("_import_urls_normalize", False):
        raw = st.session_state.get("import_urls_input", "")
        normalized = format_urls_one_per_line(raw)
        st.session_state.import_urls_input = normalized
        st.session_state.import_urls_ready = bool(normalize_urls(normalized))
    if st.session_state.pop("_import_urls_clear", False):
        st.session_state.import_urls_input = ""
        st.session_state.import_urls_ready = False

    # --- URLs ---
    st.subheader(t("import.url_title", lang))
    urls_text = st.text_area(
        t("import.url_label", lang),
        placeholder=t("import.url_placeholder", lang),
        height=120,
        key="import_urls_input",
    )

    pc1, pc2 = st.columns(2)
    with pc1:
        if pc1.button(t("import.url_prepare", lang), use_container_width=True, key="url_prepare"):
            st.session_state._import_urls_normalize = True
            st.rerun()
    with pc2:
        if pc2.button(t("import.url_clear", lang), use_container_width=True, key="url_clear"):
            st.session_state._import_urls_clear = True
            st.rerun()

    prepared = normalize_urls(st.session_state.get("import_urls_input", urls_text))
    if prepared and st.session_state.get("import_urls_ready"):
        with st.expander(t("import.url_preview", lang, count=len(prepared)), expanded=True):
            for i, url in enumerate(prepared, 1):
                st.markdown(f"{i}. `{url}`")

    elif prepared and not st.session_state.get("import_urls_ready"):
        st.caption(t("import.url_prepare_hint", lang))

    if st.session_state.get("import_urls_ready") and prepared:
        if st.button(
            f"✅ {t('import.url_confirm', lang)}",
            type="primary",
            use_container_width=True,
            key="url_confirm_convert",
        ):
            progress = st.progress(0)
            results = []
            for i, url in enumerate(prepared):
                try:
                    _, stats = converters.download_and_convert(url)
                    results.append(("ok", stats))
                    st.success(t(
                        "import.url_success", lang,
                        status="✅", name=stats["filename"],
                        chars=stats["chars"], format=stats["format"],
                    ))
                except Exception as e:
                    results.append(("err", url, str(e)))
                    st.error(t(
                        "import.url_error", lang,
                        status="❌", name=url[:60], detail=str(e),
                    ))
                progress.progress((i + 1) / len(prepared))
            st.session_state.import_url_results = results
            st.session_state.import_urls_ready = False
            if any(r[0] == "ok" for r in results):
                _show_next_step(lang)

    st.divider()

    # --- Upload ---
    st.subheader(t("import.upload_title", lang))
    uploaded_files = st.file_uploader(
        t("import.upload_label", lang),
        accept_multiple_files=True,
        type=["txt", "md", "html", "htm", "xml", "pdf", "epub", "docx",
              "zip", "7z", "tar", "gz", "bz2", "xz"],
        key="import_file_uploader",
    )

    if uploaded_files:
        st.session_state.import_upload_pending = [f.name for f in uploaded_files]
        with st.expander(
            t("import.upload_preview", lang, count=len(uploaded_files)),
            expanded=True,
        ):
            for i, uf in enumerate(uploaded_files, 1):
                st.markdown(f"{i}. **{uf.name}** ({uf.size:,} bytes)")
    else:
        st.session_state.import_upload_pending = []

    upload_ready = bool(uploaded_files)
    if upload_ready:
        st.caption(t("import.upload_confirm_hint", lang))
        if st.button(
            f"✅ {t('import.upload_confirm', lang)}",
            type="primary",
            use_container_width=True,
            key="upload_confirm_convert",
        ):
            progress = st.progress(0)
            any_ok = False
            for i, uf in enumerate(uploaded_files):
                try:
                    _, stats = converters.upload_and_convert(uf)
                    any_ok = True
                    st.success(t(
                        "import.upload_success", lang,
                        name=stats["filename"], chars=stats["chars"], format=stats["format"],
                    ))
                except Exception as e:
                    st.error(t("import.upload_error", lang, name=uf.name, detail=str(e)))
                progress.progress((i + 1) / len(uploaded_files))
            st.session_state.import_upload_pending = []
            if any_ok:
                _show_next_step(lang)

    if registry.list_txt_files() and not upload_ready and not prepared:
        st.caption(t("import.has_files_hint", lang))

    with st.expander(t("import.sources_title", lang)):
        st.markdown(t("import.sources_body", lang))
