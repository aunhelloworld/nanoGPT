"""Import tab: URL download and file upload."""

import streamlit as st

from language import t
from lib import converters
from ui.state import get_lang


def render():
    lang = get_lang()

    st.subheader(t("import.url_title", lang))
    urls_text = st.text_area(
        t("import.url_label", lang),
        placeholder=t("import.url_placeholder", lang),
        height=80,
    )
    if st.button(f"⬇️ {t('import.url_btn', lang)}", use_container_width=True):
        urls = [u.strip() for u in urls_text.strip().split("\n") if u.strip()]
        if not urls:
            st.warning(t("import.url_empty", lang))
        else:
            progress = st.progress(0)
            for i, url in enumerate(urls):
                try:
                    _, stats = converters.download_and_convert(url)
                    st.success(t(
                        "import.url_success", lang,
                        status="✅", name=stats["filename"],
                        chars=stats["chars"], format=stats["format"],
                    ))
                except Exception as e:
                    st.error(t(
                        "import.url_error", lang,
                        status="❌", name=url[:40], detail=str(e),
                    ))
                progress.progress((i + 1) / len(urls))

    st.divider()
    st.subheader(t("import.upload_title", lang))
    uploaded_files = st.file_uploader(
        t("import.upload_label", lang),
        accept_multiple_files=True,
        type=["txt", "md", "html", "htm", "xml", "pdf", "epub", "docx",
              "zip", "7z", "tar", "gz", "bz2", "xz"],
    )
    if uploaded_files and st.button(f"📤 {t('import.upload_btn', lang)}", use_container_width=True):
        progress = st.progress(0)
        for i, uf in enumerate(uploaded_files):
            try:
                _, stats = converters.upload_and_convert(uf)
                st.success(t(
                    "import.upload_success", lang,
                    name=stats["filename"], chars=stats["chars"], format=stats["format"],
                ))
            except Exception as e:
                st.error(t("import.upload_error", lang, name=uf.name, detail=str(e)))
            progress.progress((i + 1) / len(uploaded_files))

    with st.expander(t("import.sources_title", lang)):
        st.markdown(t("import.sources_body", lang))
