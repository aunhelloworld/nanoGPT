"""Data tab: file list, metadata, preview, delete."""

import os

import streamlit as st

from language import t
from lib import registry
from ui.state import get_lang


def render():
    lang = get_lang()
    st.subheader(t("data.title", lang))
    reg = registry.load_registry()
    files = registry.list_txt_files()

    if not files:
        st.warning(t("data.empty", lang))
        return

    total_chars = 0
    for fname in files:
        stats = registry.get_file_stats(fname)
        meta = reg["files"].get(fname, {})
        total_chars += stats["chars"] if stats else 0

        label = f"📄 {fname} — {stats['chars']:,} chars" if stats else f"📄 {fname}"
        with st.expander(label):
            c1, c2, c3 = st.columns(3)
            c1.metric(t("data.chars", lang), f"{stats['chars']:,}" if stats else "?")
            c2.metric(t("data.trained", lang), meta.get("train_count", 0))
            c3.metric(t("data.source_format", lang), meta.get("format", "?"))
            if meta.get("original_name"):
                st.caption(t(
                    "data.original_file", lang,
                    name=meta["original_name"], source=meta.get("source", "?"),
                ))
            if meta.get("imported_at"):
                st.caption(t("data.imported_at", lang, time=meta["imported_at"][:19]))
            if stats:
                with open(os.path.join("data", fname), encoding="utf-8", errors="ignore") as f:
                    st.text(f.read()[:500] + "...")
            if st.button(f"🗑️ {t('data.delete', lang, name=fname)}", key=f"del_{fname}"):
                os.remove(os.path.join("data", fname))
                registry.remove_file(fname)
                st.rerun()

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric(t("data.file_count", lang), len(files))
    c2.metric(t("data.total_chars", lang), f"{total_chars:,}")
    all_chars = set()
    for fname in files:
        with open(os.path.join("data", fname), encoding="utf-8", errors="ignore") as f:
            all_chars.update(f.read())
    c3.metric(t("data.vocab_est", lang), len(all_chars))
