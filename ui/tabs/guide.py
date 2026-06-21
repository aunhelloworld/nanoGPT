"""Guide tab: workflow, formats, CLI, FAQ."""

import streamlit as st

from language import t
from ui.state import get_lang


def render():
    lang = get_lang()
    st.subheader(t("guide.title", lang))
    st.markdown(t("guide.steps", lang))

    st.subheader(t("guide.formats_title", lang))
    st.markdown(t("guide.formats_table", lang))

    st.subheader(t("guide.cli_title", lang))
    cli_cmds = {
        "guide.cli_prepare": "python prepare_char.py my_ai",
        "guide.cli_train_cpu": "python train.py --dataset=my_ai --out_dir=out-my_ai --device=cpu --compile=False --max_iters=5000",
        "guide.cli_train_resume": "python train.py --dataset=my_ai --out_dir=out-my_ai --init_from=resume --device=cpu --compile=False",
        "guide.cli_sample": "python sample.py --out_dir=out-my_ai --device=cpu",
    }
    for label_key, cmd in cli_cmds.items():
        st.code(f"# {t(label_key, lang)}\n{cmd}", language="bash")

    with st.expander(t("guide.faq_title", lang)):
        st.markdown(t("guide.faq_body", lang))

    st.divider()
    st.subheader(t("guide.credits_title", lang))
    st.markdown(t("guide.credits_body", lang))
