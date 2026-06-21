"""Chat tab — settings outside fragment; messages inside fragment only."""

import streamlit as st
import torch

from language import t
from ui.components import scroll_chat_to_bottom
from ui.live import fragment_rerun
from ui.model_loader import load_model
from ui.state import get_lang


def render():
    lang = get_lang()
    model, stoi, itos, info = load_model()

    if model is None:
        st.warning(t("chat.not_trained", lang))
        return

    if info:
        m1, m2, m3, m4, m5 = st.columns([1, 1, 1, 1, 1])
        m1.metric("Layers", info.get("n_layer"))
        m2.metric("Params", f"{info.get('params', 0):,}")
        m3.metric("Iter", info.get("iter_num", 0))
        m4.metric("Best val", f"{info.get('best_val_loss', 0):.4f}")
        if m5.button(f"🗑️ {t('chat.clear', lang)}", use_container_width=True, key="chat_clear"):
            st.session_state.msgs = []
            st.rerun()

    with st.expander("Generation settings", expanded=False):
        s1, s2, s3 = st.columns(3)
        s1.slider("temperature", 0.1, 2.0, 0.8, key="chat_temp")
        s2.slider("top_k", 1, 100, 40, key="chat_topk")
        s3.slider("max_tokens", 20, 500, 100, key="chat_max")

    @st.fragment
    def _chat_messages():
        with st.container(height=420, border=False):
            for m in st.session_state.msgs:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

        if prompt := st.chat_input(t("chat.input_placeholder", lang)):
            st.session_state.msgs.append({"role": "user", "content": prompt})
            ids = [stoi.get(c, 0) for c in prompt]
            x = torch.tensor(ids, dtype=torch.long)[None, ...]
            with st.spinner("..."):
                y = model.generate(
                    x,
                    max_new_tokens=int(st.session_state.get("chat_max", 100)),
                    temperature=float(st.session_state.get("chat_temp", 0.8)),
                    top_k=int(st.session_state.get("chat_topk", 40)),
                )
                resp = "".join(itos[i] for i in y[0].tolist())
            st.session_state.msgs.append({"role": "assistant", "content": resp})
            fragment_rerun()

        if st.session_state.msgs:
            scroll_chat_to_bottom()

    _chat_messages()
