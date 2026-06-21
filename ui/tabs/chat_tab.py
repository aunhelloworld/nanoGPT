"""Chat tab: interact with the trained model."""

import streamlit as st
import torch

from language import t
from ui.model_loader import load_model
from ui.state import get_lang


def render():
    lang = get_lang()
    st.subheader(t("chat.title", lang))
    model, stoi, itos, info = load_model()

    if model is None:
        st.warning(t("chat.not_trained", lang))
        return

    if info:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Layers", info.get("n_layer"))
        c2.metric("Params", f"{info.get('params', 0):,}")
        c3.metric("Iter", info.get("iter_num", 0))
        c4.metric("Best val", f"{info.get('best_val_loss', 0):.4f}")

    c1, c2, c3 = st.columns(3)
    temperature = c1.slider("temperature", 0.1, 2.0, 0.8)
    top_k = c2.slider("top_k", 1, 100, 40)
    max_tokens = c3.slider("max_tokens", 20, 500, 100)

    for m in st.session_state.msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input(t("chat.input_placeholder", lang)):
        st.session_state.msgs.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            ids = [stoi.get(c, 0) for c in prompt]
            x = torch.tensor(ids, dtype=torch.long)[None, ...]
            y = model.generate(x, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
            resp = "".join(itos[i] for i in y[0].tolist())
            st.markdown(resp)
        st.session_state.msgs.append({"role": "assistant", "content": resp})

    if st.button(f"🗑️ {t('chat.clear', lang)}"):
        st.session_state.msgs = []
        st.rerun()
