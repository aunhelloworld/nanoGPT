"""Cached model loader for the chat tab."""

import os
import pickle

import streamlit as st
import torch

from config import AI_NAME, OUT_DIR
from lib import diagnostics
from model import GPT, GPTConfig


@st.cache_resource
def load_model():
    ckpt = os.path.join(OUT_DIR, "ckpt.pt")
    if not os.path.exists(ckpt):
        return None, None, None, None

    ck = torch.load(ckpt, map_location="cpu")
    cfg = GPTConfig(**ck["model_args"])
    model = GPT(cfg)
    sd = ck["model"]
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[10:]] = sd.pop(k)
    model.load_state_dict(sd)

    meta_path = os.path.join("data", AI_NAME, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        stoi, itos = meta["stoi"], meta["itos"]
    else:
        stoi, itos = {}, {}

    model.eval()
    info = diagnostics.get_model_info(OUT_DIR)
    return model, stoi, itos, info
