"""Global UI styles."""

import streamlit as st


def inject_styles():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 0.75rem; padding-bottom: 1.5rem; max-width: 1080px; }
        h1 { font-size: 1.6rem !important; margin-bottom: 0.25rem !important; }
        [data-testid="stMetric"] {
            background: rgba(128,128,128,0.07);
            padding: 0.35rem 0.55rem;
            border-radius: 0.35rem;
        }
        [data-testid="stMetricValue"] { font-size: 1rem; }
        [data-testid="stMetricLabel"] { font-size: 0.72rem; }
        [data-testid="stSidebar"] .block-container { padding-top: 1rem; }
        .live-bar {
            background: linear-gradient(90deg, rgba(255,160,0,0.15), rgba(255,100,0,0.08));
            border: 1px solid rgba(255, 160, 0, 0.4);
            border-radius: 0.45rem;
            padding: 0.45rem 0.85rem;
            margin: 0.25rem 0 0.65rem 0;
            font-size: 0.88rem;
            animation: pulse-bar 2s ease-in-out infinite;
        }
        @keyframes pulse-bar {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.85; }
        }
        .chat-scroll [data-testid="stVerticalBlockBorderWrapper"] {
            overflow-y: auto !important;
        }
        [data-testid="stChatMessage"] { font-size: 0.95rem; }
        [data-testid="stChatInput"] { position: sticky; bottom: 0; z-index: 100; }
        div[data-testid="stPills"] button { font-size: 0.82rem; padding: 0.2rem 0.55rem; }
        .stCode pre { font-size: 0.78rem; max-height: 280px; overflow-y: auto; }
        </style>
        """,
        unsafe_allow_html=True,
    )
