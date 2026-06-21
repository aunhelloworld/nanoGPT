"""Chat scroll helper."""

import streamlit.components.v1 as components


def scroll_chat_to_bottom():
    components.html(
        """
        <script>
        (function () {
            const doc = window.parent.document;
            const boxes = doc.querySelectorAll('[data-testid="stVerticalBlockBorderWrapper"]');
            for (const box of boxes) {
                if (box.closest('.chat-scroll') || box.querySelector('[data-testid="stChatMessage"]')) {
                    box.scrollTop = box.scrollHeight;
                }
            }
        })();
        </script>
        """,
        height=0,
    )
