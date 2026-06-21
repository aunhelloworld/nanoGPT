# Contributing

Thank you for helping improve the NanoGPT Training Dashboard!

## Ways to contribute

### 1. Add a translation

The easiest contribution — no Python required.

1. Copy `language/en.json` to `language/<code>.json` (ISO 639-1 code, e.g. `ja`, `de`, `zh`)
2. Translate every **value**; do not change **keys**
3. Test: run `streamlit run app.py`, select your language in the sidebar
4. Open a pull request

See [language/README.md](language/README.md) for key naming conventions.

### 2. Improve the UI

UI code lives in `ui/` — one module per tab. Keep strings in `language/en.json`, not hardcoded in Python.

```python
from language import t
from ui.state import get_lang

st.subheader(t("data.title", get_lang()))
```

### 3. Extend file converters

Add new formats in `lib/converters.py`:

- Add extractor function (e.g. `extract_rtf()`)
- Register in `_extract_text_recursive()`
- Document in `language/en.json` guide section + README

### 4. Backend improvements

Logic belongs in `lib/` — keep `app.py` and `ui/` thin.

---

## Development setup

```bash
git clone https://github.com/aunhelloworld/nanoGPT.git
cd nanoGPT
pip install -r requirements.txt
streamlit run app.py
```

---

## Pull request guidelines

- **English** for code, comments, commit messages, and `language/en.json`
- One feature per PR when possible
- Do not commit user data (`data/*.txt`, `out-my_ai/`, checkpoints)
- Keep the MIT license and upstream credit in README

---

## Upstream sync

This project forks [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT). When syncing:

```bash
git fetch upstream
git merge upstream/master
```

Resolve conflicts carefully — preserve dashboard files (`app.py`, `ui/`, `lib/`, `language/`).
