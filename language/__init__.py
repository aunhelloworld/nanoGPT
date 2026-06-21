"""Internationalization (i18n) — load JSON locale files from this folder."""
import json
import os

DEFAULT_LANG = "en"
_LANG_DIR = os.path.dirname(os.path.abspath(__file__))
_cache: dict[str, dict] = {}


def available_languages() -> list[str]:
    return sorted(
        fname[:-5]
        for fname in os.listdir(_LANG_DIR)
        if fname.endswith(".json")
    )


def _load(lang: str) -> dict:
    if lang not in _cache:
        path = os.path.join(_LANG_DIR, f"{lang}.json")
        if not os.path.exists(path):
            lang = DEFAULT_LANG
            path = os.path.join(_LANG_DIR, f"{lang}.json")
        with open(path, encoding="utf-8") as f:
            _cache[lang] = json.load(f)
    return _cache[lang]


def t(key: str, lang: str = DEFAULT_LANG, **kwargs) -> str:
    """Translate a key. Falls back to English, then the key itself."""
    text = _load(lang).get(key)
    if text is None and lang != DEFAULT_LANG:
        text = _load(DEFAULT_LANG).get(key)
    if text is None:
        text = key
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, ValueError):
            return text
    return text


def language_label(code: str, lang: str = DEFAULT_LANG) -> str:
    return t(f"lang.{code}", lang=lang)
