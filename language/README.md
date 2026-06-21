# Translations

UI strings live in JSON files in this folder. **English (`en.json`) is the source of truth.**

## Add a new language

1. Copy `en.json` to `<code>.json` (e.g. `ja.json`, `de.json`)
2. Translate every value — keep keys unchanged
3. The new language appears automatically in the UI language selector

## Key naming convention

Flat dot-separated keys grouped by area:

| Prefix | Section |
|--------|---------|
| `app.*` | App title |
| `sidebar.*` | Sidebar controls |
| `tabs.*` | Tab labels |
| `guide.*` | Guide tab |
| `import.*` | Import tab |
| `data.*` | Data tab |
| `train.*` | Train tab |
| `analysis.*` | Analysis tab |
| `chat.*` | Chat tab |
| `tools.*` | Tools tab |
| `preset.*` | Training presets |
| `error.*` | Error messages |

## Usage in code

```python
from language import t

label = t("tabs.guide", lang=st.session_state.lang)
```

## Format placeholders

Some strings use Python `str.format` placeholders:

```json
"train.started": "Training started (PID {pid}) — open the Analysis tab"
```

Do not rename `{pid}`, `{count}`, etc. when translating.
