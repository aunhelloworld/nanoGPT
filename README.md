# NanoGPT Training Dashboard

A **Streamlit UI** on top of [nanoGPT](https://github.com/karpathy/nanoGPT) — train a character-level GPT on your own text without touching the command line.

> **Based on [nanoGPT](https://github.com/karpathy/nanoGPT)** by [Andrej Karpathy](https://github.com/karpathy) (MIT License).  
> This fork adds a web dashboard, smart file import, training history, and i18n support.

---

## Why this fork?

The original [nanoGPT](https://github.com/karpathy/nanoGPT) is a minimal, powerful training codebase — but it expects you to use the terminal, prepare datasets manually, and understand hyperparameters.

**This project makes it accessible:**

1. **Import** — Drop in HTML, PDF, EPUB, DOCX, ZIP, 7z, or plain text
2. **Train** — Pick a preset, click Start, watch loss live
3. **Chat** — Talk to your model when training finishes

No sample datasets are bundled. You bring your own text.

---

## Quick start

```bash
git clone https://github.com/aunhelloworld/nanoGPT.git
cd nanoGPT
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Workflow

```
Import files  →  Manage data  →  Train  →  Chat
     ↑                              ↓
     └──────── Analysis (live loss) ┘
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Smart import** | Converts HTML, PDF, EPUB, DOCX, XML, archives (ZIP/7z/tar/gz) to clean `.txt` |
| **Training dashboard** | Presets (fast / normal / quality), full hyperparameter control |
| **Non-blocking training** | UI stays responsive; live loss chart + log terminal |
| **Resume training** | Continue from checkpoint or start fresh |
| **Registry** | Tracks imported files, train counts, run history |
| **Chat** | Temperature, top-k, adjustable generation length |
| **Tools** | System diagnostics, test conversion, sample generation |
| **i18n** | English (default) + Thai — add languages via JSON files |

---

## Project structure

```
app.py              # Streamlit entry point
config.py           # Shared constants (AI_NAME, paths)
prepare_char.py     # Character-level dataset preparation (CLI)

language/           # UI translations (JSON)
  en.json           # English (source of truth)
  th.json           # Thai
  README.md         # How to add a language

ui/                 # Streamlit UI modules
  sidebar.py
  tabs/             # One file per tab

lib/                # Backend logic (no UI)
  converters.py     # File → text pipeline
  training.py       # Start/stop training, read logs
  registry.py       # File & run history (data/registry.json)
  diagnostics.py    # System checks, sample generation

data/               # User text files (empty in git — .gitkeep only)
out-my_ai/          # Checkpoints & logs (gitignored)

train.py            # nanoGPT training loop (upstream)
model.py            # GPT model definition (upstream)
sample.py           # CLI sampling (upstream)
```

---

## Supported file formats

| Format | Notes |
|--------|-------|
| `.txt`, `.md` | Plain text |
| `.html`, `.htm` | Strips scripts, styles, nav |
| `.xml` | Wikipedia dumps (streaming) |
| `.pdf` | pymupdf extraction |
| `.epub` | All chapters |
| `.docx` | Word paragraphs |
| `.zip`, `.7z`, `.tar.gz`, `.gz`, `.bz2`, `.xz` | Nested archives supported |

---

## CLI (still available)

The original nanoGPT CLI works alongside the UI:

```bash
# Prepare merged data manually
python prepare_char.py my_ai

# Train from terminal
python train.py --dataset=my_ai --out_dir=out-my_ai --device=cpu --compile=False --max_iters=5000

# Resume
python train.py --dataset=my_ai --out_dir=out-my_ai --init_from=resume --device=cpu --compile=False

# Sample
python sample.py --out_dir=out-my_ai --device=cpu
```

See [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md) for the full original nanoGPT documentation.

---

## Translations

UI language is selected in the sidebar. To add a language:

1. Copy `language/en.json` → `language/<code>.json`
2. Translate all values (keep keys unchanged)
3. See [language/README.md](language/README.md)

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Git remotes (for contributors)

```bash
origin    → your fork
upstream  → https://github.com/karpathy/nanoGPT
```

Pull upstream updates:

```bash
git fetch upstream
git merge upstream/master
```

---

## What is NOT committed

These are generated at runtime and listed in `.gitignore`:

- `data/*.txt` — your imported text
- `data/my_ai/` — prepared binaries
- `data/registry.json` — import/train history
- `out-my_ai/` — checkpoints, loss logs

---

## License

MIT License — see [LICENSE](LICENSE).

Original nanoGPT copyright © 2022 Andrej Karpathy.  
Dashboard extensions © contributors.

---

## Credits

- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- [minGPT](https://github.com/karpathy/minGPT) (predecessor)
- Built with [Streamlit](https://streamlit.io/), [PyTorch](https://pytorch.org/)
