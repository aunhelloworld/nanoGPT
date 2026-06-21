"""System diagnostics and dataset validation."""
import os
import shutil
import subprocess
import psutil


def check_import(name, import_stmt=None):
    try:
        if import_stmt:
            exec(import_stmt, {})
        else:
            __import__(name)
        return True, "OK"
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def run_diagnostics():
    results = []

    def add(category, name, ok, detail=""):
        results.append({"category": category, "name": name, "ok": ok, "detail": detail})

    try:
        import torch
        add("PyTorch", "torch", True, torch.__version__)
        cuda_ok = torch.cuda.is_available()
        add("PyTorch", "CUDA", cuda_ok, torch.cuda.get_device_name(0) if cuda_ok else "No GPU found")
    except ImportError as e:
        add("PyTorch", "torch", False, str(e))

    deps = [
        ("streamlit", None),
        ("numpy", None),
        ("psutil", None),
        ("requests", None),
        ("bs4", "from bs4 import BeautifulSoup"),
        ("lxml", None),
        ("charset_normalizer", None),
        ("py7zr", None),
        ("fitz", "import fitz"),
        ("ebooklib", None),
        ("docx", "from docx import Document"),
    ]
    for name, stmt in deps:
        ok, detail = check_import(name, stmt)
        add("Dependencies", name, ok, detail if not ok else "OK")

    add("System", "7z CLI", shutil.which("7z") is not None,
        shutil.which("7z") or "Not found (py7zr works without 7z CLI)")

    disk = shutil.disk_usage(".")
    add("System", "Disk free", disk.free > 500 * 1024 * 1024,
        f"{disk.free / (1024**3):.1f} GB free")

    mem = psutil.virtual_memory()
    add("System", "RAM available", mem.available > 512 * 1024 * 1024,
        f"{mem.available / (1024**3):.1f} GB free")

    return results


def validate_dataset(dataset_name=None):
    from config import AI_NAME
    if dataset_name is None:
        dataset_name = AI_NAME
    data_dir = os.path.join("data", dataset_name)
    input_file = os.path.join(data_dir, "input.txt")
    if not os.path.exists(input_file):
        return False, f"File not found: {input_file}"

    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()

    chars = sorted(set(data))
    return True, {
        "chars_total": len(data),
        "lines": data.count("\n") + 1,
        "vocab_size": len(chars),
        "sample_chars": "".join(chars[:50]),
    }


def dry_run_prepare(dataset_name=None):
    return validate_dataset(dataset_name)


def generate_sample(out_dir=None, prompt="", max_new_tokens=100, temperature=0.8, top_k=40, device=None):
    from config import AI_NAME, OUT_DIR as DEFAULT_OUT
    if out_dir is None:
        out_dir = DEFAULT_OUT
    import torch
    import pickle
    from model import GPTConfig, GPT

    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        return None, "Checkpoint not found"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(ckpt_path, map_location=device)
    cfg = GPTConfig(**ck["model_args"])
    model = GPT(cfg)
    sd = ck["model"]
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[10:]] = sd.pop(k)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    meta_path = os.path.join("data", AI_NAME, "meta.pkl")
    if os.path.exists(meta_path):
        meta = pickle.load(open(meta_path, "rb"))
        stoi, itos = meta["stoi"], meta["itos"]
    else:
        return None, "meta.pkl not found"

    if not prompt:
        prompt = "The "

    ids = [stoi.get(c, 0) for c in prompt]
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    text = "".join(itos[i] for i in y[0].tolist())
    return text, None


def clean_temp_files():
    removed = []
    if not os.path.exists("data"):
        return removed
    for name in os.listdir("data"):
        if name.startswith("_tmp") or name in ("temp_extract", "temp_convert"):
            path = os.path.join("data", name)
            if os.path.isfile(path):
                os.remove(path)
                removed.append(name)
            elif os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
                removed.append(name + "/")
    return removed


def get_model_info(out_dir=None):
    from config import OUT_DIR as DEFAULT_OUT
    if out_dir is None:
        out_dir = DEFAULT_OUT
    import torch
    from model import GPTConfig, GPT

    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        return None

    ck = torch.load(ckpt_path, map_location="cpu")
    args = ck["model_args"]
    cfg = GPTConfig(**args)
    model = GPT(cfg)
    sd = ck["model"]
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[10:]] = sd.pop(k)
    model.load_state_dict(sd)

    return {
        "n_layer": args.get("n_layer"),
        "n_head": args.get("n_head"),
        "n_embd": args.get("n_embd"),
        "block_size": args.get("block_size"),
        "vocab_size": args.get("vocab_size"),
        "params": model.get_num_params(),
        "iter_num": ck.get("iter_num", 0),
        "best_val_loss": ck.get("best_val_loss"),
    }
