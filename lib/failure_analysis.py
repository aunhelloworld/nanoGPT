"""Analyze training logs and suggest concrete config fixes."""

from __future__ import annotations

import os
import re

from config import AI_NAME, OUT_DIR

CKPT_PATH = os.path.join(OUT_DIR, "ckpt.pt")
META_PATH = os.path.join("data", AI_NAME, "meta.pkl")


def _match(text: str, *needles: str) -> bool:
    lower = text.lower()
    return any(n.lower() in lower for n in needles)


def _extract_exception(log: str) -> str:
    if not log:
        return ""
    lines = [ln.strip() for ln in log.strip().splitlines() if ln.strip()]
    for line in reversed(lines):
        if re.match(r"^[A-Za-z_][\w.]*Error:", line):
            return line
        if re.match(r"^[A-Za-z_][\w.]*Exception:", line):
            return line
        if line.startswith("RuntimeError:") or line.startswith("torch.OutOfMemoryError:"):
            return line
    for line in reversed(lines):
        if ": " in line and not line.startswith("File "):
            return line
    return lines[-1] if lines else ""


def _parse_log_context(log: str) -> dict:
    ctx: dict = {}
    for key, pattern in [
        ("device", r"Overriding:\s*device\s*=\s*(\S+)"),
        ("init_from", r"Overriding:\s*init_from\s*=\s*(\S+)"),
        ("batch_size", r"Overriding:\s*batch_size\s*=\s*(\d+)"),
        ("block_size", r"Overriding:\s*block_size\s*=\s*(\d+)"),
        ("n_layer", r"Overriding:\s*n_layer\s*=\s*(\d+)"),
        ("n_embd", r"Overriding:\s*n_embd\s*=\s*(\d+)"),
        ("compile", r"Overriding:\s*compile\s*=\s*(\S+)"),
        ("eval_iters", r"Overriding:\s*eval_iters\s*=\s*(\d+)"),
        ("vocab_size", r"found vocab_size\s*=\s*(\d+)"),
        ("params", r"number of parameters:\s*([\d.]+[MK]?)"),
        ("tokens_per_iter", r"tokens per iteration will be:\s*([\d,]+)"),
    ]:
        m = re.search(pattern, log, re.IGNORECASE)
        if m:
            val = m.group(1).replace(",", "")
            if val.isdigit():
                ctx[key] = int(val)
            elif val in ("True", "False"):
                ctx[key] = val == "True"
            else:
                ctx[key] = val

    if "Resuming training" in log:
        ctx["init_from"] = ctx.get("init_from") or "resume"
    if "Initializing a new model from scratch" in log:
        ctx["init_from"] = "scratch"
    return ctx


def _checkpoint_vocab() -> int | None:
    if not os.path.isfile(CKPT_PATH):
        return None
    try:
        import torch
        ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        return ck.get("model_args", {}).get("vocab_size")
    except Exception:
        return None


def _meta_vocab() -> int | None:
    if not os.path.isfile(META_PATH):
        return None
    try:
        import pickle
        with open(META_PATH, "rb") as f:
            return pickle.load(f).get("vocab_size")
    except Exception:
        return None


def _classify(log: str, exc: str, ctx: dict, config: dict) -> str:
    combined = f"{log}\n{exc}"
    ck_vocab = _checkpoint_vocab()
    meta_vocab = ctx.get("vocab_size") or _meta_vocab()

    if _match(combined, "killed", "sigkill", "signal 9"):
        return "killed"
    if _match(combined, "out of memory", "oom", "cuda out of memory", "allocat"):
        return "oom"
    if _match(combined, "size mismatch", "missing key", "unexpected key", "state_dict", "strict="):
        return "resume_mismatch"
    if _match(combined, "index out of range", "device-side assert", "embedding"):
        if ctx.get("init_from") == "resume" or config.get("init_from") == "resume":
            return "vocab_changed"
        return "data_index"
    if ck_vocab and meta_vocab and ck_vocab != meta_vocab:
        if ctx.get("init_from") == "resume" or config.get("init_from") == "resume":
            return "vocab_changed"
    if _match(combined, "torch.compile", "inductor", "triton", "backend compiler"):
        return "compile_error"
    if _match(combined, "prepare_char", "input.txt", "no such file", "filenotfounderror"):
        return "prepare_data"
    if _match(combined, "nan", "inf", "non-finite"):
        return "nan_loss"
    if "estimate_loss" in combined and _match(combined, "model(", "forward"):
        if ctx.get("device") == "cpu" or config.get("device") == "cpu":
            return "eval_heavy_cpu"
        return "eval_heavy"
    if _match(combined, "cuda", "gpu") and _match(combined, "not available", "no cuda"):
        return "no_cuda"
    if _match(combined, "encoding", "unicode", "decode"):
        return "encoding"
    return "unknown"


def _patch_config(config: dict, category: str, ctx: dict) -> dict:
    patch = dict(config)
    bs = int(patch.get("batch_size", ctx.get("batch_size", 8)))
    block = int(patch.get("block_size", ctx.get("block_size", 32)))
    n_embd = int(patch.get("n_embd", ctx.get("n_embd", 384)))
    n_layer = int(patch.get("n_layer", ctx.get("n_layer", 3)))
    device = patch.get("device", ctx.get("device", "cpu"))

    patch["compile"] = False

    if category in ("oom", "killed", "eval_heavy", "eval_heavy_cpu"):
        patch["batch_size"] = max(2, bs // 2)
        if block > 32:
            patch["block_size"] = 32
        if n_embd > 256:
            patch["n_embd"] = 256
            patch["n_head"] = 4
        if n_layer > 2:
            patch["n_layer"] = 2
        patch["eval_iters"] = 50 if category in ("eval_heavy_cpu", "eval_heavy") else min(100, int(patch.get("eval_iters", 200)))
        if device == "cpu" or patch.get("device") == "cpu":
            patch["eval_iters"] = 25
            patch["batch_size"] = min(patch["batch_size"], 4)

    if category == "compile_error":
        patch["compile"] = False

    if category in ("resume_mismatch", "vocab_changed"):
        patch["init_from"] = "scratch"

    if category == "no_cuda":
        patch["device"] = "cpu"
        patch["compile"] = False

    if category == "nan_loss":
        lr = float(patch.get("learning_rate", 1e-3))
        patch["learning_rate"] = lr / 2
        patch["min_lr"] = patch["learning_rate"] / 10
        patch["dropout"] = min(0.3, float(patch.get("dropout", 0.2)) + 0.05)

    if category == "eval_heavy_cpu":
        patch["eval_iters"] = 25
        patch["batch_size"] = min(int(patch.get("batch_size", 4)), 4)

    # Ensure n_head divides n_embd
    n_head = int(patch.get("n_head", 6))
    while n_head > 1 and patch["n_embd"] % n_head != 0:
        n_head -= 1
    patch["n_head"] = n_head

    return patch


def _recommendations(category: str, ctx: dict, config: dict, exc: str) -> list[dict]:
    recs: list[dict] = []
    ck_vocab = _checkpoint_vocab()
    meta_vocab = ctx.get("vocab_size") or _meta_vocab()

    if category == "oom":
        recs.append({"key": "fail.rec.reduce_batch", "kwargs": {
            "from": config.get("batch_size", ctx.get("batch_size", "?")),
            "to": max(2, int(config.get("batch_size", 8)) // 2),
        }})
        recs.append({"key": "fail.rec.reduce_eval", "kwargs": {"to": 50}})
        recs.append({"key": "fail.rec.disable_compile", "kwargs": {}})

    elif category == "killed":
        recs.append({"key": "fail.rec.killed", "kwargs": {}})
        recs.append({"key": "fail.rec.reduce_batch", "kwargs": {
            "from": config.get("batch_size", "?"),
            "to": max(2, int(config.get("batch_size", 8)) // 2),
        }})

    elif category == "resume_mismatch":
        recs.append({"key": "fail.rec.arch_mismatch", "kwargs": {}})
        recs.append({"key": "fail.rec.use_scratch", "kwargs": {}})

    elif category == "vocab_changed":
        recs.append({"key": "fail.rec.vocab_mismatch", "kwargs": {
            "ckpt": ck_vocab or "?",
            "data": meta_vocab or "?",
        }})
        recs.append({"key": "fail.rec.reprepare", "kwargs": {}})
        recs.append({"key": "fail.rec.use_scratch", "kwargs": {}})

    elif category in ("eval_heavy", "eval_heavy_cpu"):
        recs.append({"key": "fail.rec.eval_crash", "kwargs": {
            "device": ctx.get("device", config.get("device", "?")),
            "eval_iters": ctx.get("eval_iters", 200),
        }})
        recs.append({"key": "fail.rec.reduce_eval", "kwargs": {"to": 25 if category == "eval_heavy_cpu" else 50}})
        recs.append({"key": "fail.rec.reduce_batch", "kwargs": {
            "from": config.get("batch_size", ctx.get("batch_size", "?")),
            "to": min(4, int(config.get("batch_size", 8))),
        }})

    elif category == "compile_error":
        recs.append({"key": "fail.rec.disable_compile", "kwargs": {}})

    elif category == "prepare_data":
        recs.append({"key": "fail.rec.check_data", "kwargs": {}})

    elif category == "nan_loss":
        recs.append({"key": "fail.rec.lower_lr", "kwargs": {}})

    elif category == "no_cuda":
        recs.append({"key": "fail.rec.switch_cpu", "kwargs": {}})

    elif category == "unknown":
        recs.append({"key": "fail.rec.generic", "kwargs": {}})

    return recs


def _changes_list(before: dict, after: dict) -> list[dict]:
    changes = []
    for key in sorted(set(before) | set(after)):
        if key in ("min_lr",) and before.get(key) != after.get(key):
            pass
        b, a = before.get(key), after.get(key)
        if b != a and a is not None:
            changes.append({"param": key, "from": b, "to": a})
    return changes


def analyze_failure(log: str, config: dict | None = None) -> dict:
    config = config or {}
    exc = _extract_exception(log)
    ctx = _parse_log_context(log)
    category = _classify(log, exc, ctx, config)
    suggested = _patch_config(config, category, ctx)
    recommendations = _recommendations(category, ctx, config, exc)
    changes = _changes_list(config, suggested)

    # Legacy simple hint keys for backwards compatibility
    hints = []
    hint_map = {
        "oom": ["hint.oom", "hint.reduce_batch"],
        "killed": ["hint.oom", "hint.reduce_batch"],
        "resume_mismatch": ["hint.retrain_or_import"],
        "vocab_changed": ["hint.vocab_changed"],
        "eval_heavy_cpu": ["hint.reduce_batch", "hint.reduce_eval"],
        "eval_heavy": ["hint.reduce_batch", "hint.reduce_eval"],
        "compile_error": ["hint.disable_compile"],
        "prepare_data": ["hint.prepare_data"],
        "no_cuda": ["hint.use_cpu"],
        "encoding": ["hint.encoding"],
        "nan_loss": ["hint.lower_lr"],
    }
    hints.extend(hint_map.get(category, ["hint.generic"]))

    return {
        "category": category,
        "exception": exc,
        "context": ctx,
        "recommendations": recommendations,
        "suggested_config": suggested,
        "changes": changes,
        "hints": list(dict.fromkeys(hints)),
        "delete_checkpoint": category in ("resume_mismatch", "vocab_changed"),
    }


def hints_for_text(log: str, config: dict | None = None) -> list[str]:
    return analyze_failure(log, config)["hints"]


def hints_for_result(result, config: dict | None = None) -> list[str]:
    if isinstance(result, tuple):
        detail = result[1] if len(result) > 1 else ""
        return analyze_failure(f"{result[0]} {detail}", config)["hints"]
    if isinstance(result, str):
        return analyze_failure(result, config)["hints"]
    return analyze_failure(str(result), config)["hints"]
