"""Auto-tune training hyperparameters from dataset size and hardware."""

from __future__ import annotations

import torch

from lib import registry
from lib.training import PRESETS, default_device


def get_vram_mb() -> int | None:
    if not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(0)
        return props.total_memory // (1024 * 1024)
    except Exception:
        return None


def collect_dataset_stats() -> dict | None:
    files = registry.list_txt_files()
    if not files:
        return None

    total_chars = 0
    unique_chars: set[str] = set()
    for fname in files:
        stats = registry.get_file_stats(fname)
        if not stats:
            continue
        total_chars += stats["chars"]
        with open(f"data/{fname}", encoding="utf-8", errors="ignore") as f:
            unique_chars.update(f.read())

    if total_chars == 0:
        return None

    return {
        "file_count": len(files),
        "total_chars": total_chars,
        "unique_chars": len(unique_chars),
        "train_chars_est": int(total_chars * 0.9),
    }


def _round_iters(n: int) -> int:
    return max(100, int(round(n / 100) * 100))


def _ensure_head_divisible(n_embd: int, n_head: int) -> int:
    while n_head > 1 and n_embd % n_head != 0:
        n_head -= 1
    return n_head


def _base_from_data(total_chars: int) -> tuple[dict, str]:
    if total_chars < 20_000:
        base = dict(PRESETS["fast"])
        base["max_iters"] = _round_iters(max(300, min(800, total_chars // 40)))
        tier = "tiny"
    elif total_chars < 200_000:
        base = dict(PRESETS["normal"])
        base["max_iters"] = _round_iters(max(1000, min(5000, total_chars // 200)))
        tier = "small"
    elif total_chars < 2_000_000:
        base = dict(PRESETS["normal"])
        tier = "medium"
    elif total_chars < 20_000_000:
        base = dict(PRESETS["quality"])
        tier = "large"
    else:
        base = dict(PRESETS["quality"])
        base["max_iters"] = 30_000
        tier = "xlarge"
    return base, tier


def _apply_data_tuning(base: dict, stats: dict, tier: str) -> list[dict]:
    adjustments: list[dict] = []
    total = stats["total_chars"]
    vocab = stats["unique_chars"]

    if tier == "tiny":
        base["n_layer"] = 2
        base["n_head"] = 4
        base["n_embd"] = 128
        base["block_size"] = 32
        adjustments.append({
            "param": "model size",
            "value": "2 layers, 128 embd",
            "reason_key": "auto.reason.tiny_model",
        })
    elif tier == "small" and vocab < 80:
        base["n_layer"] = 2
        base["n_embd"] = 256
        base["n_head"] = 4
        adjustments.append({
            "param": "model size",
            "value": "2 layers, 256 embd",
            "reason_key": "auto.reason.small_vocab",
            "kwargs": {"vocab": vocab},
        })

    if total >= 500_000 and tier in ("medium", "large", "xlarge"):
        if base.get("block_size", 32) < 64:
            base["block_size"] = 64
            adjustments.append({
                "param": "block_size",
                "value": 64,
                "reason_key": "auto.reason.block_large_data",
                "kwargs": {"chars": f"{total:,}"},
            })

    base["n_head"] = _ensure_head_divisible(base["n_embd"], base["n_head"])

    iters = base["max_iters"]
    adjustments.append({
        "param": "max_iters",
        "value": iters,
        "reason_key": "auto.reason.max_iters",
        "kwargs": {"chars": f"{total:,}", "iters": iters},
    })
    return adjustments


def _apply_hardware_tuning(
    base: dict,
    *,
    device: str,
    vram_mb: int | None,
) -> tuple[int, float, bool, list[dict], list[dict]]:
    adjustments: list[dict] = []
    warnings: list[dict] = []
    batch_size = 8
    dropout = 0.2
    compile = device != "cpu"

    if device == "cpu":
        batch_size = 4
        compile = False
        adjustments.append({
            "param": "batch_size",
            "value": 4,
            "reason_key": "auto.reason.batch_cpu",
        })
        adjustments.append({
            "param": "compile",
            "value": False,
            "reason_key": "auto.reason.compile_off_cpu",
        })
        warnings.append({"key": "auto.warn.cpu_slow"})
    elif vram_mb is not None:
        if vram_mb < 4096:
            batch_size = 4
            if base.get("block_size", 32) > 32:
                base["block_size"] = 32
            if base.get("n_embd", 384) > 256:
                base["n_embd"] = 256
                base["n_head"] = _ensure_head_divisible(base["n_embd"], base["n_head"])
            adjustments.append({
                "param": "batch_size / model",
                "value": f"batch=4, vram={vram_mb}MB",
                "reason_key": "auto.reason.low_vram",
                "kwargs": {"vram": vram_mb},
            })
        elif vram_mb < 8192:
            batch_size = 8
            adjustments.append({
                "param": "batch_size",
                "value": 8,
                "reason_key": "auto.reason.vram_mid",
                "kwargs": {"vram": vram_mb},
            })
        else:
            batch_size = 16
            adjustments.append({
                "param": "batch_size",
                "value": 16,
                "reason_key": "auto.reason.vram_high",
                "kwargs": {"vram": vram_mb},
            })
    else:
        adjustments.append({
            "param": "device",
            "value": device,
            "reason_key": "auto.reason.device_cuda",
        })

    if base.get("n_embd", 384) >= 384:
        learning_rate = 3e-4
        adjustments.append({
            "param": "learning_rate",
            "value": learning_rate,
            "reason_key": "auto.reason.lr_large_model",
        })
    else:
        learning_rate = 1e-3
        adjustments.append({
            "param": "learning_rate",
            "value": learning_rate,
            "reason_key": "auto.reason.lr_small_model",
        })

    tier = base.get("_tier", "")
    if tier in ("tiny", "small"):
        dropout = 0.3
        adjustments.append({
            "param": "dropout",
            "value": dropout,
            "reason_key": "auto.reason.dropout_small_data",
        })

    return batch_size, learning_rate, dropout, compile, adjustments, warnings


def compute_auto_settings(
    *,
    preset_key: str = "auto",
    init_from: str = "scratch",
) -> dict | None:
    stats = collect_dataset_stats()
    if stats is None:
        return None

    device = default_device()
    vram_mb = get_vram_mb()

    if preset_key == "auto":
        base, tier = _base_from_data(stats["total_chars"])
    elif preset_key in PRESETS:
        base = dict(PRESETS[preset_key])
        tier = preset_key
    else:
        return None

    base["_tier"] = tier
    adjustments: list[dict] = []
    warnings: list[dict] = []

    if preset_key == "auto":
        adjustments.extend(_apply_data_tuning(base, stats, tier))
    else:
        adjustments.append({
            "param": "preset",
            "value": preset_key,
            "reason_key": "auto.reason.preset_manual",
        })

    batch_size, learning_rate, dropout, compile, hw_adj, hw_warn = _apply_hardware_tuning(
        base, device=device, vram_mb=vram_mb,
    )
    adjustments.extend(hw_adj)
    warnings.extend(hw_warn)

    if stats["total_chars"] < 10_000:
        warnings.append({"key": "auto.warn.tiny_data", "kwargs": {"chars": f"{stats['total_chars']:,}"}})
    if stats["unique_chars"] > 512:
        warnings.append({"key": "auto.warn.large_vocab", "kwargs": {"vocab": stats["unique_chars"]}})

    base.pop("_tier", None)

    config = {
        "max_iters": base["max_iters"],
        "n_layer": base["n_layer"],
        "n_head": base["n_head"],
        "n_embd": base["n_embd"],
        "block_size": base["block_size"],
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "init_from": init_from,
        "device": device,
        "compile": compile,
        "min_lr": learning_rate / 10,
        "eval_iters": 25 if device == "cpu" else 200,
    }

    return {
        "config": config,
        "stats": stats,
        "device": device,
        "vram_mb": vram_mb,
        "tier": tier,
        "adjustments": adjustments,
        "warnings": warnings,
    }
