"""Registry for imported files and training run history."""
import json
import os
from datetime import datetime, timezone

REGISTRY_PATH = os.path.join("data", "registry.json")


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _empty_registry():
    return {"files": {}, "runs": []}


def load_registry():
    if not os.path.exists(REGISTRY_PATH):
        return _empty_registry()
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry):
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def register_file(filename, *, source, original_name, fmt, chars):
    registry = load_registry()
    entry = registry["files"].get(filename, {})
    entry.update({
        "source": source,
        "original_name": original_name,
        "format": fmt,
        "chars": chars,
        "imported_at": _now_iso(),
        "train_count": entry.get("train_count", 0),
        "last_trained_at": entry.get("last_trained_at"),
    })
    registry["files"][filename] = entry
    save_registry(registry)
    return entry


def remove_file(filename):
    registry = load_registry()
    registry["files"].pop(filename, None)
    save_registry(registry)


def increment_train_count(filenames):
    registry = load_registry()
    now = _now_iso()
    for fn in filenames:
        if fn in registry["files"]:
            registry["files"][fn]["train_count"] = registry["files"][fn].get("train_count", 0) + 1
            registry["files"][fn]["last_trained_at"] = now
    save_registry(registry)


def start_run(*, iters, init_from, files_used, config, preset=None, auto_meta=None):
    registry = load_registry()
    run_id = len(registry["runs"]) + 1
    run = {
        "id": run_id,
        "started_at": _now_iso(),
        "finished_at": None,
        "iters": iters,
        "init_from": init_from,
        "files_used": files_used,
        "final_val_loss": None,
        "config": config,
        "preset": preset,
        "auto_meta": auto_meta,
        "status": "running",
    }
    registry["runs"].append(run)
    save_registry(registry)
    return run_id


def get_run(run_id):
    for run in load_registry().get("runs", []):
        if run["id"] == run_id:
            return run
    return None


def finish_run(run_id, *, final_val_loss=None, status="completed"):
    registry = load_registry()
    for run in registry["runs"]:
        if run["id"] == run_id:
            run["finished_at"] = _now_iso()
            run["final_val_loss"] = final_val_loss
            run["status"] = status
            break
    save_registry(registry)


def get_active_run():
    registry = load_registry()
    for run in reversed(registry["runs"]):
        if run.get("status") == "running":
            return run
    return None


def list_txt_files():
    if not os.path.exists("data"):
        return []
    return sorted(
        f for f in os.listdir("data")
        if f.endswith(".txt") and not f.startswith("_tmp")
    )


def get_file_stats(filename):
    path = os.path.join("data", filename)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return {
        "bytes": os.path.getsize(path),
        "chars": len(content),
        "lines": content.count("\n") + 1,
        "unique_chars": len(set(content)),
    }


def export_registry_json():
    return json.dumps(load_registry(), ensure_ascii=False, indent=2)


def import_registry_json(raw):
    data = json.loads(raw)
    if "files" not in data or "runs" not in data:
        raise ValueError("error.invalid_registry")
    save_registry(data)


def reset_registry():
    if os.path.exists(REGISTRY_PATH):
        os.remove(REGISTRY_PATH)
