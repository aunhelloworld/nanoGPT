"""Training orchestration: merge data, prepare, start/stop, read logs."""
import os
import signal
import subprocess
import sys
import psutil
import torch

from config import AI_NAME

OUT_DIR = f"out-{AI_NAME}"
DATASET_DIR = f"data/{AI_NAME}"
PID_FILE = os.path.join("data", "training.pid")
LOSS_FILE = os.path.join(OUT_DIR, "loss.txt")
LOG_FILE = os.path.join(OUT_DIR, "train.log")
CKPT_FILE = os.path.join(OUT_DIR, "ckpt.pt")

PRESETS = {
    "fast": {"max_iters": 500, "n_layer": 2, "n_head": 4, "n_embd": 256, "block_size": 32},
    "normal": {"max_iters": 5000, "n_layer": 3, "n_head": 6, "n_embd": 384, "block_size": 32},
    "quality": {"max_iters": 20000, "n_layer": 4, "n_head": 6, "n_embd": 384, "block_size": 64},
}


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def has_checkpoint():
    return os.path.exists(CKPT_FILE)


def kill_train_process():
    killed = False
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if cmdline and "train.py" in " ".join(cmdline):
                os.kill(proc.info["pid"], signal.SIGTERM)
                killed = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            pass
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
    return killed


def is_training():
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE) as f:
                pid = int(f.read().strip())
            return psutil.pid_exists(pid)
        except (ValueError, OSError):
            pass
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if cmdline and "train.py" in " ".join(cmdline):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False


def list_data_files():
    if not os.path.exists("data"):
        return []
    return sorted(
        f for f in os.listdir("data")
        if f.endswith(".txt") and not f.startswith("_tmp")
    )


def merge_all_txt():
    os.makedirs(DATASET_DIR, exist_ok=True)
    files = list_data_files()
    total = 0
    with open(f"{DATASET_DIR}/input.txt", "w", encoding="utf-8") as out:
        for i, fname in enumerate(files):
            fpath = os.path.join("data", fname)
            with open(fpath, "r", encoding="utf-8", errors="ignore") as inf:
                content = inf.read()
                out.write(content)
                total += len(content)
                if i < len(files) - 1:
                    out.write("\n\n")
    return files, total


def prepare_dataset():
    result = subprocess.run(
        [sys.executable, "prepare_char.py", AI_NAME],
        capture_output=True, text=True,
    )
    return result.returncode == 0, result.stdout + result.stderr


def build_train_command(config):
    max_iters = config["max_iters"]
    warmup = max(1, min(100, max_iters - 1))
    device = config.get("device", default_device())
    compile_flag = config.get("compile", device != "cpu")
    eval_iters = config.get("eval_iters")
    if eval_iters is None:
        eval_iters = 25 if device == "cpu" else 200

    cmd = [
        sys.executable, "train.py",
        f"--dataset={AI_NAME}",
        f"--out_dir={OUT_DIR}",
        f"--init_from={config.get('init_from', 'scratch')}",
        f"--batch_size={config.get('batch_size', 8)}",
        f"--block_size={config.get('block_size', 32)}",
        f"--n_layer={config.get('n_layer', 3)}",
        f"--n_head={config.get('n_head', 6)}",
        f"--n_embd={config.get('n_embd', 384)}",
        f"--dropout={config.get('dropout', 0.2)}",
        f"--learning_rate={config.get('learning_rate', 1e-3)}",
        f"--max_iters={max_iters}",
        f"--lr_decay_iters={max_iters}",
        f"--min_lr={config.get('min_lr', 1e-4)}",
        f"--warmup_iters={warmup}",
        "--eval_interval=250",
        f"--eval_iters={eval_iters}",
        "--log_interval=10",
        "--beta2=0.99",
        "--always_save_checkpoint=False",
        "--wandb_log=False",
        f"--device={device}",
        f"--compile={compile_flag}",
        "--gradient_accumulation_steps=1",
    ]
    return cmd


def start_training(config, *, fresh_loss_log=False):
    if is_training():
        return None, "error.training_in_progress"

    files, total = merge_all_txt()
    if not files:
        return None, "error.no_txt_files"

    ok, prep_out = prepare_dataset()
    if not ok:
        return None, ("error.prepare_failed", prep_out)

    os.makedirs(OUT_DIR, exist_ok=True)
    if fresh_loss_log and os.path.exists(LOSS_FILE):
        os.remove(LOSS_FILE)

    log_fh = open(LOG_FILE, "a", encoding="utf-8")
    cmd = build_train_command(config)
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        text=True,
    )
    with open(PID_FILE, "w") as f:
        f.write(str(proc.pid))

    return proc, {"files": files, "total_chars": total, "pid": proc.pid}


def read_loss_data():
    if not os.path.exists(LOSS_FILE):
        return [], []

    train_loss, val_loss = [], []
    with open(LOSS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "_val," in line:
                parts = line.split("_val,")
                try:
                    val_loss.append({"iter": int(parts[0]), "loss": float(parts[1])})
                except ValueError:
                    pass
            elif "," in line:
                parts = line.split(",", 1)
                try:
                    train_loss.append({"iter": int(parts[0]), "loss": float(parts[1])})
                except ValueError:
                    pass
    return train_loss, val_loss


def read_log_tail(n_lines=80):
    if not os.path.exists(LOG_FILE):
        return ""
    with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    return "".join(lines[-n_lines:])


def get_best_val_loss():
    _, val = read_loss_data()
    if not val:
        return None
    return min(v["loss"] for v in val)


def get_latest_train_iter():
    train, _ = read_loss_data()
    if not train:
        return 0
    return train[-1]["iter"]


def get_training_target_iters() -> int | None:
    """Max iterations for the active training run."""
    import streamlit as st
    if st.session_state.get("train_max_iters"):
        return st.session_state.train_max_iters
    run_id = st.session_state.get("train_run_id")
    if run_id:
        from lib.registry import get_run
        run = get_run(run_id)
        if run:
            return run.get("iters")
    from lib.registry import get_active_run
    active = get_active_run()
    if active:
        return active.get("iters")
    return None


def format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:
        return "—"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def get_progress_info() -> dict:
    """Progress fraction and ETA while training."""
    import time
    import streamlit as st

    max_iters = get_training_target_iters() or 0
    latest = get_latest_train_iter()
    fraction = min(latest / max_iters, 1.0) if max_iters > 0 else 0.0

    eta_seconds = None
    start = st.session_state.get("train_start_time")
    if start and latest > 0 and max_iters > latest:
        elapsed = time.time() - start
        eta_seconds = (max_iters - latest) * (elapsed / latest)

    return {
        "latest": latest,
        "max_iters": max_iters,
        "fraction": fraction,
        "eta": format_eta(eta_seconds),
        "percent": int(fraction * 100),
    }


def check_training_finished(proc):
    if proc is None:
        return True, None
    ret = proc.poll()
    if ret is None:
        return False, None
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
    return True, ret


def reset_all():
    from lib.cleanup import delete_everything
    delete_everything()
