"""Selective cleanup and full reset."""

import os
import shutil

from config import AI_NAME, OUT_DIR
from lib.registry import list_txt_files, reset_registry
from lib.training import DATASET_DIR, kill_train_process, is_training

META_PATH = os.path.join("data", AI_NAME, "meta.pkl")
CKPT_PATH = os.path.join(OUT_DIR, "ckpt.pt")


def has_checkpoints() -> bool:
    return os.path.exists(CKPT_PATH) or os.path.isdir(OUT_DIR)


def has_import_data() -> bool:
    return bool(list_txt_files()) or os.path.isdir(DATASET_DIR)


def has_history() -> bool:
    return os.path.exists(os.path.join("data", "registry.json"))


def _require_idle():
    if is_training():
        raise RuntimeError("error.training_in_progress")


def delete_checkpoints() -> None:
    _require_idle()
    kill_train_process()
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)


def delete_import_data() -> None:
    _require_idle()
    if os.path.isdir(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    if os.path.isdir("data"):
        for name in os.listdir("data"):
            if name.endswith(".txt") or name.startswith("_tmp"):
                path = os.path.join("data", name)
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
        for temp in ("temp_extract", "temp_convert"):
            path = os.path.join("data", temp)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)


def delete_history() -> None:
    _require_idle()
    reset_registry()


def delete_everything() -> None:
    kill_train_process()
    delete_checkpoints()
    delete_import_data()
    delete_history()
