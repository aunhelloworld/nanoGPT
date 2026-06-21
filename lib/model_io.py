"""Export and import trained model checkpoints."""

import io
import os
import zipfile

from config import AI_NAME, OUT_DIR

CKPT_PATH = os.path.join(OUT_DIR, "ckpt.pt")
META_PATH = os.path.join("data", AI_NAME, "meta.pkl")
ZIP_NAME = f"nanogpt-{AI_NAME}-model.zip"


def can_export() -> bool:
    return os.path.isfile(CKPT_PATH)


def export_model_zip() -> bytes | None:
    if not can_export():
        return None
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(CKPT_PATH, "ckpt.pt")
        if os.path.isfile(META_PATH):
            zf.write(META_PATH, "meta.pkl")
    return buf.getvalue()


def import_model_zip(raw: bytes) -> tuple[bool, str]:
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names = set(zf.namelist())
            if "ckpt.pt" not in names:
                return False, "error.import_no_ckpt"
            os.makedirs(OUT_DIR, exist_ok=True)
            zf.extract("ckpt.pt", OUT_DIR)
            if "meta.pkl" in names:
                os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
                with zf.open("meta.pkl") as src, open(META_PATH, "wb") as dst:
                    dst.write(src.read())
        return True, ZIP_NAME
    except zipfile.BadZipFile:
        return False, "error.import_bad_zip"
    except Exception as exc:
        return False, str(exc)
