"""Smart file conversion pipeline: any format -> clean .txt"""
import bz2
import gzip
import lzma
import os
import re
import shutil
import tarfile
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from lib.registry import register_file

SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".html", ".htm", ".xml",
    ".pdf", ".epub", ".docx",
    ".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".7z",
}

TEMP_DIR = os.path.join("data", "temp_convert")
DATA_DIR = "data"


def normalize_text(text):
    """Clean and normalize extracted text."""
    import unicodedata

    if isinstance(text, bytes):
        text = decode_bytes(text)

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove zero-width and control chars (keep \n \t)
    text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)
    text = re.sub(r"[^\S\n\t]+", " ", text)
    text = "".join(c for c in text if c == "\n" or c == "\t" or ord(c) >= 32)

    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            lines.append("")
            continue
        # Skip garbage lines (mostly symbols, very short)
        alpha = sum(1 for c in line if c.isalnum() or "\u0e00" <= c <= "\u0e7f")
        if len(line) < 3 and alpha == 0:
            continue
        if len(line) > 5 and alpha / len(line) < 0.1:
            continue
        lines.append(line)

    # Collapse 3+ blank lines to 2
    result = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run <= 2:
                result.append("")
        else:
            blank_run = 0
            result.append(line)

    return "\n".join(result).strip()


def decode_bytes(raw):
    from charset_normalizer import from_bytes
    result = from_bytes(raw).best()
    if result:
        return str(result)
    return raw.decode("utf-8", errors="ignore")


def read_text_file(filepath):
    with open(filepath, "rb") as f:
        raw = f.read()
    return normalize_text(decode_bytes(raw))


def extract_html(filepath):
    with open(filepath, "rb") as f:
        raw = f.read()
    html = decode_bytes(raw)
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return normalize_text(text)


def extract_xml(filepath):
    texts = []
    try:
        for _, elem in ET.iterparse(filepath, events=("end",)):
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if tag in ("text", "content", "body", "article"):
                if elem.text and len(elem.text) > 50:
                    t = elem.text
                    t = re.sub(r"\{\{.*?\}\}", "", t, flags=re.DOTALL)
                    t = re.sub(r"\[\[(.*?)\]\]", r"\1", t)
                    t = re.sub(r"<.*?>", "", t)
                    t = re.sub(r"&[a-z]+;", " ", t)
                    texts.append(t.strip())
            elem.clear()
    except ET.ParseError:
        return read_text_file(filepath)
    return normalize_text("\n\n".join(texts))


def extract_pdf(filepath):
    import fitz
    doc = fitz.open(filepath)
    parts = []
    for page in doc:
        t = page.get_text()
        if t.strip():
            parts.append(t)
    doc.close()
    return normalize_text("\n\n".join(parts))


def extract_epub(filepath):
    from ebooklib import epub
    from bs4 import BeautifulSoup as BS

    book = epub.read_epub(filepath)
    parts = []
    for item in book.get_items():
        if item.get_type() == 9:  # ITEM_DOCUMENT
            soup = BS(item.get_content(), "lxml")
            for s in soup(["script", "style"]):
                s.decompose()
            parts.append(soup.get_text(separator="\n"))
    return normalize_text("\n\n".join(parts))


def extract_docx(filepath):
    from docx import Document
    doc = Document(filepath)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    return normalize_text("\n".join(parts))


def _extract_archive_to_dir(filepath, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    lower = filepath.lower()

    if lower.endswith(".zip"):
        with zipfile.ZipFile(filepath, "r") as zf:
            zf.extractall(dest_dir)
        return dest_dir

    if lower.endswith(".7z"):
        import py7zr
        with py7zr.SevenZipFile(filepath, "r") as zf:
            zf.extractall(dest_dir)
        return dest_dir

    if lower.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(filepath, "r:*") as tf:
            tf.extractall(dest_dir)
        return dest_dir

    if lower.endswith(".gz") and not lower.endswith(".tar.gz"):
        out = os.path.join(dest_dir, Path(filepath).stem)
        with gzip.open(filepath, "rb") as f_in:
            with open(out, "wb") as f_out:
                f_out.write(f_in.read())
        return dest_dir

    if lower.endswith(".bz2"):
        out = os.path.join(dest_dir, Path(filepath).stem)
        with bz2.open(filepath, "rb") as f_in:
            with open(out, "wb") as f_out:
                f_out.write(f_in.read())
        return dest_dir

    if lower.endswith(".xz"):
        out = os.path.join(dest_dir, Path(filepath).stem)
        with lzma.open(filepath, "rb") as f_in:
            with open(out, "wb") as f_out:
                f_out.write(f_in.read())
        return dest_dir

    return None


def _is_archive(filepath):
    lower = filepath.lower()
    return lower.endswith((".zip", ".7z", ".tar", ".tar.gz", ".tgz", ".gz", ".bz2", ".xz"))


def _collect_files(directory):
    collected = []
    for root, _, files in os.walk(directory):
        for fname in files:
            collected.append(os.path.join(root, fname))
    return collected


def convert_file_at_path(filepath, *, source="upload", original_name=None):
    """Convert a single file (or archive) to .txt in data/. Returns (output_path, stats_dict)."""
    filepath = os.path.abspath(filepath)
    original_name = original_name or os.path.basename(filepath)
    ext = Path(original_name).suffix.lower()
    if not ext and "." in os.path.basename(filepath):
        ext = Path(filepath).suffix.lower()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    work_dir = tempfile.mkdtemp(dir=TEMP_DIR)
    try:
        text = _extract_text_recursive(filepath, work_dir, depth=0)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    if not text or len(text) < 10:
        raise ValueError(f"Converted text too short ({len(text) if text else 0} characters)")

    base = Path(original_name).stem
    base = re.sub(r"[^\w\-]", "_", base)[:80] or "converted"
    output_path = os.path.join(DATA_DIR, f"{base}.txt")
    # Avoid overwrite collision
    counter = 1
    while os.path.exists(output_path):
        output_path = os.path.join(DATA_DIR, f"{base}_{counter}.txt")
        counter += 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    stats = {
        "chars": len(text),
        "lines": text.count("\n") + 1,
        "unique_chars": len(set(text)),
        "format": ext.lstrip(".") or "unknown",
        "output": output_path,
        "filename": os.path.basename(output_path),
    }

    register_file(
        stats["filename"],
        source=source,
        original_name=original_name,
        fmt=stats["format"],
        chars=stats["chars"],
    )
    return output_path, stats


def _extract_text_recursive(filepath, work_dir, depth=0):
    if depth > 5:
        return ""

    lower = filepath.lower()

    if _is_archive(lower):
        extract_dir = os.path.join(work_dir, f"ext_{depth}")
        _extract_archive_to_dir(filepath, extract_dir)
        parts = []
        for fpath in _collect_files(extract_dir):
            part = _extract_text_recursive(fpath, work_dir, depth + 1)
            if part:
                parts.append(part)
        return normalize_text("\n\n".join(parts))

    if lower.endswith((".html", ".htm")):
        return extract_html(filepath)
    if lower.endswith(".xml"):
        return extract_xml(filepath)
    if lower.endswith(".pdf"):
        return extract_pdf(filepath)
    if lower.endswith(".epub"):
        return extract_epub(filepath)
    if lower.endswith(".docx"):
        return extract_docx(filepath)
    if lower.endswith((".txt", ".md")):
        return read_text_file(filepath)

    # Try as text
    try:
        return read_text_file(filepath)
    except Exception:
        return ""


def download_and_convert(url):
    r = requests.get(url, timeout=120, headers={"User-Agent": "NanoGPT-Trainer/1.0"})
    r.raise_for_status()

    filename = url.split("/")[-1].split("?")[0] or "downloaded"
    os.makedirs(DATA_DIR, exist_ok=True)
    tmppath = os.path.join(DATA_DIR, f"_tmp_{filename}")
    with open(tmppath, "wb") as f:
        f.write(r.content)

    try:
        return convert_file_at_path(tmppath, source="url", original_name=filename)
    finally:
        if os.path.exists(tmppath):
            os.remove(tmppath)


def upload_and_convert(uploaded_file):
    os.makedirs(DATA_DIR, exist_ok=True)
    tmppath = os.path.join(DATA_DIR, f"_tmp_{uploaded_file.name}")
    with open(tmppath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    try:
        return convert_file_at_path(tmppath, source="upload", original_name=uploaded_file.name)
    finally:
        if os.path.exists(tmppath):
            os.remove(tmppath)


def convert_for_preview(uploaded_file):
    """Convert without saving to registry (for tools tab test)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    tmppath = os.path.join(DATA_DIR, f"_tmp_preview_{uploaded_file.name}")
    with open(tmppath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    work_dir = tempfile.mkdtemp(dir=TEMP_DIR)
    try:
        text = _extract_text_recursive(tmppath, work_dir, depth=0)
        text = normalize_text(text) if text else ""
        return text, None
    except Exception as e:
        return None, str(e)
    finally:
        if os.path.exists(tmppath):
            os.remove(tmppath)
        shutil.rmtree(work_dir, ignore_errors=True)
