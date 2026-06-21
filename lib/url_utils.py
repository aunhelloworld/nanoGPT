"""Normalize pasted URL lists to one URL per line."""

import re

_URL_RE = re.compile(r"https?://[^\s,\]>\)\"']+", re.IGNORECASE)


def normalize_urls(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _URL_RE.finditer(text):
        url = match.group(0).rstrip(".,;)")
        if url not in seen:
            seen.add(url)
            ordered.append(url)
    if ordered:
        return ordered
    for part in re.split(r"[\n,\s;]+", text.strip()):
        part = part.strip()
        if part.startswith("http://") or part.startswith("https://"):
            if part not in seen:
                seen.add(part)
                ordered.append(part)
    return ordered


def format_urls_one_per_line(text: str) -> str:
    return "\n".join(normalize_urls(text))
