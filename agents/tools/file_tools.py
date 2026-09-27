"""Uploaded-file storage and the read_file tool.

Files uploaded through Streamlit are saved under data/uploads/<task_id>/ with sanitized
names. The read_file tool can only open files that belong to the current task's upload
list, and each resolved path must stay inside that task's upload directory, so there is
no arbitrary filesystem access.
"""

from __future__ import annotations

import csv
import io
import re
import statistics
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.config import get_settings
from core.schemas import UploadedFile

SUPPORTED_TYPES = {".txt": "text", ".md": "markdown", ".pdf": "pdf", ".csv": "csv"}


class FileAccessError(PermissionError):
    pass


class ReadFileInput(BaseModel):
    file_name: str = Field(description="Exact name of an uploaded file for this task")


class ReadFileOutput(BaseModel):
    file_name: str
    file_type: str
    size_bytes: int
    metadata: dict[str, str]
    text: str
    truncated: bool


def _sanitize(name: str) -> str:
    base = Path(name).name
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .") or "file"
    return base[:120]


def save_uploads(task_id: str, files: list[tuple[str, bytes]]) -> tuple[list[UploadedFile], list[str]]:
    """Persist uploads for a task. Returns (saved files, user-facing rejection messages)."""
    settings = get_settings()
    task_dir = (settings.uploads_dir / _sanitize(task_id)).resolve()
    task_dir.mkdir(parents=True, exist_ok=True)
    saved: list[UploadedFile] = []
    rejected: list[str] = []
    for original_name, data in files:
        name = _sanitize(original_name)
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_TYPES:
            rejected.append(f"{original_name}: unsupported type (allowed: {', '.join(SUPPORTED_TYPES)})")
            continue
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            rejected.append(f"{original_name}: larger than {settings.max_upload_mb} MB")
            continue
        path = task_dir / name
        path.write_bytes(data)
        saved.append(UploadedFile(file_name=name, file_type=SUPPORTED_TYPES[suffix], size_bytes=len(data),
                                  stored_path=str(path)))
    return saved, rejected


def _resolve(file_name: str, uploads: list[UploadedFile]) -> UploadedFile:
    settings = get_settings()
    root = settings.uploads_dir.resolve()
    for item in uploads:
        if item.file_name == file_name or item.file_name == _sanitize(file_name):
            path = Path(item.stored_path).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise FileAccessError("File is outside the upload area or no longer exists.")
            return item
    available = ", ".join(u.file_name for u in uploads) or "none"
    raise FileAccessError(f"'{file_name}' is not an uploaded file for this task. Available: {available}")


def _read_pdf(path: Path) -> tuple[str, dict[str, str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    meta = {"pages": str(len(reader.pages))}
    info = reader.metadata or {}
    for key in ("title", "author", "subject"):
        value = getattr(info, key, None)
        if value:
            meta[key] = str(value)
    text = "\n\n".join(f"[page {i + 1}]\n{p}" for i, p in enumerate(pages) if p)
    if not text:
        meta["note"] = "No extractable text (the PDF may be scanned images)."
    return text, meta


def _read_csv(path: Path) -> tuple[str, dict[str, str]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(raw[:5000])
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.reader(io.StringIO(raw), dialect))
    if not rows:
        return "", {"rows": "0"}
    header, body = rows[0], rows[1:]
    meta = {"rows": str(len(body)), "columns": ", ".join(header)}
    stats_lines = []
    for col_idx, col in enumerate(header):
        values = []
        for row in body:
            if col_idx < len(row):
                try:
                    values.append(float(row[col_idx].replace(",", "")))
                except ValueError:
                    pass
        if len(values) >= max(1, len(body) // 2):
            stats_lines.append(
                f"- {col}: n={len(values)}, min={min(values):,.4g}, max={max(values):,.4g}, "
                f"mean={statistics.fmean(values):,.4g}, sum={sum(values):,.4g}"
            )
    preview = "\n".join(" | ".join(r) for r in rows[:51])
    text = f"Columns: {', '.join(header)}\nRows: {len(body)}\n"
    if stats_lines:
        text += "Numeric column statistics:\n" + "\n".join(stats_lines) + "\n"
    text += f"\nFirst {min(50, len(body))} rows:\n{preview}"
    return text, meta


def read_uploaded_file(file_name: str, uploads: list[UploadedFile]) -> ReadFileOutput:
    settings = get_settings()
    item = _resolve(file_name, uploads)
    path = Path(item.stored_path)
    if item.file_type == "pdf":
        text, meta = _read_pdf(path)
    elif item.file_type == "csv":
        text, meta = _read_csv(path)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        meta = {"lines": str(text.count("\n") + 1), "characters": str(len(text))}
    truncated = len(text) > settings.max_file_chars
    return ReadFileOutput(
        file_name=item.file_name, file_type=item.file_type, size_bytes=item.size_bytes, metadata=meta,
        text=text[: settings.max_file_chars], truncated=truncated,
    )


def run(args: ReadFileInput, context: Any) -> ReadFileOutput:
    return read_uploaded_file(args.file_name, context.uploads)
