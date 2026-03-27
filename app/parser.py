import io
import logging
import re
import zipfile
from datetime import datetime
from pathlib import Path

from pdfminer.high_level import extract_text as pdf_extract_text
from pdfminer.pdfparser import PDFSyntaxError

logger = logging.getLogger(__name__)

_SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ""}
_DEBUG_DIR = Path(__file__).parent.parent / "temp" / "parser_debug"


class ParseError(Exception):
    """Raised when text extraction fails for a given file."""


def extract_resume_text(file_bytes: bytes, filename: str) -> str:
    """Extract and clean text from PDF or TXT file bytes."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_bytes(file_bytes, filename)
    if suffix in {".txt", ".md", ""}:
        return _extract_txt_bytes(file_bytes, filename)
    raise ParseError(f"Unsupported file type: {suffix!r}. Expected .pdf or .txt.")


def extract_zip_resumes(zip_bytes: bytes) -> list[tuple[str, str]]:
    """
    Extract all supported resume files from a ZIP archive.

    Returns list of (filename, text) tuples, skipping unsupported or
    unreadable files with a warning rather than raising.
    """
    results: list[tuple[str, str]] = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for info in zf.infolist():
                name = info.filename
                # Skip directories, macOS metadata, hidden files
                if info.is_dir():
                    continue
                basename = Path(name).name
                if basename.startswith(".") or "__MACOSX" in name:
                    continue
                if Path(name).suffix.lower() not in _SUPPORTED_SUFFIXES:
                    continue
                try:
                    text = extract_resume_text(zf.read(name), basename)
                    results.append((basename, text))
                except ParseError as exc:
                    logger.warning("Skipping '%s' in ZIP: %s", name, exc)
    except zipfile.BadZipFile as exc:
        raise ParseError(f"Invalid ZIP file: {exc}") from exc

    if not results:
        raise ParseError("ZIP contained no supported resume files (.pdf / .txt).")
    return results


def extract_text_from_path(path: Path) -> str:
    """Extract and clean text from a file path (used by eval notebook)."""
    content = path.read_bytes()
    return extract_resume_text(content, path.name)


def clean_text(text: str) -> str:
    """Normalize whitespace and remove special characters."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\-\.,]", " ", text)
    return text.strip()


def _safe_debug_name(filename: str) -> str:
    stem = Path(filename).stem or "document"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "document"


def _write_debug_text(filename: str, text: str) -> None:
    try:
        _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
        debug_path = _DEBUG_DIR / f"{_safe_debug_name(filename)}_{timestamp}.txt"
        debug_path.write_text(text, encoding="utf-8")
        logger.info("Saved extracted PDF text for debugging: %s", debug_path)
    except Exception as exc:
        logger.warning("Failed to write debug text for '%s': %s", filename, exc)


def _extract_pdf_bytes(file_bytes: bytes, filename: str) -> str:
    try:
        text = pdf_extract_text(io.BytesIO(file_bytes))
    except PDFSyntaxError as exc:
        raise ParseError(f"Failed to parse PDF '{filename}': {exc}") from exc
    text = clean_text(text)
    if not text:
        raise ParseError(f"PDF '{filename}' produced no extractable text (may be scanned/image-only).")
    _write_debug_text(filename, text)
    return text


def _extract_txt_bytes(file_bytes: bytes, filename: str) -> str:
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1", errors="ignore")
    text = clean_text(text)
    if not text:
        raise ParseError(f"File '{filename}' is empty.")
    return text
