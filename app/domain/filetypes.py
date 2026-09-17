"""File-type knowledge shared by the API edge and the RAG converters.

Kept in the domain layer so the API can validate uploads without importing the
RAG core, while the converters resolve the same extension/language mapping.
"""

from __future__ import annotations

import mimetypes
import os

#: Plain-text formats handled by the markdown passthrough converter.
MARKDOWN_EXTENSIONS: frozenset[str] = frozenset({".md", ".markdown", ".txt"})

#: Source-code extensions mapped to the language hint used in the code fence.
CODE_LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".sql": "sql",
    ".swift": "swift",
    ".scala": "scala",
    ".lua": "lua",
    ".r": "r",
    ".pl": "perl",
    ".m": "objectivec",
    ".mm": "objectivec",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".xml": "xml",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".ps1": "powershell",
}

#: Raster image formats captioned via the vision provider.
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"})

#: PDF documents converted to Markdown by Docling.
PDF_EXTENSIONS: frozenset[str] = frozenset({".pdf"})

#: Every extension the upload API accepts.
ALLOWED_EXTENSIONS: frozenset[str] = (
    MARKDOWN_EXTENSIONS
    | IMAGE_EXTENSIONS
    | PDF_EXTENSIONS
    | frozenset(CODE_LANGUAGE_BY_EXTENSION)
)


def extension_of(uri: str) -> str:
    """Return the lower-cased extension of ``uri`` (``""`` when there is none)."""
    return os.path.splitext(uri)[1].lower()


def is_allowed_uri(uri: str) -> bool:
    """Return whether ``uri`` names a file type the ingestion pipeline handles."""
    return extension_of(uri) in ALLOWED_EXTENSIONS


def mime_for_uri(uri: str) -> str:
    """Best-effort MIME type for a file name, with a text fallback for source files."""
    guessed, _ = mimetypes.guess_type(uri)
    if guessed:
        return guessed
    extension = extension_of(uri)
    if extension in CODE_LANGUAGE_BY_EXTENSION or extension in MARKDOWN_EXTENSIONS:
        return "text/plain"
    return "application/octet-stream"


def looks_like_image(data: bytes) -> bool:
    """Sniff common raster-image magic numbers (no external dependency)."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if data.startswith(b"\xff\xd8\xff"):
        return True
    if data.startswith((b"GIF87a", b"GIF89a")):
        return True
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return True
    return data.startswith((b"II*\x00", b"MM\x00*"))


def validate_content(data: bytes, uri: str) -> None:
    """Raise :class:`ValueError` when ``data`` does not match its extension.

    Only the two binary families are sniffed; text formats are accepted as-is
    because their content is decoded leniently by the converters.
    """
    extension = extension_of(uri)
    if extension in PDF_EXTENSIONS and not data.startswith(b"%PDF"):
        raise ValueError(f"file {uri!r} does not contain PDF data")
    if extension in IMAGE_EXTENSIONS and not looks_like_image(data):
        raise ValueError(f"file {uri!r} is not a recognised image")
