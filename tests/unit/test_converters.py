"""Tests for the format converters (markdown, code, PDF, standalone image).

PDF conversion is exercised with a mocked ``DocumentConverter`` so Docling's
models are never loaded; the other converters are pure and need no mocking.
"""

from __future__ import annotations

import io
from typing import Any

from PIL import Image

from app.rag.convert import code as code_converter
from app.rag.convert import convert_document, pdf_docling
from app.rag.convert import image as image_converter
from app.rag.convert import markdown as markdown_converter
from app.rag.convert.base import KIND_CODE, KIND_TEXT, extract_image_refs


def _pil_image(color: str = "red", size: tuple[int, int] = (4, 4)) -> Image.Image:
    """A real PIL image, matching what Docling's ``picture.get_image`` returns."""
    return Image.new("RGB", size, color=color)


def _png_bytes(color: str = "red", size: tuple[int, int] = (4, 4)) -> bytes:
    buffer = io.BytesIO()
    _pil_image(color, size).save(buffer, format="PNG")
    return buffer.getvalue()


# --- markdown passthrough -------------------------------------------------


def test_given_markdown_when_converting_then_text_preserved_verbatim() -> None:
    source = "# Title\n\nSome *prose* with a [link](http://example.com).\n"
    data = source.encode("utf-8")

    result = markdown_converter.convert(data, "notes.md")

    assert result.text == source
    assert result.kind == KIND_TEXT


def test_given_markdown_with_image_when_converting_then_image_ref_extracted() -> None:
    text = "Intro\n\n![a diagram](images/diagram.png)\n\nOutro\n"
    data = text.encode("utf-8")
    resolved: list[str] = []

    def resolve(path: str) -> bytes | None:
        resolved.append(path)
        return b"image-bytes"

    result = markdown_converter.convert(data, "notes.md", resolve_image=resolve)

    assert resolved == ["images/diagram.png"]
    assert len(result.images) == 1
    image = result.images[0]
    assert image.data == b"image-bytes"
    assert image.alt == "a diagram"
    assert image.uri == "images/diagram.png"
    assert image.position == text.index("![a diagram]")


def test_given_markdown_with_unresolvable_image_when_converting_then_image_skipped() -> None:
    data = b"![missing](images/nope.png)\n"

    result = markdown_converter.convert(data, "notes.md", resolve_image=lambda _path: None)

    assert result.images == []
    assert result.text == "![missing](images/nope.png)\n"


def test_given_markdown_without_resolver_when_converting_then_no_images_collected() -> None:
    data = b"![present](images/present.png)\n"

    result = markdown_converter.convert(data, "notes.md")

    assert result.images == []


def test_given_remote_image_when_extracting_refs_then_skipped() -> None:
    text = "![remote](https://example.com/x.png) ![local](x.png)"

    refs = extract_image_refs(text)

    assert [ref.path for ref in refs] == ["x.png"]


def test_given_markdown_with_title_when_extracting_refs_then_title_ignored() -> None:
    refs = extract_image_refs('![alt](path/to.png "the title")')

    assert len(refs) == 1
    assert refs[0].path == "path/to.png"
    assert refs[0].alt == "alt"


# --- code -----------------------------------------------------------------


def test_given_python_file_when_converting_then_python_fence() -> None:
    result = code_converter.convert(b"print('hi')", "script.py")

    assert result.kind == KIND_CODE
    assert result.text == "```python\nprint('hi')\n```"
    assert result.images == []


def test_given_typescript_file_when_converting_then_typescript_fence() -> None:
    result = code_converter.convert(b"const x = 1", "app.ts")

    assert result.text == "```typescript\nconst x = 1\n```"


def test_given_go_file_when_converting_then_go_fence() -> None:
    result = code_converter.convert(b"package main", "main.go")

    assert result.text == "```go\npackage main\n```"


def test_given_unknown_extension_when_converting_then_unlabelled_fence() -> None:
    result = code_converter.convert(b"mystery", "file.zzz")

    assert result.kind == KIND_CODE
    assert result.text == "```\nmystery\n```"


def test_given_uppercase_extension_when_resolving_language_then_lower_cased() -> None:
    assert code_converter.language_for("SCRIPT.PY") == "python"


def test_given_no_extension_when_resolving_language_then_empty() -> None:
    assert code_converter.language_for("Makefile") == ""


# --- pdf (mocked DocumentConverter) ---------------------------------------


class _FakePicture:
    """Minimal Docling picture stand-in: an image plus an optional caption."""

    def __init__(self, image: Image.Image | None, caption: str = "") -> None:
        self._image = image
        self._caption = caption

    def get_image(self, _document: Any) -> Image.Image | None:
        return self._image

    def caption_text(self, _document: Any) -> str:
        return self._caption


class _FakeDocument:
    def __init__(self, markdown: str, pictures: list[_FakePicture]) -> None:
        self._markdown = markdown
        self.pictures = pictures

    def export_to_markdown(self) -> str:
        return self._markdown


class _FakeResult:
    def __init__(self, document: _FakeDocument) -> None:
        self.document = document


class _FakeConverter:
    """Records the converted stream and returns a canned document."""

    def __init__(self, document: _FakeDocument) -> None:
        self._document = document
        self.streams: list[Any] = []

    def convert(self, stream: Any) -> _FakeResult:
        self.streams.append(stream)
        return _FakeResult(self._document)


def test_given_pdf_when_converting_then_markdown_and_pictures_returned() -> None:
    document = _FakeDocument(
        markdown="# Page one\n\ntext",
        pictures=[_FakePicture(_pil_image("blue"), caption="a blue square")],
    )
    converter = _FakeConverter(document)

    result = pdf_docling.convert(b"%PDF-1.4 fake", "report.pdf", converter=converter)

    assert result.text == "# Page one\n\ntext"
    assert result.kind == KIND_TEXT
    assert len(result.images) == 1
    image = result.images[0]
    assert image.data.startswith(b"\x89PNG")
    assert image.alt == "a blue square"
    assert image.uri == "report.pdf#picture-0"
    assert image.position == 0


def test_given_picture_without_image_when_converting_then_picture_skipped() -> None:
    document = _FakeDocument(markdown="body", pictures=[_FakePicture(None)])
    converter = _FakeConverter(document)

    result = pdf_docling.convert(b"%PDF-1.4 fake", "report.pdf", converter=converter)

    assert result.images == []


def test_given_picture_caption_raises_when_converting_then_caption_blank() -> None:
    class _BrokenCaptionPicture(_FakePicture):
        def caption_text(self, _document: Any) -> str:
            raise RuntimeError("no caption")

    document = _FakeDocument(
        markdown="body", pictures=[_BrokenCaptionPicture(_pil_image("green"))]
    )
    converter = _FakeConverter(document)

    result = pdf_docling.convert(b"%PDF-1.4 fake", "report.pdf", converter=converter)

    assert len(result.images) == 1
    assert result.images[0].alt == ""


def test_given_multiple_pictures_when_converting_then_positions_and_uris_indexed() -> None:
    document = _FakeDocument(
        markdown="body",
        pictures=[_FakePicture(_pil_image("red")), _FakePicture(_pil_image("blue"))],
    )
    converter = _FakeConverter(document)

    result = pdf_docling.convert(b"%PDF-1.4 fake", "report.pdf", converter=converter)

    assert [image.uri for image in result.images] == [
        "report.pdf#picture-0",
        "report.pdf#picture-1",
    ]
    assert [image.position for image in result.images] == [0, 1]


def test_given_pdf_when_converting_then_stream_named_after_uri() -> None:
    converter = _FakeConverter(_FakeDocument(markdown="body", pictures=[]))

    pdf_docling.convert(b"%PDF-1.4 fake", "report.pdf", converter=converter)

    assert converter.streams[0].name == "report.pdf"


# --- standalone image -----------------------------------------------------


def test_given_image_when_converting_then_single_image_and_empty_text() -> None:
    data = _png_bytes("purple")

    result = image_converter.convert(data, "photo.png")

    assert result.text == ""
    assert result.kind == KIND_TEXT
    assert len(result.images) == 1
    extracted = result.images[0]
    assert extracted.data == data
    assert extracted.alt == ""
    assert extracted.position == 0
    assert extracted.uri == "photo.png"


# --- dispatch -------------------------------------------------------------


def test_given_markdown_uri_when_dispatching_then_markdown_converter() -> None:
    result = convert_document(b"# heading\n", "doc.markdown")

    assert result.text == "# heading\n"
    assert result.kind == KIND_TEXT


def test_given_code_uri_when_dispatching_then_code_converter() -> None:
    result = convert_document(b"x", "main.go")

    assert result.kind == KIND_CODE


def test_given_image_uri_when_dispatching_then_image_converter() -> None:
    result = convert_document(_png_bytes(), "photo.webp".replace(".webp", ".png"))

    assert result.text == ""
    assert len(result.images) == 1
