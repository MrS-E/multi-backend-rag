"""PDF conversion via Docling: PDF → Markdown plus extracted picture images.

Docling is imported lazily inside :func:`_build_converter`, so importing this
module (and therefore booting the app) never pulls in its heavy dependencies.
The converter is injectable so tests can supply a mocked ``DocumentConverter``.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

from app.rag.convert.base import KIND_TEXT, ConvertedDocument, ExtractedImage

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps docling out of import time
    from docling.document_converter import DocumentConverter

#: Raster format used when serializing Docling's PIL images.
_PICTURE_FORMAT = "PNG"


def _build_converter() -> DocumentConverter:
    """Construct a Docling converter that also renders picture images.

    Imported here (not at module import time) so the app boots without Docling's
    ~1 GB of models or its heavy dependency tree loaded.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def _picture_bytes(picture: Any, document: Any) -> bytes | None:
    """Render a Docling picture item to PNG bytes, or ``None`` if unavailable."""
    image = picture.get_image(document)
    if image is None:
        return None
    buffer = io.BytesIO()
    image.save(buffer, format=_PICTURE_FORMAT)
    return buffer.getvalue()


def convert(
    data: bytes,
    uri: str,
    converter: DocumentConverter | None = None,
) -> ConvertedDocument:
    """Convert PDF ``data`` to Markdown and collect its rendered pictures.

    ``converter`` is built lazily when omitted; passing one lets tests exercise
    the mapping without Docling's models.
    """
    from docling.datamodel.base_models import DocumentStream

    active = converter if converter is not None else _build_converter()
    stream = DocumentStream(name=uri or "document.pdf", stream=io.BytesIO(data))
    result = active.convert(stream)
    document = result.document

    images: list[ExtractedImage] = []
    for index, picture in enumerate(document.pictures):
        rendered = _picture_bytes(picture, document)
        if rendered is None:
            continue
        try:
            caption = picture.caption_text(document).strip()
        except Exception:  # noqa: BLE001 - a malformed caption must not fail conversion
            caption = ""
        images.append(
            ExtractedImage(
                data=rendered,
                alt=caption,
                position=index,
                uri=f"{uri}#picture-{index}",
            )
        )

    return ConvertedDocument(
        text=document.export_to_markdown(),
        images=images,
        kind=KIND_TEXT,
    )
