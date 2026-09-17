"""Shared converter result types.

Every converter returns a :class:`ConvertedDocument`: the extracted text plus any
images found along the way. Downstream stages (caption → sanitize → chunk →
embed → index) work on this uniform shape regardless of the source format.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

#: Chunk kinds recorded in the Qdrant payload.
KIND_TEXT = "text"
KIND_IMAGE_CAPTION = "image_caption"
KIND_CODE = "code"

#: Markdown image syntax ``![alt](path "optional title")`` (ported from the PoC).
IMAGE_REF_PATTERN = re.compile(r'!\[(.*?)\]\(([^)\s]+)(?:\s+"[^"]*")?\)')

#: Resolves an image reference to its bytes, or ``None`` when unavailable.
ImageResolver = Callable[[str], "bytes | None"]


@dataclass(frozen=True, slots=True)
class ImageRef:
    """A markdown image reference: alt text, target path and character offset."""

    alt: str
    path: str
    position: int


@dataclass(frozen=True, slots=True)
class ExtractedImage:
    """An image extracted by a converter, ready for captioning.

    ``position`` is the character offset of the reference in the source text
    (``0`` for standalone images) and ``uri`` the original reference, so a chunk
    can point back at where the image came from.
    """

    data: bytes
    alt: str = ""
    position: int = 0
    uri: str = ""


@dataclass(frozen=True, slots=True)
class ConvertedDocument:
    """Uniform converter output: text plus extracted images.

    ``kind`` labels the text portion (``text`` for prose, ``code`` for source
    files). Images always become ``image_caption`` chunks after captioning.
    """

    text: str
    images: list[ExtractedImage] = field(default_factory=list)
    kind: str = KIND_TEXT


def extract_image_refs(text: str) -> list[ImageRef]:
    """Return every markdown image reference in ``text``, in document order.

    Remote (``http://``/``https://``) targets are skipped, mirroring the PoC:
    they cannot be read from the local filesystem.
    """
    refs: list[ImageRef] = []
    for match in IMAGE_REF_PATTERN.finditer(text):
        alt, raw = match.group(1), match.group(2)
        if raw.startswith(("http://", "https://")):
            continue
        refs.append(ImageRef(alt=alt.strip(), path=raw, position=match.start()))
    return refs
