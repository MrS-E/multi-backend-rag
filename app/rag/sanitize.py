"""Content sanitization for injection-safe prompt assembly.

Applied at **prompt-assembly time**, not at ingest: ingested content stays
faithful in storage and Qdrant payloads (plan Key Decision 8), and the escaping
below happens only when a chunk is wrapped for the model.

Two independent concerns, both applied to every text that may end up inside a
``<![CDATA[...]]>`` envelope:

* **invisible characters** — bidi controls and zero-width characters are stripped
  so they cannot smuggle or reorder text past a reviewer;
* **CDATA terminators** — a literal ``]]>`` is rewritten to ``]]]]><![CDATA[>``,
  which keeps the surrounding CDATA section well-formed while preserving the
  original characters when the document is read back.

Both functions are pure: same input, same output, no I/O.
"""

from __future__ import annotations

#: Replacement for a literal ``]]>`` (closes and immediately reopens the section).
CDATA_ESCAPE = "]]]]><![CDATA[>"

#: The sequence that would terminate a CDATA section early.
CDATA_TERMINATOR = "]]>"

#: Bidi formatting controls (Unicode category ``Cf``) that can reorder rendering.
BIDI_CONTROL_CHARS: frozenset[str] = frozenset(
    {
        "\u061c",  # ARABIC LETTER MARK
        "\u200e",  # LEFT-TO-RIGHT MARK
        "\u200f",  # RIGHT-TO-LEFT MARK
        "\u202a",  # LEFT-TO-RIGHT EMBEDDING
        "\u202b",  # RIGHT-TO-LEFT EMBEDDING
        "\u202c",  # POP DIRECTIONAL FORMATTING
        "\u202d",  # LEFT-TO-RIGHT OVERRIDE
        "\u202e",  # RIGHT-TO-LEFT OVERRIDE
        "\u2066",  # LEFT-TO-RIGHT ISOLATE
        "\u2067",  # RIGHT-TO-LEFT ISOLATE
        "\u2068",  # FIRST STRONG ISOLATE
        "\u2069",  # POP DIRECTIONAL ISOLATE
    }
)

#: Zero-width and invisible characters that can hide instructions from a reader.
ZERO_WIDTH_CHARS: frozenset[str] = frozenset(
    {
        "\u200b",  # ZERO WIDTH SPACE
        "\u200c",  # ZERO WIDTH NON-JOINER
        "\u200d",  # ZERO WIDTH JOINER
        "\u2060",  # WORD JOINER
        "\ufeff",  # ZERO WIDTH NO-BREAK SPACE / BOM
        "\u00ad",  # SOFT HYPHEN
    }
)

#: Every character stripped by :func:`strip_invisible`.
INVISIBLE_CHARS: frozenset[str] = BIDI_CONTROL_CHARS | ZERO_WIDTH_CHARS


def strip_invisible(text: str) -> str:
    """Remove bidi controls and zero-width characters from ``text``.

    Only the explicitly listed characters are removed; other ``Cf`` characters
    (e.g. emoji variation selectors) are left untouched to avoid mangling
    legitimate content.
    """
    if not any(char in text for char in INVISIBLE_CHARS):
        return text
    return "".join(char for char in text if char not in INVISIBLE_CHARS)


def escape_cdata(text: str) -> str:
    """Neutralize ``]]>`` sequences so ``text`` can sit inside a CDATA section."""
    return text.replace(CDATA_TERMINATOR, CDATA_ESCAPE)


def sanitize(text: str) -> str:
    """Apply the full sanitization pipeline: strip invisibles, then escape ``]]>``.

    Deliberately no Unicode normalization: it would rewrite legitimate source
    code and prose, and the checks below operate on literal characters anyway.
    """
    return escape_cdata(strip_invisible(text))
