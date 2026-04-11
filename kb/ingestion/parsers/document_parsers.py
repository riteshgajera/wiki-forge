"""
Document parsers for text/markdown, PDF, and image files.
Each parser returns a normalized ParsedDocument.
"""
from __future__ import annotations

import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kb.utils.logging import get_logger

logger = get_logger("ingestion.parsers")


@dataclass
class ParsedDocument:
    path: str
    content: str
    mime_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    page_count: int = 1
    word_count: int = 0
    error: str = ""

    def __post_init__(self) -> None:
        if self.content and not self.word_count:
            self.word_count = len(self.content.split())


# ─── Text / Markdown Parser ───────────────────────────────────────────────────

class TextParser:
    """Parse plain text and markdown files."""

    SUPPORTED = {".txt", ".md", ".rst", ".html", ".htm", ".csv"}

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUPPORTED

    def parse(self, path: Path) -> ParsedDocument:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            mime = mimetypes.guess_type(str(path))[0] or "text/plain"
            return ParsedDocument(
                path=str(path),
                content=content,
                mime_type=mime,
                metadata={"filename": path.name, "suffix": path.suffix},
            )
        except OSError as e:
            return ParsedDocument(
                path=str(path), content="", mime_type="text/plain",
                error=str(e)
            )


# ─── PDF Parser ───────────────────────────────────────────────────────────────

class PDFParser:
    """Parse PDF files using PyMuPDF (fitz)."""

    SUPPORTED = {".pdf"}

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUPPORTED

    def parse(self, path: Path) -> ParsedDocument:
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(path))
            pages: list[str] = []
            for page in doc:
                pages.append(page.get_text("text"))
            doc.close()

            content = "\n\n".join(p.strip() for p in pages if p.strip())
            meta = doc.metadata if hasattr(doc, "metadata") else {}

            return ParsedDocument(
                path=str(path),
                content=content,
                mime_type="application/pdf",
                page_count=len(pages),
                metadata={
                    "filename": path.name,
                    "title": meta.get("title", ""),
                    "author": meta.get("author", ""),
                    "subject": meta.get("subject", ""),
                    "page_count": len(pages),
                },
            )
        except ImportError:
            return ParsedDocument(
                path=str(path), content="", mime_type="application/pdf",
                error="PyMuPDF not installed"
            )
        except Exception as e:
            logger.error("pdf_parse_error", path=str(path), error=str(e))
            return ParsedDocument(
                path=str(path), content="", mime_type="application/pdf",
                error=str(e)
            )


# ─── Image Parser ─────────────────────────────────────────────────────────────

class ImageParser:
    """Extract metadata from images. Optional OCR via pytesseract."""

    SUPPORTED = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"}

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUPPORTED

    def parse(self, path: Path) -> ParsedDocument:
        meta: dict[str, Any] = {"filename": path.name}
        content_parts: list[str] = []

        try:
            from PIL import Image
            img = Image.open(path)
            meta["width"] = img.width
            meta["height"] = img.height
            meta["mode"] = img.mode
            meta["format"] = img.format
            if hasattr(img, "_getexif") and img._getexif():
                exif = img._getexif() or {}
                meta["exif"] = {str(k): str(v)[:200] for k, v in exif.items()}
            content_parts.append(
                f"Image: {path.name} ({img.width}x{img.height}, {img.mode})"
            )
        except ImportError:
            content_parts.append(f"Image: {path.name}")
        except Exception as e:
            logger.warning("image_meta_error", path=str(path), error=str(e))
            content_parts.append(f"Image: {path.name}")

        # Optional OCR
        try:
            import pytesseract
            from PIL import Image
            ocr_text = pytesseract.image_to_string(Image.open(path))
            if ocr_text.strip():
                meta["ocr_text"] = ocr_text.strip()
                content_parts.append(f"\nOCR Text:\n{ocr_text.strip()}")
        except (ImportError, Exception):
            pass  # OCR is optional

        return ParsedDocument(
            path=str(path),
            content="\n".join(content_parts),
            mime_type=mimetypes.guess_type(str(path))[0] or "image/unknown",
            metadata=meta,
        )


# ─── Parser Registry ──────────────────────────────────────────────────────────

class ParserRegistry:
    """Routes files to the appropriate parser."""

    def __init__(self) -> None:
        self._parsers = [TextParser(), PDFParser(), ImageParser()]

    def parse(self, path: Path) -> ParsedDocument:
        for parser in self._parsers:
            if parser.can_parse(path):
                logger.debug("parsing_file", path=str(path), parser=type(parser).__name__)
                return parser.parse(path)
        return ParsedDocument(
            path=str(path), content="", mime_type="unknown",
            error=f"No parser for extension: {path.suffix}"
        )

    def can_parse(self, path: Path) -> bool:
        return any(p.can_parse(path) for p in self._parsers)
