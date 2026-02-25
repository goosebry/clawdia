"""Tool: read_pdf — convert a local PDF to a cached Markdown file."""
from __future__ import annotations

import logging
from pathlib import Path

from ella.tools.registry import ella_tool

logger = logging.getLogger(__name__)


@ella_tool(
    name="read_pdf",
    description=(
        "Convert a local PDF file to a Markdown file saved alongside the original "
        "(same path, .md extension) and return the .md file path. "
        "Use after download_file when the downloaded file is a PDF. "
        "Then use read_file on the returned path to read the content. "
        "Images inside the PDF are not described — only text is extracted. "
        "The .md file is cached permanently: if it already exists it is returned immediately."
    ),
)
def read_pdf(path: str) -> str:
    """Extract text from a PDF and save as a Markdown file, returning the .md path.

    path: absolute local path to the PDF file (as returned by download_file)
    """
    try:
        import pymupdf  # type: ignore  # pip install pymupdf
    except ImportError:
        return "Error: pymupdf is not installed. Run: pip install pymupdf"

    pdf_path = Path(path)
    if not pdf_path.exists():
        return f"Error: file not found: {path}"
    if pdf_path.suffix.lower() != ".pdf":
        return f"Error: not a PDF file: {path}"

    md_path = pdf_path.with_suffix(".md")
    if md_path.exists():
        logger.info("[read_pdf] Cache hit: %s", md_path)
        return str(md_path)

    logger.info("[read_pdf] Converting %s → %s", pdf_path.name, md_path.name)
    try:
        doc = pymupdf.open(str(pdf_path))
        parts: list[str] = []

        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks")  # [(x0, y0, x1, y1, text, block_no, block_type)]
            page_lines: list[str] = []
            for block in blocks:
                text = block[4].strip() if len(block) > 4 else ""
                if not text:
                    continue
                # Heuristic: short all-caps or large text → heading
                lines = text.splitlines()
                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if len(stripped) < 80 and stripped.isupper():
                        page_lines.append(f"## {stripped.title()}")
                    else:
                        page_lines.append(stripped)
            if page_lines:
                parts.append(f"<!-- Page {page_num} -->\n" + "\n".join(page_lines))

        doc.close()
        md_content = "\n\n".join(parts)
        md_path.write_text(md_content, encoding="utf-8")
        logger.info("[read_pdf] Wrote %d chars to %s", len(md_content), md_path)
        return str(md_path)

    except Exception as e:
        md_path.unlink(missing_ok=True)
        return f"Error converting PDF {path}: {e}"
