import os
import re
from dataclasses import dataclass

import fitz  # PyMuPDF

# ~4 chars per token heuristic. Target 300–600 tokens per chunk.
CHUNK_MIN_CHARS = 1200
CHUNK_MAX_CHARS = 2400
CHUNK_TARGET_CHARS = 1800
OVERLAP_SENTENCES = 2

# Don't split after these (case-insensitive). Period must follow.
ABBREV_PREFIXES = re.compile(
    r"(?i)(?:Fig|Figs|Figure|Table|Dr|Mr|Mrs|Ms|Prof|No|Vol|etc|e\.g|i\.e|vs|al|Sr|Jr|St)\."
)


@dataclass
class ExtractedImage:
    page: int
    image_index: int
    rel_path: str  # relative to UPLOAD_DIR


@dataclass
class ExtractedChunk:
    page: int
    chunk_index: int
    text: str
    heading: str | None = None  # preserved section/figure/table title when present


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def extract_text_by_page(pdf_path: str) -> list[tuple[int, str]]:
    """
    Extract text exactly as returned by the PDF text layer.
    NOTE: This is not OCR. If the PDF is scanned images, text may be empty.
    """
    doc = fitz.open(pdf_path)
    pages: list[tuple[int, str]] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        pages.append((i + 1, text))
    doc.close()
    return pages


def extract_images_with_pages(pdf_path: str, out_dir_abs: str, out_dir_rel: str) -> list[ExtractedImage]:
    """Extract embedded images with page numbers."""
    _ensure_dir(out_dir_abs)
    doc = fitz.open(pdf_path)
    images: list[ExtractedImage] = []
    for page_i in range(doc.page_count):
        page = doc.load_page(page_i)
        img_list = page.get_images(full=True)
        for idx, img in enumerate(img_list):
            xref = img[0]
            base = doc.extract_image(xref)
            img_bytes = base.get("image")
            ext = base.get("ext", "png")
            filename = f"page_{page_i+1:04d}_img_{idx:03d}.{ext}"
            abs_path = os.path.join(out_dir_abs, filename)
            with open(abs_path, "wb") as f:
                f.write(img_bytes)
            rel_path = "/".join([out_dir_rel.strip("/"), filename])
            images.append(ExtractedImage(page=page_i + 1, image_index=idx, rel_path=rel_path))
    doc.close()
    return images


def _sent_tokenize(text: str) -> list[str]:
    """Split into sentences. Avoid splitting after common abbreviations."""
    if not text.strip():
        return []
    # Split on . ! ? followed by space/newline and then uppercase. Avoid abbrevs.
    pattern = r"(?<=[.!?])\s+(?=[A-Z])"
    parts = re.split(pattern, text)
    out: list[str] = []
    i = 0
    while i < len(parts):
        s = parts[i].strip()
        if not s:
            i += 1
            continue
        # Merge "Fig." + "2" etc. so we don't split after abbreviations (not "Fig. 2" + "The next sentence")
        nxt = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if nxt and ABBREV_PREFIXES.search(s) and (len(nxt) < 25 or re.match(r"^[\dA-Za-z\.\s]+$", nxt)):
            s = s + " " + nxt
            i += 2
        else:
            i += 1
        if s:
            out.append(s)
    return out if out else [text.strip()]


def _looks_like_heading(line: str) -> bool:
    """Short line, no trailing period, or matches Figure/Table/Recommendations/COR/LOE."""
    t = line.strip()
    if not t or len(t) > 200:
        return False
    if re.search(r"(?i)(?:Figure|Fig\.?|Table|Recommendations?|COR|LOE|Class\s+[IIVX\d\-]+)", t):
        return True
    if re.search(r"[.!?]\s*$", t):
        return False
    if len(t) < 80 and "\n" not in t:
        return True
    return False


def _paragraphs_and_headings(text: str) -> list[tuple[str | None, str]]:
    """
    Split into (heading, paragraph) pairs. Heading is None for normal paragraphs.
    Leading heading is attached to the following paragraph.
    """
    blocks = re.split(r"\n\s*\n", text)
    result: list[tuple[str | None, str]] = []
    pending_heading: str | None = None
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        if _looks_like_heading(b):
            if pending_heading is not None:
                result.append((pending_heading, ""))
            pending_heading = b
            continue
        result.append((pending_heading, b))
        pending_heading = None
    if pending_heading is not None:
        result.append((pending_heading, ""))
    return result


def _build_chunks_for_page(page_num: int, text: str) -> list[ExtractedChunk]:
    """Sentence-based chunking with overlap. Preserve headings. 300–600 token proxy."""
    pairs = _paragraphs_and_headings(text)
    sentences_with_meta: list[tuple[str, str | None]] = []  # (sentence, heading for chunk start)
    for heading, para in pairs:
        sents = _sent_tokenize(para)
        h = heading if heading else None
        for i, s in enumerate(sents):
            if i == 0 and h is not None:
                sentences_with_meta.append((s, h))
            else:
                sentences_with_meta.append((s, None))
        if not sents and heading:
            sentences_with_meta.append((heading, heading))

    chunks: list[ExtractedChunk] = []
    start = 0
    chunk_idx = 0
    while start < len(sentences_with_meta):
        acc: list[str] = []
        chunk_heading: str | None = None
        size = 0
        i = start
        while i < len(sentences_with_meta) and size < CHUNK_MAX_CHARS:
            s, h = sentences_with_meta[i]
            if chunk_heading is None and h is not None:
                chunk_heading = h
            add = s + (" " if acc else "")
            if size + len(add) > CHUNK_MAX_CHARS and acc:
                break
            acc.append(s)
            size += len(add)
            if size >= CHUNK_TARGET_CHARS:
                i += 1
                break
            i += 1
        if not acc:
            break
        chunk_text = " ".join(acc)
        chunks.append(
            ExtractedChunk(
                page=page_num,
                chunk_index=chunk_idx,
                text=chunk_text,
                heading=chunk_heading,
            )
        )
        chunk_idx += 1
        # Overlap: next chunk starts overlap sentences before end of this one
        overlap = min(OVERLAP_SENTENCES, max(0, len(acc) - 1))
        start = start + len(acc) - overlap
        if start >= len(sentences_with_meta):
            break
    return chunks


def chunk_text_page_aware(
    pages: list[tuple[int, str]],
    max_chars: int = 1500,
    overlap_chars: int = 200,
) -> list[ExtractedChunk]:
    """
    Sentence-based chunking within each page.
    - 300–600 tokens per chunk (~1200–2400 chars).
    - Overlap of 1–2 sentences.
    - Never split mid-sentence.
    - Preserve headings (Figure/Table/Recommendations, short lines); attach to chunk.
    """
    out: list[ExtractedChunk] = []
    for page_num, text in pages:
        if not text.strip():
            continue
        page_chunks = _build_chunks_for_page(page_num, text)
        for c in page_chunks:
            out.append(c)
    return out
