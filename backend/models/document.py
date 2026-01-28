from pydantic import BaseModel


class DocumentPublic(BaseModel):
    id: str
    pdf_name: str
    uploaded_at: str


class ImagePublic(BaseModel):
    url: str
    page: int
    image_index: int


class ChunkPublic(BaseModel):
    chunk_id: str
    pdf_name: str
    doc_id: str
    page: int
    text: str
    heading: str | None = None


class SectionGroup(BaseModel):
    section_label: str  # e.g. "A", "B", "C"
    section_title: str  # e.g. "Definition & Classification"
    items: list[ChunkPublic]


class SearchResponse(BaseModel):
    query: str
    results: list[ChunkPublic]
    # Additional chunks beyond the primary 8–10 pages / 2–3 chunks per page.
    # Exposed separately so the UI can hide them behind an explicit toggle to
    # avoid "recall explosion" while keeping retrieval verbatim.
    advanced_results: list[ChunkPublic] | None = None
    images: list[ImagePublic]
    note: str | None = None
    groups: list[SectionGroup] | None = None
    coverage_note: str | None = None
    source_pdf: str | None = None
