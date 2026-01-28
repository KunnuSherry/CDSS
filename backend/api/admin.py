import os
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.auth import require_role
from models.db import get_db
from models.document import DocumentPublic
from services.embedding import get_collection
from services.pdf_loader import (
    chunk_text_page_aware,
    extract_images_with_pages,
    extract_text_by_page,
)
from settings import settings


router = APIRouter()


def _safe_filename(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in (" ", ".", "_", "-")).strip().replace(" ", "_")


@router.get("/documents", response_model=list[DocumentPublic])
async def list_documents(_user=Depends(require_role("admin"))):
    db = get_db()
    docs = []
    async for d in db["documents"].find({}, sort=[("uploaded_at", -1)]):
        docs.append(
            DocumentPublic(
                id=d["_id"],
                pdf_name=d["pdf_name"],
                uploaded_at=d["uploaded_at"],
            )
        )
    return docs


@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    _user=Depends(require_role("admin")),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    os.makedirs(settings.upload_dir, exist_ok=True)
    db = get_db()

    doc_id = f"doc_{int(datetime.now(timezone.utc).timestamp() * 1000)}"
    pdf_name = _safe_filename(file.filename)

    doc_dir_abs = os.path.join(settings.upload_dir, doc_id)
    images_dir_abs = os.path.join(doc_dir_abs, "images")
    os.makedirs(images_dir_abs, exist_ok=True)

    pdf_abs_path = os.path.join(doc_dir_abs, pdf_name)
    with open(pdf_abs_path, "wb") as f:
        f.write(await file.read())

    # Extract
    pages = extract_text_by_page(pdf_abs_path)
    chunks = chunk_text_page_aware(pages)
    images = extract_images_with_pages(
        pdf_abs_path,
        out_dir_abs=images_dir_abs,
        out_dir_rel=f"{doc_id}/images",
    )

    uploaded_at = datetime.now(timezone.utc).isoformat()
    await db["documents"].insert_one(
        {
            "_id": doc_id,
            "pdf_name": pdf_name,
            "uploaded_at": uploaded_at,
            "pdf_path": pdf_abs_path,
        }
    )

    # Store chunks in Mongo (for traceability + exact text source)
    chunk_docs = []
    for c in chunks:
        chunk_id = f"{doc_id}_p{c.page}_c{c.chunk_index}"
        chunk_docs.append(
            {
                "_id": chunk_id,
                "doc_id": doc_id,
                "pdf_name": pdf_name,
                "page": c.page,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "heading": getattr(c, "heading", None),
            }
        )
    if chunk_docs:
        await db["chunks"].insert_many(chunk_docs)

    # Store image metadata
    img_docs = []
    for im in images:
        img_docs.append(
            {
                "doc_id": doc_id,
                "pdf_name": pdf_name,
                "page": im.page,
                "image_index": im.image_index,
                "rel_path": im.rel_path,
            }
        )
    if img_docs:
        await db["images"].insert_many(img_docs)

    # Ingest into Chroma (one collection per PDF for independent retrieval)
    col = get_collection(f"pdf_{doc_id}")
    if chunk_docs:
        metadatas = []
        for c in chunk_docs:
            m = {"doc_id": doc_id, "pdf_name": pdf_name, "page": c["page"], "chunk_index": c["chunk_index"]}
            if c.get("heading"):
                m["heading"] = c["heading"]
            metadatas.append(m)
        col.add(
            ids=[c["_id"] for c in chunk_docs],
            documents=[c["text"] for c in chunk_docs],
            metadatas=metadatas,
        )

    return {
        "doc_id": doc_id,
        "pdf_name": pdf_name,
        "uploaded_at": uploaded_at,
        "chunks": len(chunk_docs),
        "images": len(img_docs),
        "note": "Text is extracted from the PDF text layer (no OCR).",
    }

