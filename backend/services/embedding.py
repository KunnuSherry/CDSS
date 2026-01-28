from __future__ import annotations

import os

import chromadb
from chromadb.utils import embedding_functions

from settings import settings


def get_chroma_client() -> chromadb.PersistentClient:
    os.makedirs(settings.chroma_dir, exist_ok=True)
    return chromadb.PersistentClient(path=settings.chroma_dir)


def get_embedding_function():
    # Local embeddings so retrieval does not depend on LLM.
    # TODO: If you prefer remote embeddings, swap this out explicitly.
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


def get_collection(name: str):
    client = get_chroma_client()
    ef = get_embedding_function()
    return client.get_or_create_collection(name=name, embedding_function=ef)

