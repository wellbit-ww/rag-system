import os
import time
import fitz

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
)

from llama_index.core.schema import Document

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from ingestion.chunking import get_text_splitter
from services.chroma_service import get_chroma_collection


# =====================================================
# PDF LOADER
# =====================================================

def load_pdf(file_path: str):

    print("[PDF] Opening PDF with PyMuPDF...")

    pdf = fitz.open(file_path)

    full_text = ""

    print(f"[PDF] Total pages: {len(pdf)}")

    for page_number, page in enumerate(pdf):

        text = page.get_text()

        print(
            f"[PDF] Page {page_number + 1} "
            f"characters: {len(text)}"
        )

        full_text += text + "\n"

    pdf.close()

    print(
        f"[PDF] Total extracted characters: "
        f"{len(full_text)}"
    )

    return [
        Document(text=full_text)
    ]


# =====================================================
# INGEST FUNCTION
# =====================================================

def ingest_file(file_path: str, project: str):

    start_time = time.time()

    print("\n" + "=" * 50)
    print("START INGESTION")
    print("=" * 50)

    print(f"[1] File path: {file_path}")
    print(f"[2] Project: {project}")

    # =====================================================
    # CHROMA DB
    # =====================================================

    print("\n[3] Connecting to ChromaDB...")

    collection = get_chroma_collection()

    print("[OK] Chroma connected")

    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    # =====================================================
    # EMBEDDING MODEL
    # =====================================================

    print("\n[4] Loading embedding model...")

    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434"
    )

    print("[OK] Embedding model loaded")

    # =====================================================
    # LOAD PDF
    # =====================================================

    print("\n[5] Loading PDF document...")

    documents = load_pdf(file_path)

    print(f"[OK] Documents loaded: {len(documents)}")

    # =====================================================
    # ADD METADATA TO DOCUMENTS
    # =====================================================

    print("\n[6] Adding metadata...")

    for doc in documents:

        doc.metadata = {
            "project": project,
            "filename": os.path.basename(file_path),
        }

    print("[OK] Metadata added")

    # =====================================================
    # CHUNKING
    # =====================================================

    print("\n[7] Creating chunks...")

    splitter = get_text_splitter()

    nodes = splitter.get_nodes_from_documents(documents)

    # =====================================================
    # ADD METADATA TO EVERY CHUNK
    # =====================================================

    for node in nodes:

        node.metadata = {
            "project": project,
            "filename": os.path.basename(file_path),
        }

    print(f"[OK] Chunks created: {len(nodes)}")

    # =====================================================
    # SHOW FIRST CHUNKS
    # =====================================================

    for i, node in enumerate(nodes[:5]):

        chunk_text = (
            node.text[:300]
            .replace("\n", " ")
            .strip()
        )

        print(f"\n--- CHUNK {i+1} ---")
        print(f"Length: {len(node.text)}")
        print(chunk_text)

    # =====================================================
    # EMBEDDINGS + INDEXING
    # =====================================================

    print("\n[8] Generating embeddings and indexing...")

    embed_start = time.time()

    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    embed_end = time.time()

    print("[OK] Embeddings generated")

    print(
        f"[INFO] Embedding time: "
        f"{embed_end - embed_start:.2f}s"
    )

    total_time = time.time() - start_time

    print("\n" + "=" * 50)
    print("INGESTION FINISHED")
    print("=" * 50)

    print(f"[TOTAL TIME] {total_time:.2f}s")
    print(f"[TOTAL CHUNKS] {len(nodes)}")