import time

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
)

from llama_index.core.vector_stores import (
    MetadataFilters,
    ExactMatchFilter,
)

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from llama_index.vector_stores.chroma import ChromaVectorStore

from services.chroma_service import get_chroma_collection


def ask_question(question: str, project: str):

    start_time = time.time()

    print("\n" + "=" * 50)
    print("START QUERY")
    print("=" * 50)

    print(f"[QUESTION] {question}")
    print(f"[PROJECT] {project}")

    # =====================================================
    # Chroma
    # =====================================================

    print("\n[1] Connecting to ChromaDB...")

    collection = get_chroma_collection()

    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    print("[OK] Chroma connected")

    # =====================================================
    # Embedding model
    # =====================================================

    print("\n[2] Loading embedding model...")

    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434"
    )

    print("[OK] Embedding model loaded")

    # =====================================================
    # LLM
    # =====================================================

    print("\n[3] Loading LLM...")

    llm = Ollama(
        model="llama3.2:3b",
        base_url="http://localhost:11434",
        request_timeout=300.0,
        temperature=0.1,
    )

    print("[OK] LLM loaded")

    # =====================================================
    # Vector index
    # =====================================================

    print("\n[4] Loading vector index...")

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    print("[OK] Index loaded")

    # =====================================================
    # Metadata filter
    # =====================================================

    print(f"\n[5] Applying project filter: {project}")

    filters = MetadataFilters(
        filters=[
            ExactMatchFilter(
                key="project",
                value=project
            )
        ]
    )

    # =====================================================
    # Retriever
    # =====================================================

    print("\n[6] Creating retriever...")

    retriever = index.as_retriever(
        similarity_top_k=5,
        filters=filters,
    )

    print("[OK] Retriever created")

    # =====================================================
    # Retrieve chunks
    # =====================================================

    print("\n[7] Retrieving chunks...")

    nodes = retriever.retrieve(question)

    print(f"[OK] Retrieved chunks: {len(nodes)}")

    for i, node in enumerate(nodes):

        print("\n" + "-" * 50)
        print(f"CHUNK {i + 1}")
        print("-" * 50)

        print(f"Score: {node.score}")

        filename = node.metadata.get(
            "filename",
            "Unknown"
        )

        print(f"File: {filename}")

        preview = (
            node.text[:500]
            .replace("\n", " ")
            .strip()
        )

        print(preview)

    # =====================================================
    # Query engine
    # =====================================================

    print("\n[8] Creating query engine...")

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=5,
        filters=filters,
    )

    print("[OK] Query engine created")

    # =====================================================
    # Generate response
    # =====================================================

    print("\n[9] Generating response...")

    response = query_engine.query(question)

    print("[OK] Response generated")

    print("\n[ANSWER]")
    print(str(response))

    # =====================================================
    # Final stats
    # =====================================================

    total_time = time.time() - start_time

    print("\n" + "=" * 50)
    print("QUERY FINISHED")
    print("=" * 50)

    print(f"[TOTAL TIME] {total_time:.2f}s")

    return {
        "answer": str(response),
        "sources": list(
            set(
                [
                    node.metadata.get(
                        "filename",
                        "Unknown"
                    )
                    for node in nodes
                ]
            )
        )
    }