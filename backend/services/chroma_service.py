import chromadb


def get_chroma_collection():

    chroma_client = chromadb.HttpClient(
        host="localhost",
        port=8000
    )

    collection = chroma_client.get_or_create_collection(
        name="shared_knowledge_base"
    )

    return collection