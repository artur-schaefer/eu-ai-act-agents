"""OpenAI embeddings + ChromaDB vector store."""

import chromadb
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LANG = "de"
CHROMA_PATH = "chroma_db"
BATCH_SIZE = 100


def _collection_name(lang: str = DEFAULT_LANG) -> str:
    return f"eu_ai_act_{lang}"


def get_openai_client() -> OpenAI:
    from dotenv import load_dotenv
    load_dotenv()
    return OpenAI()


def embed_texts(texts: list[str], client: OpenAI | None = None) -> list[list[float]]:
    """Embed a list of texts using OpenAI."""
    if client is None:
        client = get_openai_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [e.embedding for e in response.data]


def build_index(chunks: list[dict], chroma_path: str = CHROMA_PATH,
                client: OpenAI | None = None, lang: str = DEFAULT_LANG) -> chromadb.Collection:
    """Build ChromaDB index from chunks. Embeds in batches."""
    if client is None:
        client = get_openai_client()

    chroma = chromadb.PersistentClient(path=chroma_path)
    col_name = _collection_name(lang)

    # Delete existing collection if present
    try:
        chroma.delete_collection(col_name)
    except Exception:
        pass

    collection = chroma.create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Process in batches
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        embeddings = embed_texts(texts, client)

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"  Indexed {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")

    return collection


def get_collection(chroma_path: str = CHROMA_PATH, lang: str = DEFAULT_LANG) -> chromadb.Collection:
    """Get existing ChromaDB collection."""
    chroma = chromadb.PersistentClient(path=chroma_path)
    return chroma.get_collection(_collection_name(lang))


def search(query: str, n_results: int = 8, where: dict | None = None,
           collection: chromadb.Collection | None = None,
           client: OpenAI | None = None) -> list[dict]:
    """Semantic search over the index."""
    if collection is None:
        collection = get_collection()
    if client is None:
        client = get_openai_client()

    query_embedding = embed_texts([query], client)[0]

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return hits


def lookup_by_id(provision_id: str, collection: chromadb.Collection | None = None) -> list[dict]:
    """Direct lookup of all chunks for a provision ID."""
    if collection is None:
        collection = get_collection()

    results = collection.get(
        where={"provision_id": provision_id},
        include=["documents", "metadatas"],
    )

    hits = []
    for i in range(len(results["ids"])):
        hits.append({
            "id": results["ids"][i],
            "text": results["documents"][i],
            "metadata": results["metadatas"][i],
        })
    return hits
