"""Tool schemas and dispatch for AI Act agents."""

import json
from src.embeddings import search, lookup_by_id, get_collection, get_openai_client

# Shared state
_collection = None
_openai_client = None


def init_tools(chroma_path: str = "chroma_db", lang: str = "de"):
    """Initialize shared tool dependencies."""
    global _collection, _openai_client
    _openai_client = get_openai_client()
    _collection = get_collection(chroma_path, lang=lang)


def _get_deps():
    if _collection is None or _openai_client is None:
        raise RuntimeError("Call init_tools() before using agent tools")
    return _collection, _openai_client


# --- Tool implementations ---

def search_regulation(query: str, n_results: int = 8, type_filter: str | None = None) -> str:
    """Semantic search over the EU AI Act."""
    collection, client = _get_deps()
    where = {"type": type_filter} if type_filter else None
    results = search(query, n_results=n_results, where=where, collection=collection, client=client)
    return json.dumps([{
        "id": r["id"],
        "title": r["metadata"]["title"],
        "chapter": r["metadata"].get("chapter", ""),
        "text": r["text"][:1500],
    } for r in results], indent=2)


def get_article(number: str) -> str:
    """Get full text of a specific article by number."""
    collection, _ = _get_deps()
    results = lookup_by_id(f"art_{number}", collection=collection)
    if not results:
        return json.dumps({"error": f"Article {number} not found"})
    # Combine all chunks for this article
    texts = sorted(results, key=lambda r: r["metadata"].get("chunk_idx", 0))
    combined = "\n\n".join(r["text"] for r in texts)
    return json.dumps({"article": number, "title": texts[0]["metadata"]["title"], "text": combined})


def get_recital(number: str) -> str:
    """Get full text of a specific recital by number."""
    collection, _ = _get_deps()
    results = lookup_by_id(f"rct_{number}", collection=collection)
    if not results:
        return json.dumps({"error": f"Recital {number} not found"})
    texts = sorted(results, key=lambda r: r["metadata"].get("chunk_idx", 0))
    combined = "\n\n".join(r["text"] for r in texts)
    return json.dumps({"recital": number, "text": combined})


def get_annex(number: str) -> str:
    """Get full text of a specific annex by Roman numeral."""
    collection, _ = _get_deps()
    results = lookup_by_id(f"anx_{number}", collection=collection)
    if not results:
        return json.dumps({"error": f"Annex {number} not found"})
    texts = sorted(results, key=lambda r: r["metadata"].get("chunk_idx", 0))
    combined = "\n\n".join(r["text"] for r in texts)
    return json.dumps({"annex": number, "title": texts[0]["metadata"]["title"], "text": combined})


def get_articles_for_chapter(chapter: str) -> str:
    """List all articles in a chapter."""
    collection, _ = _get_deps()
    results = collection.get(
        where={"$and": [{"type": "article"}, {"chapter": chapter}]},
        include=["metadatas"],
    )
    articles = []
    seen = set()
    for meta in results["metadatas"]:
        num = meta["number"]
        if num not in seen:
            seen.add(num)
            articles.append({"number": num, "title": meta["title"]})
    articles.sort(key=lambda a: int(a["number"]))
    return json.dumps(articles, indent=2)


# --- Tool schemas for OpenAI function calling ---

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_regulation",
            "description": "Semantic search over the EU AI Act. Returns the most relevant provisions for a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "n_results": {"type": "integer", "description": "Number of results (default 8)", "default": 8},
                    "type_filter": {
                        "type": "string",
                        "enum": ["article", "recital", "annex"],
                        "description": "Filter by provision type (optional)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_article",
            "description": "Get the full text of a specific article by its number (e.g. '5' for Article 5).",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {"type": "string", "description": "Article number (e.g. '5', '50')"},
                },
                "required": ["number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recital",
            "description": "Get the full text of a specific recital by its number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {"type": "string", "description": "Recital number (e.g. '1', '47')"},
                },
                "required": ["number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_annex",
            "description": "Get the full text of a specific annex by Roman numeral (e.g. 'III' for Annex III).",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {"type": "string", "description": "Annex Roman numeral (e.g. 'I', 'III', 'VIII')"},
                },
                "required": ["number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_articles_for_chapter",
            "description": "List all articles in a given chapter of the EU AI Act.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chapter": {"type": "string", "description": "Chapter Roman numeral (e.g. 'I', 'III', 'IX')"},
                },
                "required": ["chapter"],
            },
        },
    },
]


# Dispatch map
TOOL_DISPATCH = {
    "search_regulation": lambda args: search_regulation(**args),
    "get_article": lambda args: get_article(**args),
    "get_recital": lambda args: get_recital(**args),
    "get_annex": lambda args: get_annex(**args),
    "get_articles_for_chapter": lambda args: get_articles_for_chapter(**args),
}
