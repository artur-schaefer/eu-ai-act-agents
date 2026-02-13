"""Chunk Provisions into sized pieces for embedding."""

import re
import tiktoken
from src.parsing import Provision


ENCODER = tiktoken.encoding_for_model("gpt-4o")
MAX_TOKENS = 800


def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))


def _make_chunk(provision: Provision, text: str, chunk_idx: int = 0) -> dict:
    """Create a chunk dict with metadata."""
    chunk_id = f"{provision.id}#{chunk_idx}" if chunk_idx > 0 else provision.id
    return {
        "id": chunk_id,
        "text": text,
        "metadata": {
            "provision_id": provision.id,
            "type": provision.type,
            "number": provision.number,
            "title": provision.title,
            "chapter": provision.chapter,
            "chapter_title": provision.chapter_title,
            "chunk_idx": chunk_idx,
        },
    }


def _split_by_sentences(provision: Provision, header: str, text: str) -> list[dict]:
    """Split text by sentences, grouping into chunks under MAX_TOKENS."""
    sentences = re.split(r"(?<=[.;])\s+", text)
    chunks = []
    current_text = header
    chunk_idx = 0

    for sent in sentences:
        candidate = current_text + sent + " "
        if count_tokens(candidate) > MAX_TOKENS and current_text != header:
            chunks.append(_make_chunk(provision, current_text.strip(), chunk_idx))
            chunk_idx += 1
            current_text = header + sent + " "
        else:
            current_text = candidate

    if current_text.strip() != header.strip():
        chunks.append(_make_chunk(provision, current_text.strip(), chunk_idx))

    return chunks


def _chunk_article(article: Provision) -> list[dict]:
    """Chunk an article. If it fits in MAX_TOKENS, keep as one chunk. Otherwise split by paragraph."""
    header = f"{article.title}\n\n"

    # Try as single chunk
    full = header + article.text
    if count_tokens(full) <= MAX_TOKENS:
        return [_make_chunk(article, full)]

    # Split by paragraph
    if not article.paragraphs:
        # No paragraph structure — split by sentences
        return _split_by_sentences(article, header, article.text)

    chunks = []
    current_text = header
    chunk_idx = 0

    for para in article.paragraphs:
        para_text = para["text"]
        candidate = current_text + para_text + "\n\n"

        if count_tokens(candidate) > MAX_TOKENS and current_text != header:
            chunks.append(_make_chunk(article, current_text.strip(), chunk_idx))
            chunk_idx += 1
            current_text = header + para_text + "\n\n"
        else:
            current_text = candidate

    if current_text.strip() != header.strip():
        chunks.append(_make_chunk(article, current_text.strip(), chunk_idx))

    return chunks


def _chunk_recital(recital: Provision) -> list[dict]:
    """Recitals are typically short enough for a single chunk."""
    header = f"{recital.title}\n\n"
    full = header + recital.text
    if count_tokens(full) <= MAX_TOKENS:
        return [_make_chunk(recital, full)]
    return _split_by_sentences(recital, header, recital.text)


def _chunk_annex(annex: Provision) -> list[dict]:
    """Chunk annexes. Split long annexes by sentences to stay under token limit."""
    header = f"{annex.title}\n\n"
    full = header + annex.text
    if count_tokens(full) <= MAX_TOKENS:
        return [_make_chunk(annex, full)]
    return _split_by_sentences(annex, header, annex.text)


def chunk_provisions(provisions: list[Provision]) -> list[dict]:
    """Chunk all provisions into embedding-ready pieces."""
    chunks = []
    for p in provisions:
        if p.type == "article":
            chunks.extend(_chunk_article(p))
        elif p.type == "recital":
            chunks.extend(_chunk_recital(p))
        elif p.type == "annex":
            chunks.extend(_chunk_annex(p))
    return chunks
