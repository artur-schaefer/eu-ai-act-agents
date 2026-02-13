"""Tests for src/chunking.py."""

import pytest
from src.parsing import parse_eu_ai_act
from src.chunking import chunk_provisions, count_tokens, MAX_TOKENS


@pytest.fixture(params=["de", "en"])
def chunks(request):
    provisions = parse_eu_ai_act(lang=request.param)
    return chunk_provisions(provisions)


def test_chunk_count(chunks):
    assert len(chunks) > 300


def test_chunk_metadata(chunks):
    for c in chunks:
        assert "id" in c
        assert "text" in c
        assert "metadata" in c
        meta = c["metadata"]
        assert meta["type"] in ("article", "recital", "annex")
        assert meta["number"]
        assert meta["title"]


def test_most_chunks_under_limit(chunks):
    tokens = [count_tokens(c["text"]) for c in chunks]
    over = [t for t in tokens if t > MAX_TOKENS]
    assert len(over) < 5, f"{len(over)} chunks over {MAX_TOKENS} tokens"
