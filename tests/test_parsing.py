"""Tests for src/parsing.py — tests both DE (default) and EN."""

import pytest
from src.parsing import parse_eu_ai_act, Provision


@pytest.fixture(params=["de", "en"])
def provisions(request):
    return parse_eu_ai_act(lang=request.param), request.param


def test_parse_counts(provisions):
    provs, lang = provisions
    articles = [p for p in provs if p.type == "article"]
    recitals = [p for p in provs if p.type == "recital"]
    annexes = [p for p in provs if p.type == "annex"]

    assert len(articles) == 113
    assert len(recitals) == 180
    assert len(annexes) == 13


def test_article_structure(provisions):
    provs, lang = provisions
    art5 = next(p for p in provs if p.id == "art_5")

    assert art5.type == "article"
    assert art5.number == "5"
    assert art5.chapter == "II"
    assert len(art5.paragraphs) > 0
    assert len(art5.text) > 1000
    assert art5.lang == lang

    if lang == "de":
        assert "Verbotene" in art5.title
    else:
        assert "Prohibited" in art5.title


def test_article_chapter_mapping(provisions):
    provs, lang = provisions
    articles = [p for p in provs if p.type == "article"]

    for art in articles:
        assert art.chapter, f"{art.id} has no chapter"


def test_recital_text_clean(provisions):
    provs, lang = provisions
    rct1 = next(p for p in provs if p.id == "rct_1")

    assert not rct1.text.startswith("(1)")
    assert len(rct1.text) > 100

    if lang == "de":
        assert rct1.title.startswith("Erwägungsgrund")
    else:
        assert rct1.title.startswith("Recital")


def test_annex_content(provisions):
    provs, lang = provisions
    anx3 = next(p for p in provs if p.id == "anx_III")
    assert anx3.number == "III"

    if lang == "de":
        assert "Hochrisiko" in anx3.text or "hochrisiko" in anx3.text.lower()
    else:
        assert "high-risk" in anx3.text.lower() or "High-risk" in anx3.text
