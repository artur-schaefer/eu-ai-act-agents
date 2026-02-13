"""Tests for src/tools.py — tool schema validation and dispatch."""

import json
from src.tools import TOOL_SCHEMAS, TOOL_DISPATCH


def test_tool_schemas_valid():
    """All tool schemas have required fields."""
    for tool in TOOL_SCHEMAS:
        assert tool["type"] == "function"
        fn = tool["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"
        assert "required" in fn["parameters"]


def test_dispatch_covers_all_schemas():
    """Every schema has a matching dispatch entry."""
    schema_names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    dispatch_names = set(TOOL_DISPATCH.keys())
    assert schema_names == dispatch_names


def test_tool_schema_names():
    """Check expected tool names exist."""
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    expected = {"search_regulation", "get_article", "get_recital", "get_annex", "get_articles_for_chapter"}
    assert expected == names
