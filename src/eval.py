"""Evaluation helpers for the EU AI Act agents."""

import json
from src.embeddings import search, get_collection, get_openai_client


def retrieval_recall(
    queries: list[dict],
    chroma_path: str = "chroma_db",
    k: int = 8,
    lang: str = "de",
) -> dict:
    """Measure retrieval recall — for each query, check if expected provisions appear in top-k.

    queries: [{"query": str, "expected_ids": [str], "label": str}]
    Returns: {"recall": float, "details": [...]}
    """
    collection = get_collection(chroma_path, lang=lang)
    client = get_openai_client()

    details = []
    total_expected = 0
    total_found = 0

    for q in queries:
        results = search(q["query"], n_results=k, collection=collection, client=client)
        retrieved_ids = set()
        for r in results:
            retrieved_ids.add(r["metadata"]["provision_id"])

        expected = set(q["expected_ids"])
        found = expected & retrieved_ids
        total_expected += len(expected)
        total_found += len(found)

        details.append({
            "label": q.get("label", q["query"]),
            "expected": sorted(expected),
            "found": sorted(found),
            "missed": sorted(expected - found),
            "recall": len(found) / len(expected) if expected else 1.0,
        })

    return {
        "recall": total_found / total_expected if total_expected else 1.0,
        "details": details,
    }


def classification_accuracy(
    test_cases: list[dict],
    classify_fn,
) -> dict:
    """Measure classification accuracy.

    test_cases: [{"description": str, "expected_risk": str, "label": str}]
    classify_fn: callable that takes description and returns risk_level string
    Returns: {"accuracy": float, "details": [...]}
    """
    details = []
    correct = 0

    for case in test_cases:
        predicted = classify_fn(case["description"])
        is_correct = predicted.lower() == case["expected_risk"].lower()
        if is_correct:
            correct += 1

        details.append({
            "label": case.get("label", case["description"][:50]),
            "expected": case["expected_risk"],
            "predicted": predicted,
            "correct": is_correct,
        })

    return {
        "accuracy": correct / len(test_cases) if test_cases else 1.0,
        "details": details,
    }


def faithfulness_check(
    qa_pairs: list[dict],
    chroma_path: str = "chroma_db",
    lang: str = "de",
) -> dict:
    """Check if Q&A answers cite articles that actually exist in the index.

    qa_pairs: [{"answer": str, "cited_articles": [str]}]  # cited_articles like ["art_5", "art_50"]
    Returns: {"faithfulness": float, "details": [...]}
    """
    collection = get_collection(chroma_path, lang=lang)

    details = []
    total_cited = 0
    total_verified = 0

    for pair in qa_pairs:
        cited = pair["cited_articles"]
        verified = []
        for art_id in cited:
            results = collection.get(
                where={"provision_id": art_id},
                include=["metadatas"],
            )
            if results["ids"]:
                verified.append(art_id)

        total_cited += len(cited)
        total_verified += len(verified)

        details.append({
            "cited": cited,
            "verified": verified,
            "unverified": [a for a in cited if a not in verified],
            "faithfulness": len(verified) / len(cited) if cited else 1.0,
        })

    return {
        "faithfulness": total_verified / total_cited if total_cited else 1.0,
        "details": details,
    }
