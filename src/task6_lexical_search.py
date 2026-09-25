"""Task 6 — BM25+ over the exact Chroma chunks used by Task 5.

No embedding calls. Set CORPUS explicitly for offline experiments/tests;
None (the default) reads the Task 4 collection on each search.
"""

import re
import unicodedata
from functools import lru_cache

from rank_bm25 import BM25Plus

from .contracts import validate_document

CORPUS: list[dict] | None = None


def tokenize(text: str) -> list[str]:
    """Case-insensitive Unicode words/numbers, retaining Vietnamese accents."""
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFC", text).casefold())


def get_collection():
    """Lazy import keeps in-memory BM25 independent of embedding providers."""
    from .task4_chunking_indexing import get_collection as shared_collection

    return shared_collection()


def load_corpus() -> list[dict]:
    """Read stored IDs, text and metadata, without loading embedding vectors."""
    response = get_collection().get(include=["documents", "metadatas"])
    ids = response["ids"]
    documents = response.get("documents")
    metadatas = response.get("metadatas")
    if not ids:
        return []
    if documents is None or metadatas is None or not (len(ids) == len(documents) == len(metadatas)):
        raise ValueError("Chroma returned incomplete chunk documents/metadata")
    corpus = []
    for item_id, content, metadata in zip(ids, documents, metadatas):
        # Chroma may omit a nullable field; expose the common contract again.
        normalized_metadata = dict(metadata or {})
        normalized_metadata.setdefault("url", None)
        corpus.append({"id": item_id, "content": content, "metadata": normalized_metadata})
    return corpus


@lru_cache(maxsize=1)
def _build_index(tokenized: tuple[tuple[str, ...], ...]) -> BM25Plus | None:
    if not tokenized or not any(tokenized):
        return None
    # Positive IDF also works for very small corpora where Okapi IDF can be 0.
    return BM25Plus(tokenized, k1=1.5, b=0.75, delta=1.0)


def build_bm25_index(corpus: list[dict]) -> BM25Plus | None:
    """Cache the latest token snapshot; content edits invalidate it automatically."""
    return _build_index(tuple(tuple(tokenize(item["content"])) for item in corpus))


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Return unique SearchResults sorted by score, then ID to resolve ties."""
    if not query or top_k <= 0:
        return []
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    source = CORPUS if CORPUS is not None else load_corpus()
    by_id = {}
    for item in source:
        validate_document(item, require_chunk=True)
        if item["id"] in by_id and item != by_id[item["id"]]:
            raise ValueError(f"Conflicting chunks for ID: {item['id']}")
        by_id[item["id"]] = item
    corpus = sorted(by_id.values(), key=lambda item: item["id"])
    index = build_bm25_index(corpus)
    if index is None:
        return []
    scores = index.get_scores(query_tokens)
    results = []
    for position, item in enumerate(corpus):
        # BM25+ adds a baseline even to non-matches: require actual term overlap.
        if not any(token in index.doc_freqs[position] for token in query_tokens):
            continue
        results.append({
            "id": item["id"],
            "content": item["content"],
            "score": float(scores[position]),
            "metadata": dict(item["metadata"]),
            "retrieval_method": "bm25",
        })
    results.sort(key=lambda item: (-item["score"], item["id"]))
    return results[:top_k]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Search Task 4 chunks using BM25+")
    parser.add_argument("query", nargs="?", default="cách lên đồ cho Valhein")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    matches = lexical_search(args.query, top_k=args.top_k)
    if not matches:
        print("Không có chunk khớp. Kiểm tra từ khóa và dữ liệu đã index ở Task 4.")
    for result in matches:
        print(f"\n{result['id']} | score={result['score']:.4f}")
        print(f"Source: {result['metadata']['source']}")
        print(result["content"])
