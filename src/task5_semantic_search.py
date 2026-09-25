"""
Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output phải theo SearchResult, sort giảm dần và không quá top_k.
"""

from .task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""

    # Validate input
    if not query or not query.strip():
        return []

    if top_k <= 0:
        return []

    # Embed query bằng chính embedding model của Task 4
    query_vector = embed_texts([query])[0]

    # Query ChromaDB
    response = get_collection().query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    results = []
    seen_ids = set()

    # Chroma trả kết quả theo dạng:
    # ids[0], documents[0], metadatas[0], distances[0]
    for item_id, content, metadata, distance in zip(
        response["ids"][0],
        response["documents"][0],
        response["metadatas"][0],
        response["distances"][0],
    ):
        # Contract: SearchResult không được trùng ID
        if item_id in seen_ids:
            continue

        seen_ids.add(item_id)

        # Chroma đang dùng cosine distance.
        # Chuyển distance -> similarity.
        score = max(0.0, 1.0 - distance)

        results.append(
            {
                "id": item_id,
                "content": content,
                "score": float(score),
                "metadata": metadata,
                "retrieval_method": "dense",
            }
        )

    # Contract: sort score giảm dần
    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    # Contract: không vượt quá top_k
    return results[:top_k]


if __name__ == "__main__":
    results = semantic_search(
        "cách lên đồ cho Valhein",
        top_k=3,
    )

    for index, result in enumerate(results, start=1):
        print(f"\n--- Result {index} ---")
        print(f"ID: {result['id']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Metadata: {result['metadata']}")
        print(f"Content: {result['content'][:300]}...")