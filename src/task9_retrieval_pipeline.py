"""
Task 9 — Retrieval pipeline hoàn chỉnh.

Luồng xử lý:
    1. Chạy semantic_search và lexical_search.
    2. Fuse hai danh sách bằng RRF đúng một lần.
    3. Lấy best cosine score gốc từ dense results.
    4. Nếu score dưới threshold, thử PageIndex fallback.
    5. Nếu fallback lỗi, trả hybrid results thay vì crash.

Không so sánh threshold với RRF score vì hai thang đo khác nhau.
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Trả về tối đa top_k kết quả hybrid, dense hoặc PageIndex.

    Khi tắt reranking, dùng dense results làm kết quả chính. PageIndex
    chỉ được thử khi cosine score tốt nhất thấp hơn score_threshold;
    nếu provider lỗi hoặc không có kết quả, giữ kết quả chính.
    """
    if not query or not query.strip() or top_k <= 0:
        return []

    dense = semantic_search(query, top_k=top_k * 2)
    sparse = lexical_search(query, top_k=top_k * 2)

    # Giữ cosine score gốc trước khi fuse, không dùng RRF score cho threshold.
    best_dense_score = max((item["score"] for item in dense), default=0.0)
    hybrid = (
        rerank_rrf([dense, sparse], top_k=top_k)
        if use_reranking else dense[:top_k]
    )

    if best_dense_score < score_threshold:
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback[:top_k]
        except Exception:
            # Fallback là tùy chọn: lỗi provider không làm mất kết quả đã có.
            pass

    return hybrid[:top_k]


if __name__ == "__main__":
    for result in retrieve("test query", top_k=3):
        print(result)
