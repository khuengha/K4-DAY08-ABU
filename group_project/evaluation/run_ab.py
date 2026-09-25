"""Chạy A/B evaluation: Config A (dense-only) vs Config B (hybrid + RRF).

Dùng chung golden dataset, generator (gpt-4o-mini qua task10.call_llm),
SYSTEM_PROMPT và top_k; chỉ khác retrieval strategy.

Chạy:
    python -m group_project.evaluation.run_ab
Kết quả JSON: group_project/evaluation/ab_results.json
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.task10_generation import SYSTEM_PROMPT, call_llm, format_context, reorder_for_llm
from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank_rrf

load_dotenv()

EVAL_DIR = Path(__file__).parent
TOP_K = 5


def retrieve_dense(query: str, top_k: int = TOP_K) -> list[dict]:
    """Config A: chỉ semantic search."""
    return semantic_search(query, top_k=top_k)


def retrieve_hybrid(query: str, top_k: int = TOP_K) -> list[dict]:
    """Config B: dense + BM25 fusion bằng RRF."""
    dense = semantic_search(query, top_k=top_k * 2)
    sparse = lexical_search(query, top_k=top_k * 2)
    return rerank_rrf([dense, sparse], top_k=top_k)


def build_sample(case: dict, retriever) -> dict:
    """Retriever -> generator -> một sample cho ragas."""
    chunks = retriever(case["question"], top_k=TOP_K)
    context = format_context(reorder_for_llm(chunks)) if chunks else ""
    if context:
        answer = call_llm(
            SYSTEM_PROMPT,
            f"Context:\n{context}\n\nQuestion: {case['question']}",
        )
    else:
        answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
    return {
        "user_input": case["question"],
        "retrieved_contexts": [chunk["content"] for chunk in chunks],
        "response": answer,
        "reference": case["expected_answer"],
        "reference_contexts": [case["expected_context"]],
    }


def main() -> None:
    cases = json.loads(
        (EVAL_DIR / "golden_dataset.json").read_text(encoding="utf-8")
    )
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    report: dict[str, dict] = {}
    for name, retriever in (("A_dense_only", retrieve_dense), ("B_hybrid_rrf", retrieve_hybrid)):
        samples = []
        for case in cases:
            try:
                samples.append(build_sample(case, retriever))
            except Exception as exc:
                print(f"[{name}] FAIL {case['question'][:40]}: {exc}")
        (EVAL_DIR / f"ab_samples_{name}.json").write_text(
            json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result = evaluate(
            EvaluationDataset.from_list(samples),
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=llm,
            embeddings=embeddings,
        )
        report[name] = {
            "scores": result.scores,
            "overall": {k: v for k, v in result._repr_dict.items() if isinstance(v, float)},
        }
        print(name, result._repr_dict)

    (EVAL_DIR / "ab_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Saved to", EVAL_DIR / "ab_results.json")


if __name__ == "__main__":
    main()
