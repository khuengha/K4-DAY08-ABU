"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn.
    3. Embed chunks bằng một provider duy nhất.
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk theo docs/MODULE_CONTRACTS.md. ID ổn định để chạy lại
pipeline không tạo dữ liệu trùng. Task 5 dùng chung embed_texts().
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# Provider được chọn trong .env (sentence_transformers | openai | gemini).
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")

if EMBEDDING_PROVIDER == "openai":
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIM = 1536
else:
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_DIM = 1024

COLLECTION_NAME = "rag_documents"

_EMBEDDING_BATCH_SIZE = 128


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed danh sách text bằng provider đã cấu hình trong .env."""
    if not texts:
        return []

    if EMBEDDING_PROVIDER == "openai":
        from openai import OpenAI

        client = OpenAI()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + _EMBEDDING_BATCH_SIZE]
            for attempt in range(1, 4):
                try:
                    response = client.embeddings.create(
                        model=EMBEDDING_MODEL,
                        input=batch,
                        dimensions=EMBEDDING_DIM,
                    )
                    break
                except Exception as error:
                    if attempt == 3:
                        raise
                    print(f"Embedding retry {attempt}/3 after error: {error}")
                    time.sleep(3 * attempt)
            vectors.extend(item.embedding for item in response.data)
        return vectors

    if EMBEDDING_PROVIDER == "sentence_transformers":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)
        return model.encode(texts, batch_size=_EMBEDDING_BATCH_SIZE).tolist()

    raise ValueError(f"EMBEDDING_PROVIDER không hỗ trợ: {EMBEDDING_PROVIDER}")


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    documents = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        doc_type = "legal" if "legal" in path.parts else "news"
        documents.append(
            {
                "id": path.relative_to(STANDARDIZED_DIR).as_posix(),
                "content": content,
                "metadata": {
                    "source": path.name,
                    "title": path.stem,
                    "doc_type": doc_type,
                    "url": None,
                },
            }
        )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for document in documents:
        for index, text in enumerate(splitter.split_text(document["content"])):
            text = text.strip()
            if not text:
                continue
            chunks.append(
                {
                    "id": f"{document['id']}::chunk-{index}",
                    "content": text,
                    "metadata": {**document["metadata"], "chunk_index": index},
                }
            )
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk (embed theo batch)."""
    texts = [chunk["content"] for chunk in chunks]
    vectors = embed_texts(texts)
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    collection = get_collection()
    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")
    chunks = chunk_documents(documents)
    print(f"Chunked into {len(chunks)} chunks "
          f"(method={CHUNKING_METHOD}, size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks "
          f"into '{COLLECTION_NAME}' "
          f"(provider={EMBEDDING_PROVIDER}, model={EMBEDDING_MODEL})")


if __name__ == "__main__":
    run_pipeline()
