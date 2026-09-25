"""
Task 8 — PageIndex vectorless fallback.

Pipeline:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Convert Markdown sang PDF tạm.
    3. Upload PDF lên PageIndex.
    4. Cache source -> document ID.
    5. Query PageIndex và lấy retrieval result.
    6. Parse thành SearchResult có retrieval_method="pageindex".

PageIndex là dịch vụ ngoài nên mọi HTTP request đều có timeout.
Lỗi provider được chuyển thành kết quả rỗng để pipeline không crash.
"""

import json
import os
import tempfile
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from fpdf import FPDF
from pageindex import PageIndexAPIError


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")

STANDARDIZED_DIR = (
    Path(__file__).parent.parent
    / "data"
    / "standardized"
)

CACHE_FILE = (
    Path(__file__).parent.parent
    / "pageindex_cache.json"
)

PAGEINDEX_BASE_URL = "https://api.pageindex.ai"

REQUEST_TIMEOUT = 30
RETRIEVAL_POLL_INTERVAL = 2
RETRIEVAL_MAX_WAIT = 60


def _headers() -> dict:
    return {
        "api_key": PAGEINDEX_API_KEY,
    }


def _load_cache() -> dict:
    """Đọc mapping source -> PageIndex document ID."""
    if not CACHE_FILE.exists():
        return {}

    try:
        data = json.loads(
            CACHE_FILE.read_text(encoding="utf-8")
        )

        if isinstance(data, dict):
            return data

    except (OSError, json.JSONDecodeError):
        pass

    return {}


def _save_cache(cache: dict) -> None:
    """Lưu mapping source -> document ID."""
    CACHE_FILE.write_text(
        json.dumps(
            cache,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _markdown_to_pdf(markdown_path: Path) -> Path:
    """
    Convert Markdown sang PDF tạm để PageIndex xử lý.

    PageIndex SDK 0.2.8 yêu cầu file PDF cho submit_document().
    """

    markdown = markdown_path.read_text(
        encoding="utf-8"
    )

    pdf = FPDF()
    pdf.set_auto_page_break(
        auto=True,
        margin=15,
    )

    pdf.add_page()

    # Arial Unicode không có sẵn mặc định trong fpdf2.
    # Vì corpus có thể chứa Unicode tiếng Việt,
    # dùng font DejaVu Sans nếu hệ thống có.
    font_candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    ]

    font_path = next(
        (
            path
            for path in font_candidates
            if path.exists()
        ),
        None,
    )

    if font_path:
        pdf.add_font(
            "UnicodeFont",
            "",
            str(font_path),
        )
        pdf.set_font(
            "UnicodeFont",
            size=10,
        )
    else:
        # Fallback cho trường hợp không tìm thấy font Unicode.
        pdf.set_font(
            "Helvetica",
            size=10,
        )

    for line in markdown.splitlines():
        line = line.strip()

        if not line:
            pdf.ln(3)
            continue

        # Markdown heading
        while line.startswith("#"):
            line = line[1:]

        line = line.strip()

        if not line:
            continue

        # Markdown emphasis
        line = line.replace("**", "")
        line = line.replace("__", "")

        try:
            pdf.multi_cell(
                0,
                6,
                line,
            )
        except Exception:
            # Bỏ qua dòng không thể encode bằng font hiện tại.
            continue

    temp = tempfile.NamedTemporaryFile(
        suffix=".pdf",
        prefix="pageindex_",
        delete=False,
    )

    temp.close()

    pdf.output(temp.name)

    return Path(temp.name)


def _submit_document(pdf_path: Path) -> str:
    """Upload PDF và trả về doc_id."""

    response = requests.post(
        f"{PAGEINDEX_BASE_URL}/doc/",
        headers=_headers(),
        files={
            "file": (
                pdf_path.name,
                pdf_path.open("rb"),
                "application/pdf",
            )
        },
        data={
            "if_retrieval": "true",
        },
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    payload = response.json()

    doc_id = payload.get("doc_id")

    if not doc_id:
        raise PageIndexAPIError(
            "PageIndex response không chứa doc_id."
        )

    return str(doc_id)


def upload_documents() -> None:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""

    if not PAGEINDEX_API_KEY:
        print(
            "PAGEINDEX_API_KEY chưa được cấu hình. "
            "Bỏ qua PageIndex upload."
        )
        return

    markdown_files = sorted(
        STANDARDIZED_DIR.rglob("*.md")
    )

    if not markdown_files:
        print(
            "Không tìm thấy tài liệu trong "
            "data/standardized/."
        )
        return

    cache = _load_cache()

    uploaded = 0
    cached = 0
    failed = 0

    for markdown_path in markdown_files:

        source = markdown_path.relative_to(
            STANDARDIZED_DIR
        ).as_posix()

        if source in cache:
            print(
                f"[CACHE] {source} "
                f"-> {cache[source]}"
            )
            cached += 1
            continue

        pdf_path = None

        try:
            print(f"[UPLOAD] {source}")

            pdf_path = _markdown_to_pdf(
                markdown_path
            )

            doc_id = _submit_document(
                pdf_path
            )

            cache[source] = doc_id
            _save_cache(cache)

            print(
                f"[OK] {source} "
                f"-> {doc_id}"
            )

            uploaded += 1

        except requests.RequestException as error:
            print(
                f"[ERROR] Upload failed for "
                f"{source}: {error}"
            )
            failed += 1

        except Exception as error:
            print(
                f"[ERROR] {source}: {error}"
            )
            failed += 1

        finally:
            if pdf_path is not None:
                try:
                    pdf_path.unlink(
                        missing_ok=True
                    )
                except OSError:
                    pass

    print(
        f"\nPageIndex upload completed: "
        f"uploaded={uploaded}, "
        f"cached={cached}, "
        f"failed={failed}"
    )


def _submit_query(
    doc_id: str,
    query: str,
) -> str:
    """Submit retrieval query và trả về retrieval_id."""

    response = requests.post(
        f"{PAGEINDEX_BASE_URL}/retrieval/",
        headers=_headers(),
        json={
            "doc_id": doc_id,
            "query": query,
            "thinking": False,
        },
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    payload = response.json()

    retrieval_id = payload.get(
        "retrieval_id"
    )

    if not retrieval_id:
        raise PageIndexAPIError(
            "PageIndex response không chứa "
            "retrieval_id."
        )

    return str(retrieval_id)


def _get_retrieval(
    retrieval_id: str,
) -> dict:
    """Lấy kết quả retrieval từ PageIndex."""

    response = requests.get(
        f"{PAGEINDEX_BASE_URL}/retrieval/{retrieval_id}/",
        headers=_headers(),
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    return response.json()


def _wait_for_retrieval(
    retrieval_id: str,
) -> dict:
    """
    Poll retrieval cho tới khi completed
    hoặc timeout.
    """

    start = time.monotonic()

    while (
        time.monotonic() - start
        < RETRIEVAL_MAX_WAIT
    ):
        result = _get_retrieval(
            retrieval_id
        )

        status = str(
            result.get(
                "status",
                ""
            )
        ).lower()

        if status in {
            "completed",
            "complete",
            "success",
            "succeeded",
        }:
            return result

        if status in {
            "failed",
            "error",
        }:
            raise PageIndexAPIError(
                f"PageIndex retrieval failed: "
                f"{result}"
            )

        time.sleep(
            RETRIEVAL_POLL_INTERVAL
        )

    raise TimeoutError(
        "PageIndex retrieval timeout."
    )


def _extract_results(
    payload: dict,
) -> list[dict]:
    """
    Lấy danh sách retrieved nodes từ response.
    """

    retrieved_nodes = payload.get("retrieved_nodes")

    if isinstance(retrieved_nodes, list):
        return retrieved_nodes

    candidates = [
        payload.get("results"),
        payload.get("retrieval_results"),
        payload.get("data"),
        payload.get("nodes"),
    ]

    for value in candidates:
        if isinstance(value, list):
            return value

        if isinstance(value, dict):
            nested = (
                value.get("results")
                or value.get("nodes")
                or value.get("items")
            )

            if isinstance(nested, list):
                return nested

    return []


def _parse_result(
    item: dict,
    rank: int,
    source: str,
    doc_id: str,
) -> dict | None:
    """Convert một PageIndex node thành SearchResult."""

    if not isinstance(item, dict):
        return None

    content = (
        item.get("content")
        or item.get("text")
        or item.get("snippet")
        or item.get("node_content")
    )

    if not content:
        return None

    item_id = (
        item.get("id")
        or item.get("node_id")
        or item.get("page_id")
        or f"pageindex-{doc_id}-{rank}"
    )

    raw_score = item.get("score")

    if raw_score is None:
        # Nếu PageIndex không trả score,
        # dùng score giảm dần theo rank.
        score = 1.0 / rank
    else:
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 1.0 / rank

    metadata = {
        "source": source,
        "title": item.get(
            "title",
            source,
        ),
        "doc_type": (
            "legal"
            if "legal" in Path(source).parts
            else "news"
        ),
        "url": item.get("url"),
        "pageindex_doc_id": doc_id,
    }

    if "page_index" in item:
        metadata["page_index"] = item[
            "page_index"
        ]

    return {
        "id": str(item_id),
        "content": str(content),
        "score": score,
        "metadata": metadata,
        "retrieval_method": "pageindex",
    }


def pageindex_search(
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """Trả về PageIndex SearchResult."""

    if not query or not query.strip():
        return []

    if top_k <= 0:
        return []

    if not PAGEINDEX_API_KEY:
        return []

    cache = _load_cache()

    if not cache:
        return []

    results = []

    for source, doc_id in cache.items():

        try:
            retrieval_id = _submit_query(
                doc_id,
                query.strip(),
            )

            payload = _wait_for_retrieval(
                retrieval_id
            )

            items = _extract_results(
                payload
            )

            for rank, item in enumerate(
                items,
                start=1,
            ):
                result = _parse_result(
                    item,
                    rank,
                    source,
                    doc_id,
                )

                if result is not None:
                    results.append(result)

        except (
            requests.RequestException,
            TimeoutError,
            PageIndexAPIError,
        ) as error:
            print(
                f"[PageIndex] {source}: "
                f"{error}"
            )
            continue

        except Exception as error:
            print(
                f"[PageIndex] Parse error "
                f"for {source}: {error}"
            )
            continue

    # Nếu PageIndex trả score thì sort theo score.
    # Nếu không trả score, score đã được gán
    # theo rank của từng document.
    results.sort(
        key=lambda item: (
            -item["score"],
            item["id"],
        )
    )

    unique_results = []
    seen_ids = set()

    for result in results:
        if result["id"] in seen_ids:
            continue

        seen_ids.add(result["id"])
        unique_results.append(result)

        if len(unique_results) >= top_k:
            break

    return unique_results


if __name__ == "__main__":
    upload_documents()