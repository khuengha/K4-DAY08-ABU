# Evaluation A/B — Dense-only vs Hybrid + RRF

## Thông tin chạy (reproducibility)

- **Ngày chạy:** 2026-09-25
- **Corpus commit:** `ee7193f` (branch `Khue`), corpus = 3 legal (Garena ToS, 2 công bố xử phạt Liên Quân) + 8 bài wiki Liên Quân, chunked bởi Task 4, index ChromaDB `chroma_db/`
- **Golden dataset:** `golden_dataset.json` — 18 case (10 legal, 8 wiki), mỗi case có `question` / `expected_answer` / `expected_context`, tất cả trích trực tiếp từ corpus
- **Generator:** gpt-4o-mini (OpenAI) qua `src/task10_generation.call_llm`, chung `SYSTEM_PROMPT`, temperature 0.3, top_p 0.9
- **Evaluator:** RAGAS 0.4.3 (faithfulness, answer_relevancy, context_recall, context_precision), judge gpt-4o-mini temperature 0, embedding text-embedding-3-small
- **top_k:** 5 cho cả hai cấu hình; Config B dùng `rerank_rrf` (k=60) trên dense top-10 + BM25 top-10
- **Script:** `group_project/evaluation/run_ab.py` — kết quả thô `ab_results.json`, per-case `ab_samples_*.json`
- **Config A** = `semantic_search(query, top_k=5)` (dense-only)
- **Config B** = dense top-10 + BM25 top-10 → `rerank_rrf` (hybrid + RRF); PageIndex fallback tắt trong phép thử để cô lập biến RRF

## Overall scores

| Metric | Config A (dense-only) | Config B (hybrid + RRF) | Delta B − A |
|---|---|---|---|
| Faithfulness | 0.870 | 0.873 | **+0.003** |
| Answer relevancy | 0.432 | 0.479 | **+0.047** |
| Context recall | 0.750 | 0.870 | **+0.120** |
| Context precision | 0.814 | 0.860 | **+0.047** |

## A/B comparison

- **Context recall +0.120 là lợi ích lớn nhất của RRF.** Ba case mà dense-only không lấy được bằng chứng (cày thuê, phù hiệu Rừng nguyên sinh, ngọc mở khóa) đều được BM25 cứu nhờ khớp từ khóa chính xác ("cày thuê", "Rừng nguyên sinh"). Điểm context_precision của B thấp hơn ở vài case nhưng do đưa thêm chunk wiki ít liên quan vào top-5, không mất bằng chứng.
- **Faithfulness gần như không đổi (+0.003)** vì generator và prompt giữ nguyên; chỉ số này phụ thuộc vào hành vi của gpt-4o-mini với context, không phụ thuộc retrieval. Case có context đúng hướng vẫn có thể bị judge chấm 0.5 khi model gộp quy định cũ (2023) và mới (2026) trong một câu.
- **Answer relevancy thấp ở cả hai cấu hình (~0.43–0.48)**: điểm này đo bằng cosine giữa câu trả lời và các câu hỏi được LLM sinh lại từ câu trả lời; các câu trả lời trích số liệu pháp lý ngắn, dùng đúng thuật ngữ nhưng câu sinh lại thường xoay quanh "khóa tài khoản" nên cosine thấp. Đây là đặc điểm của corpus tiếng Việt + câu trả lời cài tab, không phải lỗi hệ thống; cần đọc case thay vì chỉ nhìn điểm tổng.
- **Latency/cost:** Config B chạy 2 truy vấn retrieval (embedding + BM25) + RRF nên tốn thêm ~vài chục ms và không tốn thêm chi phí LLM; số token context trong B cao hơn đôi chút khi RRF kéo chunk wiki dài vào top-5. Không đo latency chính xác trong lần chạy này.

## Worst performers (3 case kém nhất)

### 1. Case 4 — "Quảng cáo hoặc sử dụng dịch vụ cày thuê bị phạt thế nào?"
- **Điểm:** A: recall 0.0, precision 0.0, faithfulness 0.0; B: recall 1.0, precision 0.37.
- **Failure stage:** retrieval (Config A) → được hybrid sửa.
- **Root cause:** câu hỏi dùng "cày thuê" là cụm từ khóa hiếm trong corpus (chỉ nằm trong 1 hàng bảng); dense-only không lấy được chunk chứa hàng bảng này. BM25 khớp chính xác và Config B trả lời đúng "1 năm tới vĩnh viễn". Điểm precision của B thấp vì RRF kéo kèm 4 chunk bảng xử phạt khác.
- **Verification:** chạy lại `retrieve("Quảng cáo hoặc sử dụng dịch vụ cày thuê bị phạt thế nào?")` và kiểm tra chunk chứa "Quảng cáo, cung cấp hoặc sử dụng dịch vụ cày thuê" nằm trong top-5; rồi chạy lại Config A để xác nhận vẫn miss.

### 2. Case 13 — "Phù hiệu Rừng nguyên sinh thích hợp với những vai trò nào?"
- **Điểm:** A: recall 0.0, precision 0.48; B: recall 1.0, precision 0.70.
- **Failure stage:** retrieval (Config A) → được hybrid sửa.
- **Root cause:** câu hỏi cần tương đồng ngữ nghĩa ("vai trò" ↔ "Hỗ trợ và Đỡ đòn") nhưng embedding gpt text-embedding-3-small xếp chunk một dòng của bảng phù hiệu thấp hơn các chunk tổng quan; BM25 khớp "Rừng nguyên sinh" giúp RRF đưa đúng chunk lên.
- **Verification:** chạy `semantic_search("Phù hiệu Rừng nguyên sinh thích hợp với những vai trò nào?")` và kiểm tra chunk "Rừng nguyên sinh: Thích hợp với Hỗ trợ và Đỡ đòn" có rank; đối chiếu với thứ hạng sau RRF.

### 3. Case 15 — "Liên Quân Mobile có còn hỗ trợ phần mềm giả lập không?"
- **Điểm:** A: recall 0.0, precision 0.0; B: recall 0.0 (judge), precision 0.25, faithfulness 0.0 (judge) — nhưng **đọc tay cho thấy B trả lời đúng** ("đã ngừng hỗ trợ từ 12/12/2023", citation Document 1 trùng với chunk chứa đúng câu này).
- **Failure stage:** data + evaluation (chỉ số sai, không phải hệ thống sai).
- **Root cause:** (a) Config A thật sự miss: chunk chứa câu "ngừng hỗ trợ phần mềm giả lập" chỉ dài 1 câu nằm giữa chunk dài khác nên embedding không thấy. (b) Config B lấy đúng chunk nhưng judge RAGAS chấm faithfulness/recall = 0 một cách giả âm: câu trả lời dùng "Liên Quân Mobile đã ngừng…" trong khi reference dùng "Không. Liên Quân…" và context tham chiếu là 1 câu rút gọn. Đây là nhiễu của judge LLM, đã xác minh bằng tay trong `ab_samples_B_hybrid_rrf.json`.
- **Verification:** chạy lại case này, in context đã đưa vào prompt và so khớp câu trả lời với chunk gốc `legal/lienquan-cac-hanh-vi-bi-xu-phat-va-thoi-han-khoa.md` (mục LƯU Ý số 5); xem `ab_samples_B_hybrid_rrf.json` case cuối.

## Recommendations

1. **Chọn Config B (hybrid + RRF) làm mặc định** — recall +12 điểm mà không đổi generator; chi phí thêm chỉ là một truy vấn BM25.
   *Cách kiểm tra:* chạy lại `python -m group_project.evaluation.run_ab` sau mỗi lần cập nhật corpus; chấp nhận khi context_recall của B ≥ A trên mọi lần chạy.
2. **Cải thiện chunking cho bảng xử phạt** — các hàng bảng đang bị tách thành chunk quá nhỏ, gây (a) precision thấp khi RRF hút cả bảng, (b) case "giả lập" miss ở Config A. Đề xuất gộp cả bảng "Hành vi bị xử phạt | Thời hạn khóa" thành một chunk duy nhất.
   *Cách kiểm tra:* re-index với chunker giữ bảng, chạy lại A/B, kỳ vọng context_precision của B tăng (những case 2, 4 hiện 0.37–0.70) và Config A không còn miss case 15.
3. **Thêm lệnh cấm so sánh quy định cũ/mới trong prompt generation** — hai case faithfulness 0.5 (dàn xếp trận, nạp lậu) là do model trộn khung phạt 2022–2023 với khung 2026 khi cả hai chunk cùng vào context. Đã có document title+date trong metadata (format_context), cần nhắc model "nếu nhiều văn bản cùng quy định, dùng văn bản mới nhất".
   *Cách kiểm tra:* chạy lại chỉ metric faithfulness trên 18 case với prompt mới; kỳ vọng 2 case faithfulness 0.5 lên ≥ 0.8 mà recall không đổi.
4. **Không dùng answer_relevancy làm KPI chính cho corpus này** — điểm thấp ổn định ở cả hai config là artifact của judge với câu trả lời ngắn tiếng Việt; giữ để theo dõi tương đối, quyết định bằng context_recall + đọc tay case.

## Bonus tracking (không áp dụng)

- Bonus chưa thực hiện trong lần chạy này; nếu thêm reranking Jina/BGE thì phải ghi baseline mới, metric delta và đổi latency trước khi tính bonus.
