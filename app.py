import streamlit as st
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="",
    layout="wide",
)

from src.task10_generation import generate_with_citation


def render_sources(result: dict) -> None:
    """Hiển thị sources của GenerationResult kèm score và retrieval method."""
    if result.get("retrieval_source"):
        st.caption(f"Retrieval: {result['retrieval_source']}")
    for source in result.get("sources", []):
        metadata = source.get("metadata", {})
        label = metadata.get("title") or source.get("id", "Nguồn")
        with st.expander(f"{label} — score {source.get('score', 0):.4f}"):
            st.markdown(f"**Source:** {metadata.get('source', 'N/A')}")
            st.markdown(source.get("content", ""))


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("RAG Chatbot")
    st.caption("Thay mô tả theo đề tài của nhóm")
    top_k = st.slider("Số chunks", 3, 10, 5)

st.title("RAG Chatbot")
st.caption("Thay tiêu đề và hướng dẫn sử dụng")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "result" in message:
            render_sources(message["result"])

query = st.chat_input("Nhập câu hỏi...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        result = generate_with_citation(query, top_k)
        answer = result["answer"]
        st.markdown(answer)
        render_sources(result)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "result": result}
    )
