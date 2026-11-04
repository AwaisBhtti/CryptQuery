import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="CryptQuery - CS3002 Assistant", page_icon="🔒")

@st.cache_resource
def get_engine():
    return RAGEngine()

st.title("🔒 CryptQuery")
st.subheader("CS3002 Information Security Q&A Assistant")
st.caption("Answers grounded strictly in CS3002 lecture slides.")

# Initialize RAG Engine
with st.spinner("Loading course materials and vector index..."):
    engine = get_engine()

# Question Input
col_q, col_k = st.columns([4, 1])
with col_q:
    query = st.text_input("Ask a question about the course:", placeholder="e.g. What are the types of SQL injection?")
with col_k:
    top_k = st.number_input("Top Slides", min_value=3, max_value=8, value=5)

submit = st.button("Submit", type="primary")

if submit and query.strip():
    with st.spinner("Searching lecture slides and generating answer..."):
        result = engine.answer_question(query.strip(), top_k=top_k)

    st.markdown("### Answer")
    st.write(result["answer"])

    st.markdown("---")
    st.markdown("### Retrieved Slide Sources")
    for s in result["sources"]:
        with st.expander(f"📄 {s['source']} — Slide {s['slide_number']} (Score: {s['score']:.3f})"):
            st.text(s["text"])

