import streamlit as st
from rag_engine import RAGEngine
import pymupdf
from pathlib import Path

SLIDES_DIR = Path("IS Slides")

st.set_page_config(page_title="CryptQuery - CS3002 Assistant", page_icon="🔒", layout="centered")

@st.cache_resource
def get_engine():
    return RAGEngine()

@st.cache_data
def render_slide_image(source_filename: str, slide_number: int) -> bytes:
    doc = pymupdf.open(SLIDES_DIR / source_filename)
    page = doc[slide_number - 1]
    pix = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5))
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes

# ── Init ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading course materials and vector index..."):
    engine = get_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []   # {"role": "user"|"assistant", "content": str}
if "sources" not in st.session_state:
    st.session_state.sources = []    # list[list[dict]] — one entry per assistant turn

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    top_k = st.number_input("Top Slides (top_k)", min_value=3, max_value=8, value=5)
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources = []
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔒 CryptQuery")
st.subheader("CS3002 Information Security Q&A Assistant")
st.caption("Answers grounded strictly in CS3002 lecture slides with multi-turn conversation support.")

# ── Render existing chat history ───────────────────────────────────────────────
assistant_turn_index = 0

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        # If assistant message, show its sources below the answer
        if msg["role"] == "assistant":
            sources_for_turn = (
                st.session_state.sources[assistant_turn_index]
                if assistant_turn_index < len(st.session_state.sources)
                else []
            )
            if sources_for_turn:
                st.markdown("---")
                st.markdown("**📚 Retrieved Slide Sources**")
                for s in sources_for_turn:
                    with st.expander(
                        f"📄 {s['source']} — Slide {s['slide_number']} (Score: {s['score']:.3f})"
                    ):
                        try:
                            img = render_slide_image(s["source"], s["slide_number"])
                            st.image(img, use_container_width=True)
                        except Exception:
                            st.warning(f"Could not render slide image for {s['source']}.")
                        st.markdown("**Extracted Text:**")
                        st.text(s["text"])
            assistant_turn_index += 1

# ── Chat input ─────────────────────────────────────────────────────────────────
query = st.chat_input("Ask a question about the course…")

if query and query.strip():
    # 1. Display and store user message
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # 2. Call RAG engine with history (exclude the current user message)
    history_for_engine = st.session_state.messages[:-1]

    with st.spinner("Searching lecture slides and generating answer..."):
        result = engine.answer_with_history(
            query=query.strip(),
            history=history_for_engine,
            top_k=int(top_k),
        )

    # 3. Display and store assistant message
    with st.chat_message("assistant"):
        st.write(result["answer"])

        if result.get("sources"):
            st.markdown("---")
            st.markdown("**📚 Retrieved Slide Sources**")
            for s in result["sources"]:
                with st.expander(
                    f"📄 {s['source']} — Slide {s['slide_number']} (Score: {s['score']:.3f})"
                ):
                    try:
                        img = render_slide_image(s["source"], s["slide_number"])
                        st.image(img, use_container_width=True)
                    except Exception:
                        st.warning(f"Could not render slide image for {s['source']}.")
                    st.markdown("**Extracted Text:**")
                    st.text(s["text"])

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
    st.session_state.sources.append(result["sources"])
