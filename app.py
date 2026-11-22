import streamlit as st
from rag_engine import RAGEngine, SIMILARITY_THRESHOLD
import pymupdf
from quiz_ui import render_quiz
from datetime import datetime
from pathlib import Path

SLIDES_DIR = Path("IS Slides")

# Example questions shown on an empty chat (all covered by the course slides)
STARTER_QUESTIONS = [
    "What is prompt injection?",
    "How do prepared statements prevent SQL injection?",
    "What is the difference between stored and reflected XSS?",
    "What is Role Based Access Control?",
    "How does Kerberos authentication work?",
    "How does the TLS handshake work?",
]

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

def chat_to_markdown(messages, sources, oos_flags) -> str:
    """Export the conversation as study notes: question, answer and the slides it came from."""
    lines = ["# CryptQuery: CS3002 chat export", f"_Exported {datetime.now():%Y-%m-%d %H:%M}_", ""]
    q_no, turn = 0, 0
    for msg in messages:
        if msg["role"] == "user":
            q_no += 1
            lines += [f"## Q{q_no}. {msg['content']}", ""]
        else:
            is_oos = turn < len(oos_flags) and oos_flags[turn]
            lines += [msg["content"], ""]
            srcs = sources[turn] if turn < len(sources) else []
            if is_oos:
                lines += ["_Marked as out of syllabus (no relevant slides found)._", ""]
            elif srcs:
                refs = ", ".join(f"{x['source']} (slide {x['slide_number']})" for x in srcs)
                lines += [f"**Sources:** {refs}", ""]
            turn += 1
    return "\n".join(lines)


# ── Init ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading course materials and vector index..."):
    engine = get_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []   # {"role": "user"|"assistant", "content": str}
if "sources" not in st.session_state:
    st.session_state.sources = []    # list[list[dict]] — one entry per assistant turn
if "quiz" not in st.session_state:
    st.session_state.quiz = None     # active Quiz Me session (see quiz_ui.py)
if "oos" not in st.session_state:
    st.session_state.oos = []        # list[bool] — out-of-syllabus flag per assistant turn

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    view = st.radio("Mode", ["💬 Chat", "📝 Quiz Me"], horizontal=True, key="view")
    if view == "💬 Chat":
        top_k = st.number_input("Top Slides (top_k)", min_value=3, max_value=8, value=5)
        threshold = st.slider(
            "Relevance threshold",
            min_value=0.10, max_value=0.80, value=float(SIMILARITY_THRESHOLD), step=0.01,
            help="If the best slide's cosine similarity is below this, the question is "
                 "treated as out-of-syllabus and no answer is generated.",
        )
        st.markdown("---")
        export_slot = st.empty()   # filled at the end of the script (includes the newest answer)
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.sources = []
            st.session_state.oos = []
            st.rerun()

if view == "📝 Quiz Me":
    render_quiz(engine)
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔒 CryptQuery")
st.subheader("CS3002 Information Security Q&A Assistant")
st.caption("Answers grounded strictly in CS3002 lecture slides with multi-turn conversation support.")

# ── Render existing chat history ───────────────────────────────────────────────
assistant_turn_index = 0

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        is_oos = (
            msg["role"] == "assistant"
            and assistant_turn_index < len(st.session_state.oos)
            and st.session_state.oos[assistant_turn_index]
        )
        if is_oos:
            st.warning(f"⚠️ {msg['content']}")
        else:
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
                st.markdown("**🔎 Closest slides (below threshold)**" if is_oos else "**📚 Retrieved Slide Sources**")
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

# ── Starter questions (only on an empty chat) ───────────────────────────────────
if not st.session_state.messages and not st.session_state.get("pending_query"):
    st.markdown("**Try asking:**")
    cols = st.columns(2)
    for i, q in enumerate(STARTER_QUESTIONS):
        if cols[i % 2].button(q, key=f"starter_{i}", use_container_width=True):
            st.session_state.pending_query = q
            st.rerun()

# ── Chat input ─────────────────────────────────────────────────────────────────
query = st.chat_input("Ask a question about the course…")
if not query and st.session_state.get("pending_query"):
    query = st.session_state.pop("pending_query")

if query and query.strip():
    # 1. Display and store user message
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # 2. Call RAG engine with history (exclude the current user message)
    # Drop refused (out-of-syllabus) turns so they don't pollute query rewriting
    prior = st.session_state.messages[:-1]
    history_for_engine = []
    for i in range(0, len(prior), 2):
        turn = i // 2
        if turn < len(st.session_state.oos) and st.session_state.oos[turn]:
            continue
        history_for_engine.extend(prior[i:i + 2])

    with st.spinner("Searching lecture slides and generating answer..."):
        result = engine.answer_with_history(
            query=query.strip(),
            history=history_for_engine,
            top_k=int(top_k),
            threshold=float(threshold),
        )
    is_oos = result.get("out_of_scope", False)

    # 3. Display and store assistant message
    with st.chat_message("assistant"):
        if is_oos:
            st.warning(f"⚠️ {result['answer']}")
            st.caption(f"Best match score {result['top_score']:.3f} < threshold {threshold:.2f}")
        else:
            st.write(result["answer"])

        if result.get("sources"):
            st.markdown("---")
            st.markdown("**🔎 Closest slides (below threshold)**" if is_oos else "**📚 Retrieved Slide Sources**")
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
    st.session_state.oos.append(is_oos)

# ── Export button (rendered last so it contains the latest turn) ──────────────
export_slot.download_button(
    "⬇️ Download chat (.md)",
    data=chat_to_markdown(st.session_state.messages, st.session_state.sources, st.session_state.oos),
    file_name="cryptquery_chat.md",
    mime="text/markdown",
    disabled=not st.session_state.messages,
    use_container_width=True,
)