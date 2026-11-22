"""Streamlit UI for Quiz Me mode (logic lives in quiz.py)."""
import uuid

import streamlit as st

from quiz import TOPICS, QuizError, generate_mcqs, generate_scenario, grade_answer

MCQ, SCENARIO = "Multiple choice", "Scenario (written answer)"
LETTERS = "ABCD"


def _start(engine, topic: str, qtype: str, n: int) -> None:
    """Generate a new quiz and store it in session state."""
    try:
        with st.spinner("Writing questions from the slides…"):
            if qtype == MCQ:
                items = generate_mcqs(engine, topic, n)
            else:
                items = [generate_scenario(engine, topic)]
    except QuizError as e:
        st.error(str(e))
        return
    st.session_state.quiz = {
        "id": uuid.uuid4().hex[:8], "type": qtype, "topic": topic, "n": n,
        "items": items, "i": 0, "checked": {}, "correct": {}, "grade": None,
    }
    st.rerun()


def _setup(engine) -> None:
    topic = st.selectbox("Topic", list(TOPICS), key="quiz_topic")
    qtype = st.radio("Question type", [MCQ, SCENARIO], horizontal=True, key="quiz_type")
    n = st.slider("Number of questions", 3, 10, 5, key="quiz_n") if qtype == MCQ else 1
    if st.button("Generate quiz", type="primary"):
        _start(engine, topic, qtype, n)


def _mcq(engine, quiz: dict) -> None:
    items, i, qid = quiz["items"], quiz["i"], quiz["id"]

    if i >= len(items):  # results
        score = sum(quiz["correct"].values())
        st.progress(1.0)
        st.subheader(f"Score: {score} / {len(items)}")
        missed = [k for k, ok in quiz["correct"].items() if not ok]
        if missed:
            st.markdown("**Review the ones you missed:**")
            for k in missed:
                q = items[k]
                with st.expander(q["question"]):
                    st.write(f"Correct answer: **{LETTERS[q['answer_index']]}. {q['options'][q['answer_index']]}**")
                    if q["explanation"]:
                        st.write(q["explanation"])
                    if q["source_ref"]:
                        st.caption(f"📚 {q['source_ref']}")
        else:
            st.success("Perfect score! 🎉")
        c1, c2 = st.columns(2)
        if c1.button("Same topic, new questions", use_container_width=True):
            _start(engine, quiz["topic"], quiz["type"], quiz["n"])
        if c2.button("Change topic", use_container_width=True):
            st.session_state.quiz = None
            st.rerun()
        return

    q = items[i]
    checked = quiz["checked"].get(i, False)
    st.progress(i / len(items))
    st.markdown(f"**Question {i + 1} of {len(items)}** · _{quiz['topic']}_")
    st.markdown(f"#### {q['question']}")
    labels = [f"{LETTERS[k]}. {opt}" for k, opt in enumerate(q["options"])]
    choice = st.radio("Your answer", labels, index=None, key=f"{qid}_{i}", disabled=checked)

    if not checked:
        if st.button("Check answer", type="primary", disabled=choice is None):
            quiz["checked"][i] = True
            quiz["correct"][i] = labels.index(choice) == q["answer_index"]
            st.rerun()
        return

    if quiz["correct"][i]:
        st.success("Correct ✅")
    else:
        st.error(f"Not quite. Correct answer: **{labels[q['answer_index']]}**")
    if q["explanation"]:
        st.write(q["explanation"])
    if q["source_ref"]:
        st.caption(f"📚 Source: {q['source_ref']}")
    last = i == len(items) - 1
    if st.button("See results" if last else "Next question →", type="primary"):
        quiz["i"] += 1
        st.rerun()


def _scenario(engine, quiz: dict) -> None:
    item, qid = quiz["items"][0], quiz["id"]
    st.markdown(f"_{quiz['topic']}_")
    st.markdown(f"**Scenario**\n\n{item['scenario']}")
    st.markdown(f"#### {item['question']}")
    answer = st.text_area("Your answer", height=160, key=f"{qid}_ans", disabled=quiz["grade"] is not None)

    if quiz["grade"] is None:
        if st.button("Submit answer", type="primary", disabled=not answer.strip()):
            try:
                with st.spinner("Grading against the course material…"):
                    quiz["grade"] = grade_answer(engine, item, answer)
            except QuizError as e:
                st.error(str(e))
                return
            st.rerun()
        return

    g = quiz["grade"]
    st.subheader(f"Score: {g['score']} / 5")
    if g["feedback"]:
        st.write(g["feedback"])
    c1, c2 = st.columns(2)
    with c1:
        if g["covered"]:
            st.markdown("**Covered**")
            for p in g["covered"]:
                st.markdown(f"- ✅ {p}")
    with c2:
        if g["missed"]:
            st.markdown("**Missed**")
            for p in g["missed"]:
                st.markdown(f"- ❌ {p}")
    with st.expander("Model answer"):
        st.write(item["model_answer"] or "\n".join(f"- {p}" for p in item["key_points"]))
        if item["source_ref"]:
            st.caption(f"📚 {item['source_ref']}")
    c1, c2 = st.columns(2)
    if c1.button("New scenario, same topic", use_container_width=True):
        _start(engine, quiz["topic"], quiz["type"], 1)
    if c2.button("Change topic", use_container_width=True):
        st.session_state.quiz = None
        st.rerun()


def render_quiz(engine) -> None:
    st.title("📝 Quiz Me")
    st.caption("Practice questions generated from the CS3002 slides. They are AI-written, so if an "
               "answer looks wrong, check the cited slide.")
    if st.session_state.get("quiz") is None:
        _setup(engine)
        return
    quiz = st.session_state.quiz
    if st.button("← Back to topics"):
        st.session_state.quiz = None
        st.rerun()
    (_mcq if quiz["type"] == MCQ else _scenario)(engine, quiz)