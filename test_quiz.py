"""Run with:  python -m pytest test_quiz.py   (no API key needed: the LLM is faked)"""
import json
import random
import sys
import types

import pytest

import quiz
from quiz import QuizError, extract_json, generate_mcqs, generate_scenario, grade_answer, match_source, sample_chunks

@pytest.fixture(autouse=True)
def light_rag_engine(monkeypatch):
    """quiz.py reads the model name from rag_engine; stub it so tests need no faiss/torch."""
    stub = types.ModuleType("rag_engine")
    stub.LLM_MODEL_NAME = "test-model"
    monkeypatch.setitem(sys.modules, "rag_engine", stub)


CHUNKS = [
    {"source": "SQLi.pdf", "slide_number": 25, "word_count": 60, "text": "Prepared statements separate code from data."},
    {"source": "SQLi.pdf", "slide_number": 17, "word_count": 55, "text": "UNION based injection appends a second query."},
    {"source": "SQLi.pdf", "slide_number": 1, "word_count": 8, "text": "Title slide"},
    {"source": "XSS.pdf", "slide_number": 13, "word_count": 70, "text": "Reflected XSS echoes input back."},
    {"source": "Mem Safety.pdf", "slide_number": 24, "word_count": 65, "text": "ASLR randomizes memory layout."},
    {"source": "Mem Safety.pdf", "slide_number": 12, "word_count": 90, "text": "A stack buffer overflow overwrites the return address."},
]


def fake_engine(reply):
    """Engine whose Groq client returns `reply` (a str, or a list of strs used in order)."""
    replies = [reply] if isinstance(reply, str) else list(reply)

    def create(**kwargs):
        text = replies.pop(0) if len(replies) > 1 else replies[0]
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))])

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    return types.SimpleNamespace(metadata=CHUNKS, groq_client=client)


GOOD_Q = {"question": "How do prepared statements stop SQL injection?",
          "options": ["They separate code from data", "They encrypt the database", "They hide table names", "They block all queries"],
          "answer_index": 0, "explanation": "Input is never parsed as SQL.", "source": "SQLi.pdf, Slide 25"}


def test_sample_chunks_filters_by_topic_and_short_slides():
    got = sample_chunks(CHUNKS, "SQL injection (database security)", 10, random.Random(0))
    assert {c["source"] for c in got} == {"SQLi.pdf"}
    assert all(c["word_count"] >= quiz.MIN_WORDS for c in got)  # title slide excluded
    assert len(sample_chunks(CHUNKS, "All topics", 3, random.Random(0))) == 3


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert extract_json('Sure! Here you go: {"a": 1} Hope it helps') == {"a": 1}
    with pytest.raises(QuizError):
        extract_json("no json here")


def test_match_source_only_accepts_given_slides():
    assert match_source("SQLi.pdf, Slide 25", CHUNKS)["slide_number"] == 25
    assert match_source("SQLi.pdf, Slide 99", CHUNKS) is None   # invented slide
    assert match_source("", CHUNKS) is None


def test_generate_mcqs_validates_and_shuffles():
    bad = dict(GOOD_Q, options=["only", "three", "options"])       # wrong option count
    dup = dict(GOOD_Q, options=["a", "a", "b", "c"])               # duplicate options
    oob = dict(GOOD_Q, answer_index=7)                             # invalid answer index
    eng = fake_engine(json.dumps([GOOD_Q, bad, dup, oob]))
    qs = generate_mcqs(eng, "All topics", 5, random.Random(1))
    assert len(qs) == 1
    q = qs[0]
    assert q["options"][q["answer_index"]] == "They separate code from data"  # answer tracked through shuffle
    assert q["source_ref"] == "SQLi.pdf, slide 25"


def test_generate_mcqs_retries_after_bad_output():
    eng = fake_engine(["not json at all", json.dumps([GOOD_Q])])
    assert len(generate_mcqs(eng, "All topics", 3, random.Random(0))) == 1


def test_generate_mcqs_fails_cleanly():
    with pytest.raises(QuizError):
        generate_mcqs(fake_engine("garbage"), "All topics", 3)
    no_key = types.SimpleNamespace(metadata=CHUNKS, groq_client=None)
    with pytest.raises(QuizError, match="GROQ_API_KEY"):
        generate_mcqs(no_key, "All topics", 3)


def test_unsourced_question_is_marked_not_invented():
    q = dict(GOOD_Q, source="Totally Invented.pdf, Slide 3")
    out = generate_mcqs(fake_engine(json.dumps([q])), "All topics", 1, random.Random(0))
    assert out[0]["source_ref"] is None


SCENARIO = {"scenario": "A login form builds its query by string concatenation.",
            "question": "What is the vulnerability and the fix?",
            "key_points": ["SQL injection", "Use prepared statements", "Validate input"],
            "model_answer": "It is SQL injection; use prepared statements.", "source": "SQLi.pdf, Slide 25"}


def test_generate_scenario():
    s = generate_scenario(fake_engine(json.dumps(SCENARIO)), "All topics", random.Random(0))
    assert s["key_points"] and s["source_ref"] == "SQLi.pdf, slide 25"


def test_grade_answer_clamps_score_and_skips_llm_for_empty_answer():
    item = generate_scenario(fake_engine(json.dumps(SCENARIO)), "All topics", random.Random(0))
    eng = fake_engine(json.dumps({"score": 9, "covered": ["SQL injection"], "missed": [], "feedback": "Good."}))
    assert grade_answer(eng, item, "It is SQL injection; use prepared statements.")["score"] == 5
    empty = grade_answer(fake_engine("SHOULD NOT BE CALLED"), item, "  ")
    assert empty["score"] == 0 and empty["missed"] == item["key_points"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))