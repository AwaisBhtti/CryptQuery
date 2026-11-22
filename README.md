# 🔒 CryptQuery

A Retrieval-Augmented Generation (RAG) assistant for the **CS3002 Information Security** course. Students ask questions in a chat interface and get answers grounded **only** in the lecture slides, with the exact slide images shown as sources.

Built for the NLP course project (category: *RAG bot for niche domains*).

## Features

| Feature | What it does |
|---|---|
| **Grounded answers with citations** | The LLM answers strictly from retrieved slide text and cites `[Source: deck.pdf, Slide N]`. |
| **Slide previews** | Each retrieved source is shown as the rendered slide image (PyMuPDF), so diagrams such as TLS handshakes or stack layouts are visible, not just their text. |
| **Multi-turn chat with query rewriting** | Follow-ups like *"How do parameterized queries prevent it?"* are rewritten by the LLM into standalone search queries using the chat history before retrieval. |
| **Out-of-syllabus guardrail** | If the best dense cosine similarity is below a calibrated threshold, the app refuses with a warning and skips the LLM call, so it does not make up answers for questions outside the course. |
| **Vision-assisted extraction** | Slides with little text (diagrams, images) are described by a vision model, so their content becomes searchable. |

## How it works

```
IS Slides/*.pdf
     │  extract_slides.py   (PyMuPDF text + Gemini vision for image-heavy slides)
     ▼
data/processed_chunks.json            (562 chunks, one per slide; long slides split)
     │  build_vector_store.py   (all-MiniLM-L6-v2 embeddings, normalized)
     ▼
vector_store/faiss_index.bin          (FAISS IndexFlatIP = exact cosine similarity)
vector_store/chunks_metadata.json
     │
     ▼
app.py  →  rag_engine.py:
   1. condense_query()   rewrite follow-up using chat history (Groq LLM)
   2. retrieve()         embed query, FAISS top-k search
   3. is_in_scope()      guardrail: refuse if best cosine < threshold
   4. LLM answer         Groq, grounded prompt + citations
```

- **Embedding model:** `all-MiniLM-L6-v2` (Sentence-Transformers)
- **Vector index:** FAISS `IndexFlatIP` on normalized vectors (cosine similarity)
- **LLM:** `openai/gpt-oss-20b` via the Groq API
- **UI:** Streamlit
- **Corpus:** 21 lecture decks (malware, memory safety, SQLi, XSS, access control, authentication, TLS, cryptography, prompt injection, laws and ethics, …), 562 chunks

## Setup

Requires Python 3.10 or 3.11 (recommended).

```bash
git clone https://github.com/AwaisBhtti/CryptQuery.git
cd CryptQuery

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Create a `.env` file in the project root (copy `.env.example`):

```
GROQ_API_KEY=your_groq_api_key_here
SIMILARITY_THRESHOLD=0.44
```

Get a free Groq key at <https://console.groq.com/keys>.

## Run

```bash
streamlit run app.py
```

Open <http://localhost:8501>. The vector index and chunks are already committed, so you do **not** need to rebuild anything to run the app. The first run downloads the embedding model from Hugging Face.

### Rebuilding the index (only if slides change)

```bash
python extract_slides.py        # needs GEMINI_API_KEY in .env for slides that need vision
python build_vector_store.py
```

Vision descriptions are cached in `data/slide_vision_cache.json`, so existing slides are not re-sent to the API.

## Out-of-syllabus guardrail: calibration

The guardrail refuses when the top-1 dense cosine score is below `SIMILARITY_THRESHOLD`. The threshold is calibrated, not guessed:

```bash
python calibrate_threshold.py eval_queries.json
```

`eval_queries.json` has 22 in-syllabus questions, 15 out-of-syllabus questions, and 6 "hard negatives" (security topics that are *not* in the slides but are close to topics that are, e.g. stack canary, CSRF, SYN flood). The script sweeps thresholds and writes `threshold_sweep.csv` and `threshold_sweep.png` (precision, recall and F1 of refusal).

**Results**

| | Mean top score | Range edge |
|---|---|---|
| In-syllabus (22) | 0.648 | min 0.474 |
| Out-of-syllabus (15) | 0.258 | max 0.410 |

- The two groups are separated by a gap (0.41 – 0.47), so any threshold inside it gives perfect refusal precision and recall on this set. The threshold is set to **0.44**, the middle of that gap.
- A default of 0.35 would have wrongly answered two out-of-syllabus questions (*quantum key distribution*, *Kubernetes*), which is why calibration was needed.
- **Hard negatives:** at 0.44, 4 of 6 near-topic questions are refused. Questions very close to covered material (e.g. *multi-factor authentication*, *botnet*) still pass the similarity check.

**Limitations.** A similarity threshold cannot tell "close to the topic" from "covered by the slides". For those cases the LLM prompt is the second line of defence: it is instructed to answer *"this topic is not covered in the course material"* when the retrieved context does not contain the answer. The evaluation set is small and was written by the team, so the perfect score should be read as "the guardrail separates clearly unrelated questions well", not as a general accuracy claim.

The threshold can also be adjusted live from the **Relevance threshold** slider in the sidebar.

## Project structure

```
app.py                    Streamlit chat UI (sources, slide images, guardrail banner)
rag_engine.py             Retrieval, query rewriting, guardrail, LLM answering
extract_slides.py         PDF → text chunks (+ Gemini vision for image-heavy slides)
build_vector_store.py     Chunks → embeddings → FAISS index
search.py                 Command-line retrieval test (no LLM)
calibrate_threshold.py    Guardrail threshold sweep and plots
eval_queries.json         In / out / hard-negative questions for calibration
IS Slides/                Source lecture PDFs
data/                     Processed chunks and vision cache
vector_store/             FAISS index and chunk metadata
```

## Tech stack

Python · Streamlit · Sentence-Transformers (`all-MiniLM-L6-v2`) · FAISS · PyMuPDF · Groq API (`gpt-oss-20b`) · Google Gemini (slide vision)

## Team

Awais Akhtar 
Haris Fayyaz
Faiqa Waseem