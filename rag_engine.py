import os
import json
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq

# Load environment variables (.env file)
load_dotenv()

INDEX_FILE = Path("vector_store/faiss_index.bin")
METADATA_FILE = Path("vector_store/chunks_metadata.json")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "openai/gpt-oss-120b"  # Fast, highly capable LLM on Groq

class RAGEngine:
    def __init__(self, index_path=INDEX_FILE, metadata_path=METADATA_FILE):
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("Vector index or metadata not found. Run build_vector_store.py first.")
            
        print("Initializing RAG Engine...")
        # 1. Load Embedder
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # 2. Load FAISS index & metadata
        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        # 3. Setup Groq Client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("[WARNING] GROQ_API_KEY is not set in environment or .env file.")
        self.groq_client = Groq(api_key=api_key) if api_key else None
        print("RAG Engine ready.")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve the top_k most relevant slide chunks."""
        query_vec = self.embedder.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype=np.float32)
        
        scores, indices = self.index.search(query_vec, top_k)
        
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            chunk = self.metadata[idx]
            results.append({
                "rank": rank,
                "score": float(score),
                "source": chunk["source"],
                "slide_number": chunk["slide_number"],
                "text": chunk["text"]
            })
        return results

    def build_prompt(self, query: str, context_chunks: list[dict]) -> str:
        """Construct the grounded prompt with slide context."""
        context_str = ""
        for i, chunk in enumerate(context_chunks, start=1):
            context_str += f"\n--- Context Document {i} [Source: {chunk['source']}, Slide: {chunk['slide_number']}] ---\n"
            context_str += f"{chunk['text']}\n"

        prompt = f"""You are CryptQuery, an academic Q&A assistant for the CS3002 Information Security course.
Answer the question based STRICTLY and ONLY on the provided lecture slide context below.

GUIDELINES:
1. Ground every claim directly in the slide context.
2. Cite the source slide deck and slide number for every major point, for example: [Source: SQLi.pdf, Slide 10].
3. If the context does not contain enough information to answer the question, or if the question asks about something completely outside these slides, do NOT fabricate facts. Respond clearly with:
   "Based on the provided CS3002 lecture slides, this topic is not covered in the course material."
4. Be concise, structured, and informative.

CONTEXT FROM SLIDES:
{context_str}

USER QUESTION:
{query}

ANSWER:"""
        return prompt

    def answer_question(self, query: str, top_k: int = 3) -> dict:
        """End-to-end pipeline: Retrieve -> Format -> Call LLM."""
        retrieved_chunks = self.retrieve(query, top_k=top_k)
        
        if not self.groq_client:
            return {
                "answer": "Error: GROQ_API_KEY is not set. Please create a .env file with your GROQ_API_KEY.",
                "sources": retrieved_chunks
            }

        prompt = self.build_prompt(query, retrieved_chunks)

        try:
            response = self.groq_client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a precise, grounded teaching assistant for the CS3002 Information Security course."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Low temperature for factual grounding
                max_tokens=600
            )
            answer_text = response.choices[0].message.content.strip()
        except Exception as e:
            answer_text = f"Error calling Groq API: {e}"

        return {
            "answer": answer_text,
            "sources": retrieved_chunks
        }

if __name__ == "__main__":
    import sys
    # Ensure Windows console supports UTF-8 characters like dashes and bullets
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        
    engine = RAGEngine()
    
    test_question = "What is SQL Injection and how can it be prevented?"
    print(f"\n--- Testing Question: {test_question} ---")
    result = engine.answer_question(test_question, top_k=3)
    
    print("\n[ANSWER]:\n")
    print(result["answer"])
    
    print("\n[RETRIEVED SOURCES]:")
    for s in result["sources"]:
        print(f"- {s['source']} (Slide {s['slide_number']}) [Score: {s['score']:.3f}]")

