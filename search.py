import json
import os
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

INDEX_FILE = Path("vector_store/faiss_index.bin")
METADATA_FILE = Path("vector_store/chunks_metadata.json")
MODEL_NAME = "all-MiniLM-L6-v2"

def interactive_search():
    if not INDEX_FILE.exists() or not METADATA_FILE.exists():
        print("[ERROR] Index or metadata not found. Run build_vector_store.py first!")
        return

    print("Loading vector index and model (one-time load)...")
    embedder = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(str(INDEX_FILE))
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("\n" + "=" * 60)
    print("  CryptQuery — Interactive Slide Retrieval Test")
    print("  Type any question or keyword to see matching slides.")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("Ask a question: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Exiting search. Goodbye!")
                break

            # Search top 3 chunks
            q_vector = embedder.encode([query], normalize_embeddings=True)
            q_vector = np.array(q_vector, dtype=np.float32)
            scores, indices = index.search(q_vector, 3)

            print(f"\nTop Matches for: '{query}'")
            print("-" * 60)
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
                item = metadata[idx]
                print(f"Rank {rank} | Score: {score:.3f} | File: {item['source']} (Slide {item['slide_number']})")
                print("Content:")
                for line in item['text'].split("\n"):
                    print(f"   {line}")
                print("-" * 60)
            print()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting search. Goodbye!")
            break

if __name__ == "__main__":
    interactive_search()

