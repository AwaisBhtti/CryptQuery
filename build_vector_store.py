import json
import os
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Disable HF symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

CHUNKS_FILE = Path("data/processed_chunks.json")
VECTOR_STORE_DIR = Path("vector_store")
INDEX_FILE = VECTOR_STORE_DIR / "faiss_index.bin"
METADATA_FILE = VECTOR_STORE_DIR / "chunks_metadata.json"
MODEL_NAME = "all-MiniLM-L6-v2"

def build_vector_store():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Chunks file not found at {CHUNKS_FILE}. Run extract_slides.py first.")

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load chunks
    print(f"Loading chunks from {CHUNKS_FILE}...")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if not chunks:
        raise ValueError(f"No chunks found in {CHUNKS_FILE}.")
    print(f"Loaded {len(chunks)} chunks.")

    # 2. Load Sentence Transformer model
    print(f"Loading embedding model '{MODEL_NAME}'...")
    embedder = SentenceTransformer(MODEL_NAME)
    
    # 3. Compute Embeddings
    texts = [chunk["text"] for chunk in chunks]
    print(f"Encoding {len(texts)} chunks into dense vectors...")
    embeddings = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    dim = embeddings.shape[1]
    print(f"Embeddings shape: {embeddings.shape} (Dimension: {dim})")

    # 4. Build FAISS index (IndexFlatIP for exact cosine similarity with unit vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")

    # 5. Save FAISS index and metadata
    faiss.write_index(index, str(INDEX_FILE))
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
        
    print(f"\n[DONE] Vector store built successfully!")
    print(f"Index saved to: {INDEX_FILE}")
    print(f"Metadata saved to: {METADATA_FILE}")
    
    return embedder, index, chunks

def test_retrieval(query: str, embedder: SentenceTransformer, index: faiss.Index, metadata: list[dict], top_k: int = 3):
    print(f"\n=======================================================")
    print(f"Query: '{query}'")
    print(f"=======================================================")

    # Encode query with normalization
    query_vector = embedder.encode([query], normalize_embeddings=True)
    query_vector = np.array(query_vector, dtype=np.float32)

    # Search top_k
    scores, indices = index.search(query_vector, top_k)

    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        item = metadata[idx]
        print(f"\n[Rank {rank}] (Cosine Similarity: {score:.4f})")
        print(f"Source: {item['source']} (Slide {item['slide_number']})")
        # Print first 2-3 lines of snippet
        preview = "\n  ".join(item['text'].split("\n")[:4])
        print(f"Text snippet:\n  {preview}...")

if __name__ == "__main__":
    embedder, index, chunks = build_vector_store()
    test_retrieval("What is Role Based Access Control?", embedder, index, chunks)
    test_retrieval("Explain SQL injection attack and prevention", embedder, index, chunks)
