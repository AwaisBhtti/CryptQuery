import re
import json
from pathlib import Path
import pymupdf

SLIDES_DIR = Path("IS Slides")
OUTPUT_FILE = Path("data/processed_chunks.json")

def clean_text(text: str) -> str:
    """Clean common slide noise: headers, footers, duplicate whitespace, standalone page numbers."""
    lines = text.split("\n")
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Drop empty lines or standalone numbers (common page numbers in footers)
        if not stripped:
            continue
        if re.match(r"^\d{1,3}$", stripped):
            continue
        # Drop standalone generic slide footers/markers if any
        if re.match(r"^(page\s+\d+|slide\s+\d+|\d+\s*/\s*\d+)$", stripped, re.IGNORECASE):
            continue
        cleaned_lines.append(stripped)
    
    # Preserve bullet/line structure with newlines
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

def chunk_text_preserve_structure(text: str, max_words: int = 300, overlap_lines: int = 2) -> list[str]:
    """
    Chunk text while preserving line/bullet structure.
    If the text fits within max_words, keep it as-is.
    Otherwise, split by lines preserving readability and context.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]
    
    lines = text.split("\n")
    chunks = []
    curr_lines = []
    curr_word_count = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        line_word_count = len(line.split())
        
        # If adding this line exceeds max_words and we already have content
        if curr_word_count + line_word_count > max_words and curr_lines:
            chunks.append("\n".join(curr_lines))
            # Keep overlap lines for context
            curr_lines = curr_lines[-overlap_lines:] if len(curr_lines) >= overlap_lines else curr_lines
            curr_word_count = sum(len(l.split()) for l in curr_lines)
        
        curr_lines.append(line)
        curr_word_count += line_word_count
        i += 1
        
    if curr_lines:
        chunks.append("\n".join(curr_lines))
        
    return chunks

def extract_all():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    seen_hashes = set()
    
    pdf_files = sorted(list(SLIDES_DIR.glob("*.pdf")))
    print(f"Found {len(pdf_files)} PDF slide decks in '{SLIDES_DIR}'.")
    
    total_slides = 0
    duplicates_skipped = 0
    
    for pdf_path in pdf_files:
        try:
            doc = pymupdf.open(pdf_path)
            try:
                num_pages = len(doc)
                total_slides += num_pages
                
                for page_idx in range(num_pages):
                    page = doc[page_idx]
                    raw_text = page.get_text()
                    cleaned = clean_text(raw_text)
                    
                    # Skip blank slides or slides with almost no informational content (< 5 words)
                    if len(cleaned.split()) < 5:
                        continue
                    
                    # Simple dedup check (normalizing whitespace)
                    norm_hash = hash(re.sub(r"\s+", " ", cleaned).strip().lower())
                    if norm_hash in seen_hashes:
                        duplicates_skipped += 1
                        continue
                    seen_hashes.add(norm_hash)
                    
                    # Chunk while preserving line/bullet structure
                    sub_chunks = chunk_text_preserve_structure(cleaned, max_words=300, overlap_lines=2)
                    for sub_idx, sub_chunk in enumerate(sub_chunks):
                        chunk_id = f"{pdf_path.name}_p{page_idx + 1}"
                        if len(sub_chunks) > 1:
                            chunk_id += f"_s{sub_idx + 1}"
                        
                        all_chunks.append({
                            "id": chunk_id,
                            "source": pdf_path.name,
                            "slide_number": page_idx + 1,
                            "sub_chunk": sub_idx + 1 if len(sub_chunks) > 1 else None,
                            "text": sub_chunk,
                            "word_count": len(sub_chunk.split())
                        })
            finally:
                doc.close()
                
            print(f"Processed: {pdf_path.name} ({num_pages} pages)")
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
            
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
    print(f"\n[DONE] Extraction complete!")
    print(f"Total PDFs: {len(pdf_files)}")
    print(f"Total Slides inspected: {total_slides}")
    print(f"Duplicates skipped: {duplicates_skipped}")
    print(f"Total Cleaned Chunks created: {len(all_chunks)}")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_all()
