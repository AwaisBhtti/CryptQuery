import os
import re
import json
import time
import base64
from pathlib import Path
import pymupdf
import requests
from dotenv import load_dotenv

# Load environment variables (.env file)
load_dotenv(override=True)

SLIDES_DIR = Path("IS Slides")
OUTPUT_FILE = Path("data/processed_chunks.json")
VISION_CACHE_FILE = Path("data/slide_vision_cache.json")

# Vision configuration
VISION_WORD_THRESHOLD = 25   # Slides with fewer words trigger vision
VISION_RATE_LIMIT_DELAY = 4  # Seconds between API calls (respect 15 RPM free tier)
# Available multimodal model on Google AI Studio
GEMINI_MODEL = "gemini-3.5-flash-lite"

VISION_PROMPT = """You are analyzing a university lecture slide from a CS3002 Information Security course.

Your task:
1. Extract ALL visible text exactly as it appears, including titles, bullet points, labels, and captions.
2. If there are mathematical formulas or cryptographic expressions (e.g., c = m^e mod n, AES, XOR, hash functions), write them out clearly using standard notation.
3. If there is a diagram, flow chart, architecture drawing, or sequence diagram, describe it technically in 2-4 sentences: what entities are involved, what the arrows or flows represent, and what the overall concept being illustrated is.
4. Do NOT add any extraneous conversational commentary or opinions.

Output clear, structured plain text."""

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

def load_vision_cache() -> dict:
    """Load cached vision descriptions to avoid re-calling Gemini for existing slides."""
    if VISION_CACHE_FILE.exists():
        try:
            with open(VISION_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARNING] Could not read vision cache ({e}). Starting fresh.")
    return {}

def save_vision_cache(cache: dict):
    """Save the persistent vision cache."""
    VISION_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(VISION_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def render_page_to_png_bytes(page) -> bytes:
    """Render a PyMuPDF page to PNG bytes at 150 DPI."""
    mat = pymupdf.Matrix(150 / 72, 150 / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=pymupdf.csRGB)
    return pix.tobytes("png")

def call_gemini_vision(png_bytes: bytes, api_key: str) -> str:
    """Send slide image to Gemini via REST API to extract text and diagram descriptions."""
    b64_img = base64.b64encode(png_bytes).decode("utf-8")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [
                {"text": VISION_PROMPT},
                {"inline_data": {"mime_type": "image/png", "data": b64_img}}
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 600
        }
    }
    
    response = requests.post(url, json=payload, timeout=60)
    if response.status_code == 200:
        data = response.json()
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "").strip()
        return ""
    else:
        raise RuntimeError(f"Gemini API returned HTTP {response.status_code}: {response.text[:200]}")

def get_vision_description(
    page,
    cache_key: str,
    vision_cache: dict,
    api_key: str | None
) -> str | None:
    """Check cache first; call Gemini API if new and save immediately."""
    if cache_key in vision_cache:
        return vision_cache[cache_key]

    if not api_key:
        return None

    try:
        png_bytes = render_page_to_png_bytes(page)
        description = call_gemini_vision(png_bytes, api_key)
        if description:
            vision_cache[cache_key] = description
            save_vision_cache(vision_cache)
            time.sleep(VISION_RATE_LIMIT_DELAY)  # Respect free tier rate limit
            return description
    except Exception as e:
        print(f"    [Vision Error] {cache_key}: {e}")
        return None

def extract_all():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    seen_hashes = set()
    
    vision_cache = load_vision_cache()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[WARNING] GEMINI_API_KEY not found in .env. Vision extraction will be skipped.")
    else:
        print(f"[INFO] Gemini Vision active using model '{GEMINI_MODEL}'. Existing cached slides: {len(vision_cache)}")

    pdf_files = sorted(list(SLIDES_DIR.glob("*.pdf")))
    print(f"Found {len(pdf_files)} PDF slide decks in '{SLIDES_DIR}'.\n")

    total_slides = 0
    duplicates_skipped = 0
    vision_api_calls = 0
    vision_cache_hits = 0

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
                    digital_word_count = len(cleaned.split())

                    # --- Vision Check ---
                    # Trigger vision if slide has few words and contains images/diagrams
                    vision_text = ""
                    cache_key = f"{pdf_path.name}::page_{page_idx + 1}"
                    has_images = bool(page.get_images())

                    if digital_word_count < VISION_WORD_THRESHOLD and has_images:
                        if cache_key in vision_cache:
                            vision_text = vision_cache[cache_key]
                            vision_cache_hits += 1
                        else:
                            print(f"  [Vision Processing] {pdf_path.name} (Slide {page_idx + 1}) - words: {digital_word_count}...")
                            desc = get_vision_description(page, cache_key, vision_cache, api_key)
                            if desc:
                                vision_text = desc
                                vision_api_calls += 1

                    # Combine digital text and vision description
                    combined_text = cleaned
                    if vision_text:
                        if combined_text:
                            combined_text = combined_text + "\n\n[Visual / Diagram Description]:\n" + vision_text
                        else:
                            combined_text = "[Visual / Diagram Description]:\n" + vision_text

                    # Skip empty slides with no content
                    if len(combined_text.split()) < 5:
                        continue

                    # Deduplication check
                    norm_hash = hash(re.sub(r"\s+", " ", combined_text).strip().lower())
                    if norm_hash in seen_hashes:
                        duplicates_skipped += 1
                        continue
                    seen_hashes.add(norm_hash)

                    # Chunk while preserving line/bullet structure
                    sub_chunks = chunk_text_preserve_structure(combined_text, max_words=300, overlap_lines=2)
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
                            "word_count": len(sub_chunk.split()),
                            "has_vision": bool(vision_text)
                        })
            finally:
                doc.close()

            print(f"Processed: {pdf_path.name} ({num_pages} pages)")
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\n=======================================================")
    print(f"[DONE] Slide Extraction Complete!")
    print(f"=======================================================")
    print(f"Total PDFs inspected:     {len(pdf_files)}")
    print(f"Total Slides inspected:   {total_slides}")
    print(f"Vision API calls made:    {vision_api_calls}")
    print(f"Vision cache hits:        {vision_cache_hits}")
    print(f"Duplicates skipped:       {duplicates_skipped}")
    print(f"Total clean chunks saved: {len(all_chunks)}")
    print(f"Chunks file:              {OUTPUT_FILE}")
    print(f"Vision cache file:        {VISION_CACHE_FILE}")

if __name__ == "__main__":
    extract_all()
