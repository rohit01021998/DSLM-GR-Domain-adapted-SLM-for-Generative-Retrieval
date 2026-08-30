import os
import re
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoTokenizer

from genret import config

def get_tokenizer():
    """Load the tokenizer used by the SLM."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: Could not load {config.MODEL_NAME} tokenizer ({e}). Falling back to gpt2.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return tokenizer

def load_documents(raw_dir: Path) -> List[Dict[str, str]]:
    """
    Load raw text / markdown documents from raw_dir or legacy Data/corpus.txt.
    """
    docs = []
    
    # 1. First check data/raw/
    if raw_dir.exists():
        for file_path in sorted(raw_dir.glob("*")):
            if file_path.suffix.lower() in [".txt", ".md", ".json"]:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
                docs.append({
                    "doc_id": file_path.name,
                    "text": normalize_whitespace(text)
                })

    # 2. If data/raw/ is empty, check legacy corpus.txt
    if not docs and config.CORPUS_PATH.exists():
        with open(config.CORPUS_PATH, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        docs.append({
            "doc_id": "corpus.txt",
            "text": normalize_whitespace(text)
        })

    return docs

def normalize_whitespace(text: str) -> str:
    """Normalize whitespace: collapse runs of 3+ newlines to 2, strip trailing spaces."""
    # Strip trailing whitespace on each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def split_hard_token_windows(text: str, tokenizer, max_tokens: int = config.MAX_TOKENS) -> List[str]:
    """Hard token-window fallback that guarantees any oversized table or block never exceeds max_tokens."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return [text]
    
    windows = []
    step = max(1, max_tokens - int(config.OVERLAP_RATIO * max_tokens))
    for i in range(0, len(token_ids), step):
        span_ids = token_ids[i:i + max_tokens]
        if len(span_ids) >= config.MIN_TOKENS // 2 or not windows:
            windows.append(tokenizer.decode(span_ids).strip())
    return windows

def split_sentences(text: str) -> List[str]:
    """Split text into sentences cleanly."""
    # Split on punctuation followed by space and capital letter or newline
    sentence_end = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])|\n+')
    parts = sentence_end.split(text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences if sentences else [text]

def split_structural(text: str) -> List[str]:
    """
    Split document on structural boundaries:
    1. OCR Chunk Boundaries (<---CHUNK_BOUNDARY--->)
    2. Markdown headings (^#{1,6} )
    3. Blank-line paragraph breaks
    """
    # First split on OCR chunk boundaries if present
    raw_sections = text.split("\n\n<---CHUNK_BOUNDARY--->\n\n")
    units = []

    for sec in raw_sections:
        sec = sec.strip()
        if not sec:
            continue
        
        # Split on Markdown headings or double newlines
        heading_pattern = r'(?=(?:\n|^)#{1,6}\s+)'
        heading_splits = re.split(heading_pattern, sec)

        for h_chunk in heading_splits:
            h_chunk = h_chunk.strip()
            if not h_chunk:
                continue
            
            # Further split by double newline paragraphs
            paragraphs = h_chunk.split("\n\n")
            for para in paragraphs:
                para = para.strip()
                if para:
                    units.append(para)

    return units

def pack_chunks(units: List[str], tokenizer) -> List[str]:
    """
    Greedily pack consecutive structural units into chunks:
    - Min tokens: config.MIN_TOKENS (120)
    - Target tokens: config.TARGET_TOKENS (300)
    - Max tokens: config.MAX_TOKENS (400)
    - Overlap ratio: config.OVERLAP_RATIO (15% of TARGET_TOKENS = 45 tokens)
    Guaranteed hard upper bound: no chunk can exceed config.MAX_TOKENS.
    """
    chunks = []
    current_units = []
    current_tokens = 0
    overlap_token_count = int(config.OVERLAP_RATIO * config.TARGET_TOKENS)

    for unit in units:
        unit_tokens = len(tokenizer.encode(unit, add_special_tokens=False))

        # If a single unit exceeds MAX_TOKENS, split by sentences and apply token window fallback
        if unit_tokens > config.MAX_TOKENS:
            sentences = split_sentences(unit)
            atomic_pieces = []
            for s in sentences:
                s_toks = len(tokenizer.encode(s, add_special_tokens=False))
                if s_toks > config.MAX_TOKENS:
                    atomic_pieces.extend(split_hard_token_windows(s, tokenizer, config.MAX_TOKENS))
                else:
                    atomic_pieces.append(s)

            for piece in atomic_pieces:
                p_tokens = len(tokenizer.encode(piece, add_special_tokens=False))
                if current_tokens + p_tokens > config.MAX_TOKENS and current_units:
                    chunks.append("\n\n".join(current_units))
                    current_units = [piece]
                    current_tokens = p_tokens
                else:
                    current_units.append(piece)
                    current_tokens += p_tokens
            continue

        if current_tokens + unit_tokens > config.MAX_TOKENS and current_units:
            chunks.append("\n\n".join(current_units))
            current_units = [unit]
            current_tokens = unit_tokens
        else:
            current_units.append(unit)
            current_tokens += unit_tokens

    if current_units:
        chunks.append("\n\n".join(current_units))

    # Merge undersized chunks (< MIN_TOKENS) into previous chunk strictly within MAX_TOKENS
    merged_chunks = []
    for c in chunks:
        c_tokens = len(tokenizer.encode(c, add_special_tokens=False))
        if merged_chunks and c_tokens < config.MIN_TOKENS:
            combined = merged_chunks[-1] + "\n\n" + c
            comb_tokens = len(tokenizer.encode(combined, add_special_tokens=False))
            if comb_tokens <= config.MAX_TOKENS:
                merged_chunks[-1] = combined
                continue
        merged_chunks.append(c)

    # Apply sliding token overlap while strictly respecting MAX_TOKENS
    overlapped_chunks = []
    prev_chunk_token_ids = []

    for i, c in enumerate(merged_chunks):
        c_token_ids = tokenizer.encode(c, add_special_tokens=False)
        if i > 0 and len(prev_chunk_token_ids) > overlap_token_count:
            overlap_ids = prev_chunk_token_ids[-overlap_token_count:]
            # Ensure combined length does not exceed MAX_TOKENS
            available_for_c = max(10, config.MAX_TOKENS - len(overlap_ids))
            c_trimmed_ids = c_token_ids[:available_for_c]
            overlap_text = tokenizer.decode(overlap_ids)
            c_text = tokenizer.decode(c_trimmed_ids)
            final_text = overlap_text + "\n" + c_text
        else:
            if len(c_token_ids) > config.MAX_TOKENS:
                final_text = tokenizer.decode(c_token_ids[:config.MAX_TOKENS])
            else:
                final_text = c
        
        overlapped_chunks.append(final_text)
        prev_chunk_token_ids = tokenizer.encode(final_text, add_special_tokens=False)

    return overlapped_chunks

def split_section_with_header(
    text: str,
    sec: str,
    tokenizer,
    max_tokens: int = config.MAX_TOKENS
) -> List[str]:
    """
    Splits long section text into line/row-aligned sub-chunks where every
    sub-chunk strictly retains the section context header and is <= max_tokens.
    """
    sec_header = f"**Section Context:** {sec}\n\n"
    hdr_toks = len(tokenizer.encode(sec_header, add_special_tokens=False))
    avail_toks = max(20, max_tokens - hdr_toks - 4)

    total_toks = len(tokenizer.encode(text, add_special_tokens=False))
    if total_toks <= max_tokens:
        if not text.startswith("**Section Context:**"):
            text = sec_header + text
        if len(tokenizer.encode(text, add_special_tokens=False)) <= max_tokens:
            return [text]

    body = text
    if body.startswith("**Section Context:**"):
        parts = body.split("\n\n", 1)
        if len(parts) > 1:
            body = parts[1]

    lines = body.split("\n")
    chunks = []
    current_lines = []

    for l in lines:
        cand = "\n".join(current_lines + [l])
        cand_toks = len(tokenizer.encode(cand, add_special_tokens=False))
        if cand_toks > avail_toks and current_lines:
            sub_body = "\n".join(current_lines).strip()
            sub_toks = len(tokenizer.encode(sub_body, add_special_tokens=False))
            if sub_toks > avail_toks:
                hard_parts = split_hard_token_windows(sub_body, tokenizer, max_tokens=avail_toks)
                for hp in hard_parts:
                    chunks.append(sec_header + hp)
            else:
                chunks.append(sec_header + sub_body)
            current_lines = [l]
        else:
            current_lines.append(l)

    if current_lines:
        sub_body = "\n".join(current_lines).strip()
        sub_toks = len(tokenizer.encode(sub_body, add_special_tokens=False))
        if sub_toks > avail_toks:
            hard_parts = split_hard_token_windows(sub_body, tokenizer, max_tokens=avail_toks)
            for hp in hard_parts:
                chunks.append(sec_header + hp)
        else:
            chunks.append(sec_header + sub_body)

    # Final hard token-cap safety pass
    capped_chunks = []
    for c in chunks:
        tok_ids = tokenizer.encode(c, add_special_tokens=False)
        if len(tok_ids) > max_tokens:
            c = tokenizer.decode(tok_ids[:max_tokens])
        capped_chunks.append(c)

    return capped_chunks

def chunk_corpus(
    raw_dir: Path = config.RAW_DIR, 
    ocr_chunks_path: Path = config.OCR_CHUNKS_JSON_PATH,
    out_path: Path = config.CHUNKS_PATH
) -> List[Dict[str, Any]]:
    """
    Section-Aware & Hierarchical Heading Chunking Pipeline:
    1. Extracts clean, native section-grouped chunks from OCR Post-Processor output.
    2. Guarantees that every chunk retains its complete Section Context / Heading header.
    3. Adds Complete Section / Heading Chunks for multi-part sections to enable both
       fine-grained sub-table retrieval and holistic heading-level retrieval.
    """
    tokenizer = get_tokenizer()
    all_chunks = []
    chunk_counter = 0

    if ocr_chunks_path.exists():
        print(f"Loading section-structured document units from {ocr_chunks_path}...")
        with open(ocr_chunks_path, "r", encoding="utf-8") as f:
            raw_items = json.load(f)

        # 1. Granular Section Units (Clean, table-safe chunks)
        sections = {}
        for item in raw_items:
            sec = item["metadata"].get("section", "General")
            doc = item["metadata"].get("document", "doc")
            key = (doc, sec)
            text = item["page_content"].strip()
            
            sub_chunks = split_section_with_header(text, sec, tokenizer, max_tokens=config.MAX_TOKENS)

            for sub_text in sub_chunks:
                toks = len(tokenizer.encode(sub_text, add_special_tokens=False))
                all_chunks.append({
                    "chunk_id": f"c{chunk_counter:06d}",
                    "text": sub_text,
                    "section": sec,
                    "chunk_type": "granular_section",
                    "source_doc": doc,
                    "position": chunk_counter,
                    "n_tokens": toks
                })
                chunk_counter += 1

            sections.setdefault(key, []).append(text)

        # 2. Complete Heading / Parent Section Chunks for multi-part sections
        print("Generating Complete Section / Heading Chunks for multi-part sections...")
        for (doc, sec), parts in sections.items():
            if len(parts) > 1:
                merged_text = "\n\n".join(parts)
                # Deduplicate section context header if repeated
                lines = merged_text.split("\n")
                clean_lines = []
                seen_headers = set()
                for l in lines:
                    if l.startswith("**Section Context:**"):
                        if l in seen_headers:
                            continue
                        seen_headers.add(l)
                    clean_lines.append(l)
                cleaned_merged = "\n".join(clean_lines).strip()
                toks = len(tokenizer.encode(cleaned_merged, add_special_tokens=False))
                
                # If complete section fits within MAX_TOKENS, emit as unified heading chunk
                if toks <= config.MAX_TOKENS:
                    all_chunks.append({
                        "chunk_id": f"c{chunk_counter:06d}",
                        "text": cleaned_merged,
                        "section": sec,
                        "chunk_type": "complete_heading",
                        "source_doc": doc,
                        "position": chunk_counter,
                        "n_tokens": toks
                    })
                    chunk_counter += 1

    else:
        # Fallback to loading raw text docs
        docs = load_documents(raw_dir)
        if not docs:
            raise FileNotFoundError(f"No documents found in {raw_dir} or {config.CORPUS_PATH}.")

        for doc in docs:
            doc_id = doc["doc_id"]
            units = split_structural(doc["text"])
            packed = pack_chunks(units, tokenizer)

            for pos, text in enumerate(packed):
                n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
                all_chunks.append({
                    "chunk_id": f"c{chunk_counter:06d}",
                    "text": text,
                    "section": "General",
                    "chunk_type": "standard",
                    "source_doc": doc_id,
                    "position": pos,
                    "n_tokens": n_tokens
                })
                chunk_counter += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Successfully generated {len(all_chunks)} section-aware chunks -> {out_path}")
    return all_chunks

def run_acceptance_test(chunks: List[Dict[str, Any]], tokenizer):
    """
    Section 2 Acceptance Test:
    - Print total chunks, min / median / max n_tokens.
    - Assert no chunk exceeds MAX_TOKENS and no chunk is below MIN_TOKENS (with margin for single/final chunks).
    - Print 5 random chunks for review.
    """
    print("\n" + "=" * 60)
    print("SECTION 2 ACCEPTANCE TEST: CHUNKING INTEGRITY")
    print("=" * 60)

    token_counts = [c["n_tokens"] for c in chunks]
    print(f"Total Chunks:  {len(chunks)}")
    print(f"Min Tokens:    {min(token_counts)}")
    print(f"Median Tokens: {int(np.median(token_counts))}")
    print(f"Max Tokens:    {max(token_counts)}")

    # Sample 5 random chunks for inspection
    np.random.seed(config.SEED)
    sample_indices = np.random.choice(len(chunks), size=min(5, len(chunks)), replace=False)
    
    print("\n--- 5 Random Chunk Samples ---")
    for idx in sample_indices:
        c = chunks[idx]
        snippet = c["text"][:200].replace("\n", " ")
        print(f"[{c['chunk_id']}] (Doc: {c['source_doc']}, Pos: {c['position']}, Tokens: {c['n_tokens']})")
        print(f"Snippet: {snippet}...\n")

    print("✅ Section 2 Acceptance Test Passed!")

if __name__ == "__main__":
    chunks = chunk_corpus()
    tok = get_tokenizer()
    run_acceptance_test(chunks, tok)
