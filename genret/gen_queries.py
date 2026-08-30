import os
import re
import json
import yaml
import time
import requests
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Set
from tqdm import tqdm

from genret import config
from genret.embed import load_chunks_jsonl, load_embeddings

# Exact filter-out list from Emory IR Lab DUQGen
DUQGEN_FILTEROUT_ITEMS = [
    'In each of these examples', 'In each example,', 'Explanation:', 'Example 1:', 'Example 2:',
    'In this example', 'Note:', 'In the above examples', 'Note that in each example',
    'In the first three examples', 'In the first example', 'Answer:',
    'By analyzing the provided documents', 'Note that the examples', 'In both examples',
    'In each case', 'Note that the relevant queries', 'In each of the examples',
    'According to the text', 'Based on the passage'
]

def load_duqgen_template() -> str:
    """Load DUQGen 3-shot prompt template from cloned DUQGen repository."""
    duqgen_template_path = config.BASE_DIR / "DUQGen" / "data_preparation" / "prompt_templates" / "template_fiqa.yaml"
    if duqgen_template_path.exists():
        try:
            with open(duqgen_template_path, "r", encoding="utf-8") as f:
                templates = yaml.safe_load(f)
                return templates['3-shot']['template']
        except Exception as e:
            print(f"Warning: Could not parse DUQGen template ({e}). Using built-in.")
    
    return """Example 1:
Document: The CBTA scenario consists of 4 sub-scenarios: CBTAfs (Farside turn, same direction), CBTAfo (Farside turn, opposite direction). In all cases, the target speed is 15 km/h.
Relevant Query: What is the target speed for all CBTA turning sub-scenarios?

Example 2:
Document: Perform the testing with new original fitment tyres of the make, model, size, speed and load rating as specified by the vehicle manufacturer.
Relevant Query: What are the tyre requirements during vehicle preparation?

Example 3:
Document: {document}
Relevant Query: {query}"""

def clean_duqgen_query(raw_query: str) -> str:
    """Apply DUQGen post-processing and text sanitation."""
    q = raw_query.replace('\n', ' ').strip()
    for item in DUQGEN_FILTEROUT_ITEMS:
        q = q.split(item)[0].strip()
    
    # Strip quotes, numbering, prefixes
    q = re.sub(r'^(Relevant Query:\s*|\d+[\.\)]\s*|[-*•]\s*|["\'])', '', q)
    q = q.strip(' "[]\'')
    return q

def build_duqgen_prompt(chunk_text: str, style: str, n: int) -> str:
    """
    Construct multi-perspective synthetic query prompt using DUQGen principles.
    """
    if style == "question":
        style_instruction = f"Generate {n} natural, factual user questions."
    elif style == "acronym_grounded":
        style_instruction = f"Generate {n} technical questions that explicitly name the specific scenario acronyms (e.g. CBTAfs, CPTAns, CMRb, CCFhos, EMT, GVT, VUT), table parameters, or test speeds found in this passage."
    elif style == "keyword":
        style_instruction = f"Generate {n} concise keyword search queries (2 to 5 words, no grammar/punctuation)."
    elif style == "long":
        style_instruction = f"Generate {n} realistic scenario-based or multi-clause user questions."
    else:
        style_instruction = f"Generate {n} diverse search queries."

    return f"""You are an expert Information Retrieval query generator (DUQGen Engine).
Given the following technical passage, {style_instruction}

STRICT GROUND RULES:
1. Every query must be strictly answerable from this passage alone.
2. DO NOT copy exact 6-word phrases verbatim. Vary vocabulary and phrasing.
3. DO NOT use meta-phrases ("according to the passage", "the document states").
4. Return ONLY a valid JSON array of {n} strings.

Passage:
\"\"\"{chunk_text}\"\"\"

JSON Array:"""

def query_vllm_llm(
    prompt: str,
    base_url: str = config.GEN_BASE_URL,
    model_name: str = config.GEN_MODEL_NAME,
    max_tokens: int = 512,
    temperature: float = 0.7
) -> str:
    """Send generation request to local vLLM OpenAI-compatible server."""
    endpoint = f"{base_url}/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    try:
        resp = requests.post(endpoint, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"vLLM API Request Failed: {e}")
        return "[]"

def extract_json_array(response_text: str) -> List[str]:
    """Parse JSON array from model generation defensively."""
    cleaned = re.sub(r"^```json\s*", "", response_text, flags=re.MULTILINE)
    cleaned = re.sub(r"^```\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if match:
        try:
            arr = json.loads(match.group(0))
            if isinstance(arr, list):
                return [clean_duqgen_query(str(x)) for x in arr if clean_duqgen_query(str(x))]
        except Exception:
            pass

    # Fallback: line by line
    lines = [clean_duqgen_query(l) for l in cleaned.split("\n")]
    return [l for l in lines if len(l) > 6 and not l.startswith("[") and not l.endswith("]")]

def get_ngrams(text: str, n: int = 6) -> Set[str]:
    """Extract set of lowercase n-grams from text."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return set()
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}

def check_ngram_leakage(query: str, chunk_text: str, n: int = 6) -> bool:
    """Returns True if there is an exact n-gram overlap between query and chunk."""
    q_ngrams = get_ngrams(query, n)
    c_ngrams = get_ngrams(chunk_text, n)
    return bool(q_ngrams.intersection(c_ngrams))

def generate_for_chunk(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate multi-style DUQGen queries for a single chunk dynamically using config.QUERY_STYLE_MIX."""
    chunk_id = chunk["chunk_id"]
    chunk_text = chunk["text"]
    queries = []
    seen_queries: Set[str] = set()

    # Dynamic style allocation from config.QUERY_STYLE_MIX (Task 2)
    counts = {}
    for style, ratio in config.QUERY_STYLE_MIX.items():
        counts[style] = max(1, int(round(config.QUERIES_PER_CHUNK * ratio)))

    for style, count in counts.items():
        if count <= 0:
            continue
        prompt = build_duqgen_prompt(chunk_text, style, count)
        raw_output = query_vllm_llm(prompt)
        parsed_queries = extract_json_array(raw_output)

        for q in parsed_queries:
            q_clean = clean_duqgen_query(q)
            norm_q = re.sub(r"[^\w\s]", "", q_clean.lower()).strip()
            if not norm_q or norm_q in seen_queries or len(q_clean) < 6:
                continue
            seen_queries.add(norm_q)

            queries.append({
                "chunk_id": chunk_id,
                "query": q_clean,
                "style": style,
                "gen_model": config.GEN_MODEL_NAME
            })

    return queries

def run_query_generation(
    chunks_path: Path = config.CHUNKS_PATH,
    out_path: Path = config.QUERIES_PATH,
    limit: int = None,
    resume: bool = True
):
    """
    Generate synthetic queries for all chunks using DUQGen pipeline with checkpointing.
    """
    chunks = load_chunks_jsonl(chunks_path)
    if limit is not None:
        chunks = chunks[:limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume check
    completed_chunk_ids: Set[str] = set()
    if resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    completed_chunk_ids.add(record["chunk_id"])
        print(f"Resuming DUQGen query generation: {len(completed_chunk_ids)} chunks already completed.")

    remaining_chunks = [c for c in chunks if c["chunk_id"] not in completed_chunk_ids]
    print(f"Generating queries for {len(remaining_chunks)} / {len(chunks)} chunks via local vLLM DUQGen...")

    with open(out_path, "a", encoding="utf-8") as f:
        for chunk in tqdm(remaining_chunks, desc="DUQGen synthetic queries"):
            chunk_queries = generate_for_chunk(chunk)
            for q_obj in chunk_queries:
                f.write(json.dumps(q_obj, ensure_ascii=False) + "\n")
            f.flush()

    print(f"\nDUQGen query generation complete -> {out_path}")

def run_acceptance_test(queries_path: Path = config.QUERIES_PATH, chunks_path: Path = config.CHUNKS_PATH):
    """
    Section 6 Acceptance Test:
    - Total queries and 100% chunk coverage.
    - Style distribution verification.
    - 6-gram leakage check.
    - Sample preview.
    """
    print("\n" + "=" * 60)
    print("SECTION 6 ACCEPTANCE TEST: DUQGEN SYNTHETIC QUERY INTEGRITY")
    print("=" * 60)

    chunks = {c["chunk_id"]: c for c in load_chunks_jsonl(chunks_path)}
    
    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    chunk_query_counts = {}
    style_counts = {}
    leakage_count = 0

    for q in queries:
        c_id = q["chunk_id"]
        style = q.get("style", "unknown")
        chunk_query_counts[c_id] = chunk_query_counts.get(c_id, 0) + 1
        style_counts[style] = style_counts.get(style, 0) + 1
        
        if c_id in chunks:
            if check_ngram_leakage(q["query"], chunks[c_id]["text"], n=6):
                leakage_count += 1

    mean_per_chunk = len(queries) / max(1, len(chunk_query_counts))
    leakage_rate = (leakage_count / len(queries)) * 100 if queries else 0.0

    print(f"Total Queries:         {len(queries)}")
    print(f"Covered Chunks:        {len(chunk_query_counts)} / {len(chunks)}")
    print(f"Mean Queries / Chunk:  {mean_per_chunk:.2f}")
    print(f"Style Distribution:    {style_counts}")
    print(f"6-Gram Leakage Rate:   {leakage_rate:.2f}% ({leakage_count} queries)")

    # 100% Coverage Assertion (Task 1)
    assert len(chunk_query_counts) == len(chunks), f"Missing coverage: {len(chunk_query_counts)} != {len(chunks)} chunks covered!"
    print("✓ 100% Chunk Coverage Verified: All document chunks are covered by synthetic queries.")

    assert leakage_rate < 25.0, f"Leakage rate too high ({leakage_rate:.2f}%)!"
    print("✓ DUQGen Leakage Check Passed (< 25% verbatim 6-gram overlap)")

    print("\n--- 5 Random DUQGen Query Samples ---")
    import random
    random.seed(config.SEED)
    sample_queries = random.sample(queries, min(5, len(queries)))
    for q in sample_queries:
        c_snippet = chunks.get(q["chunk_id"], {}).get("text", "")[:100].replace("\n", " ")
        print(f"[{q['style'].upper()}] Query: \"{q['query']}\"")
        print(f"  Chunk [{q['chunk_id']}]: \"{c_snippet}...\"\n")

    print("✅ Section 6 Acceptance Test Passed!")

if __name__ == "__main__":
    run_query_generation(limit=None)
    run_acceptance_test()
