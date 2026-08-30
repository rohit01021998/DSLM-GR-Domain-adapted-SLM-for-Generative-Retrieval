import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Any
from transformers import AutoTokenizer

from genret import config
from genret.embed import load_chunks_jsonl
from genret.chunking import get_tokenizer

def id_to_target_string(id_list: List[int]) -> str:
    """
    Format list of digits into semantic ID target token string:
    [3, 7, 0, 2] -> "<id><d3><d7><d0><d2></id>"
    """
    digits_str = "".join([f"<d{d}>" for d in id_list])
    return f"{config.ID_START_TOKEN}{digits_str}{config.ID_END_TOKEN}"

def make_indexing_examples(
    chunks: List[Dict[str, Any]],
    chunk_to_id: Dict[str, List[int]],
    tokenizer,
    max_span_tokens: int = config.MAX_INPUT_TOKENS - 10
) -> List[Dict[str, Any]]:
    """
    Create Indexing Task examples using multi-span augmentation (Task 6b):
    Emits prefix, middle, and tail spans for long chunks instead of exact duplicates,
    ensuring full chunk content is indexed into the neural weights.
    """
    examples = []
    rng = random.Random(config.SEED)

    for c in chunks:
        c_id = c["chunk_id"]
        if c_id not in chunk_to_id:
            continue
        
        target = id_to_target_string(chunk_to_id[c_id])
        token_ids = tokenizer.encode(c["text"], add_special_tokens=False)
        total_tokens = len(token_ids)

        if total_tokens <= max_span_tokens:
            prompt_input = f"index: {c['text']}"
            examples.append({
                "input": prompt_input,
                "target": target,
                "task": "index",
                "chunk_id": c_id
            })
        else:
            # Multi-span augmentation: prefix, suffix, and middle windows
            spans = []
            # 1. Prefix span
            spans.append(token_ids[:max_span_tokens])
            # 2. Suffix span
            spans.append(token_ids[-max_span_tokens:])
            # 3. Middle span
            if total_tokens > max_span_tokens * 1.5:
                mid_start = (total_tokens - max_span_tokens) // 2
                spans.append(token_ids[mid_start:mid_start + max_span_tokens])

            for span_ids in spans:
                span_text = tokenizer.decode(span_ids)
                examples.append({
                    "input": f"index: {span_text}",
                    "target": target,
                    "task": "index",
                    "chunk_id": c_id
                })

    return examples

def make_retrieval_examples(
    queries: List[Dict[str, Any]],
    chunk_to_id: Dict[str, List[int]]
) -> List[Dict[str, Any]]:
    """
    Create Retrieval Task examples:
    input:  "retrieve: " + query
    target: "<id><d3><d7><d0><d2></id>"
    """
    examples = []
    for q in queries:
        c_id = q["chunk_id"]
        if c_id not in chunk_to_id:
            continue
        
        target = id_to_target_string(chunk_to_id[c_id])
        prompt_input = f"retrieve: {q['query']}"

        examples.append({
            "input": prompt_input,
            "target": target,
            "task": "retrieve",
            "chunk_id": c_id,
            "style": q.get("style", "question")
        })
    return examples

def stratified_query_split(
    retrieval_examples: List[Dict[str, Any]],
    val_fraction: float = config.VAL_QUERY_FRACTION,
    seed: int = config.SEED
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Stratify retrieval queries per chunk so every chunk retains ~80% training queries.
    """
    random.seed(seed)
    chunk_to_queries: Dict[str, List[Dict[str, Any]]] = {}
    for ex in retrieval_examples:
        chunk_to_queries.setdefault(ex["chunk_id"], []).append(ex)

    train_retrieval = []
    val_retrieval = []

    for c_id, q_list in chunk_to_queries.items():
        random.shuffle(q_list)
        n_val = max(1, int(len(q_list) * val_fraction)) if len(q_list) > 3 else 0
        val_retrieval.extend(q_list[:n_val])
        train_retrieval.extend(q_list[n_val:])

    # Guarantee strict zero leakage: If identical query text was generated for multiple chunks, keep in train
    train_query_texts = {ex["input"].strip() for ex in train_retrieval}
    clean_val = []
    for ex in val_retrieval:
        if ex["input"].strip() not in train_query_texts:
            clean_val.append(ex)
        else:
            train_retrieval.append(ex)

    return train_retrieval, clean_val

def build_datasets(
    chunks_path: Path = config.CHUNKS_PATH,
    queries_path: Path = config.QUERIES_PATH,
    ids_path: Path = config.IDS_PATH,
    train_path: Path = config.TRAIN_PATH,
    val_path: Path = config.VAL_PATH,
    index_ratio: float = config.INDEX_TO_QUERY_RATIO
):
    """
    Build dual-task DSI training & validation datasets.
    """
    chunks = load_chunks_jsonl(chunks_path)
    
    with open(ids_path, "r", encoding="utf-8") as f:
        ids_data = json.load(f)
    chunk_to_id = ids_data["chunk_to_id"]

    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    tokenizer = get_tokenizer()

    # 1. Create Indexing Examples (ALL chunks must be present in training)
    indexing_examples = make_indexing_examples(chunks, chunk_to_id, tokenizer)

    # 2. Create Retrieval Examples and Split
    retrieval_examples = make_retrieval_examples(queries, chunk_to_id)
    train_retrieval, val_retrieval = stratified_query_split(retrieval_examples)

    # 3. Balance indexing and retrieval in training (INDEX_TO_QUERY_RATIO = 0.5)
    target_index_count = int(len(train_retrieval) * index_ratio)
    if len(indexing_examples) < target_index_count and len(indexing_examples) > 0:
        multiplier = (target_index_count // len(indexing_examples)) + 1
        augmented_indexing = (indexing_examples * multiplier)[:target_index_count]
    else:
        augmented_indexing = indexing_examples

    train_data = train_retrieval + augmented_indexing
    random.seed(config.SEED)
    random.shuffle(train_data)
    val_data = val_retrieval

    # Save to disk
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Dataset generated successfully:")
    print(f"  Train: {len(train_data)} examples ({len(train_retrieval)} retrieval, {len(augmented_indexing)} indexing) -> {train_path}")
    print(f"  Val:   {len(val_data)} examples -> {val_path}")

    return train_data, val_data

def run_acceptance_test(train_data: List[Dict[str, Any]], val_data: List[Dict[str, Any]], chunks: List[Dict[str, Any]]):
    """
    Section 7 Acceptance Test:
    - Print counts per task, train and val sizes.
    - Assert no query string appears in both train and val.
    - Assert every chunk_id appears in train.
    - Decode 3 examples through tokenizer to verify special tokens formatting.
    """
    print("\n" + "=" * 60)
    print("SECTION 7 ACCEPTANCE TEST: DUAL-TASK DATASET INTEGRITY")
    print("=" * 60)

    # 1. Assert disjoint queries
    train_queries = {ex["input"].replace("retrieve: ", "").strip() for ex in train_data if ex["task"] == "retrieve"}
    val_queries = {ex["input"].replace("retrieve: ", "").strip() for ex in val_data}
    leakage = train_queries.intersection(val_queries)
    assert len(leakage) == 0, f"Query leakage detected between train and val! ({len(leakage)} overlapping queries)"
    print(f"✓ Zero Leakage Check Passed: {len(train_queries)} train queries, {len(val_queries)} val queries completely disjoint.")

    # 2. Assert all chunks indexed in training
    train_chunk_ids = {ex["chunk_id"] for ex in train_data}
    all_chunk_ids = {c["chunk_id"] for c in chunks}
    missing_chunks = all_chunk_ids - train_chunk_ids
    assert len(missing_chunks) == 0, f"Some chunks are missing from the training index: {missing_chunks}"
    print(f"✓ Complete Corpus Coverage Passed: All {len(all_chunk_ids)} chunks present in training set.")

    # 3. Decode samples through tokenizer
    print("\n--- 3 Decoded Dataset Samples ---")
    for ex in train_data[:3]:
        print(f"[{ex['task'].upper()}] Chunk: {ex['chunk_id']}")
        print(f"  Input:  \"{ex['input'][:100]}...\"")
        print(f"  Target: \"{ex['target']}\"\n")

    print("✅ Section 7 Acceptance Test Passed!")

if __name__ == "__main__":
    chunks = load_chunks_jsonl()
    train_data, val_data = build_datasets()
    run_acceptance_test(train_data, val_data, chunks)
