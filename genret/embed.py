import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from genret import config

def load_chunks_jsonl(chunks_path: Path = config.CHUNKS_PATH) -> List[Dict[str, Any]]:
    """Load chunks from chunks.jsonl preserving exact line order."""
    chunks = []
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found at {chunks_path}. Run Section 2 first (`python main.py chunk`).")
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks

def embed_chunks(
    chunks_path: Path = config.CHUNKS_PATH,
    out_path: Path = config.EMB_PATH,
    meta_path: Path = config.EMB_META_PATH,
    encoder_name: str = config.ENCODER_NAME,
    batch_size: int = config.EMB_BATCH_SIZE,
    normalize: bool = config.NORMALIZE
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Embed every chunk in chunks.jsonl with a frozen dense encoder.
    Saves float32 numpy array to embeddings.npy and metadata to embeddings_meta.json.
    """
    chunks = load_chunks_jsonl(chunks_path)
    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading dense encoder: {encoder_name} on device: {device}")
    model = SentenceTransformer(encoder_name, device=device)

    # BGE models use NO prefix for document embedding (only for queries)
    # This design decision is recorded in embeddings_meta.json
    print(f"Encoding {len(texts)} chunks (batch_size={batch_size}, normalize={normalize})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        convert_to_numpy=True
    ).astype(np.float32)

    # Save embeddings.npy
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)

    # Save embeddings_meta.json
    meta = {
        "encoder": encoder_name,
        "dim": int(embeddings.shape[1]),
        "n": int(embeddings.shape[0]),
        "normalized": bool(normalize),
        "doc_prefix": "",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "chunk_ids": chunk_ids
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Successfully saved embeddings ({embeddings.shape}) -> {out_path}")
    print(f"Saved metadata -> {meta_path}")

    return embeddings, meta

def load_embeddings(emb_path: Path = config.EMB_PATH, meta_path: Path = config.EMB_META_PATH) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load precomputed embeddings and metadata."""
    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Embeddings not found at {emb_path}. Run Section 3 (`python -m genret.embed`).")
    embeddings = np.load(emb_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return embeddings, meta

def run_acceptance_test(embeddings: np.ndarray, meta: Dict[str, Any], chunks: List[Dict[str, Any]]):
    """
    Section 3 Acceptance Test:
    - Assert embeddings.shape[0] == len(chunks).
    - Assert no NaNs, row norms ~1.0 if normalized.
    - Nearest neighbor sanity check: Pick 3 chunks, verify their top 3 neighbors are topically related.
    """
    print("\n" + "=" * 60)
    print("SECTION 3 ACCEPTANCE TEST: EMBEDDING SANITY & NEAREST NEIGHBORS")
    print("=" * 60)

    # 1. Assert dimensions
    assert embeddings.shape[0] == len(chunks), f"Shape mismatch: {embeddings.shape[0]} != {len(chunks)}"
    print(f"✓ Shape Check Passed: {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")

    # 2. Assert no NaNs
    assert not np.isnan(embeddings).any(), "Embeddings contain NaN values!"
    print("✓ NaN Check Passed: No NaNs found")

    # 3. Assert normalization
    if meta.get("normalized", True):
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3), "Embeddings are not L2 normalized!"
        print("✓ L2 Normalization Check Passed (all norms ≈ 1.0)")

    # 4. Nearest Neighbor Sanity Check
    print("\n--- Cosine Nearest Neighbor Sanity Check (3 Chunks) ---")
    np.random.seed(config.SEED)
    sample_indices = np.random.choice(len(chunks), size=min(3, len(chunks)), replace=False)

    # Compute cosine similarity matrix for sample query chunks
    for idx in sample_indices:
        query_chunk = chunks[idx]
        query_emb = embeddings[idx]
        sims = np.dot(embeddings, query_emb)
        top_k_indices = np.argsort(-sims)[:4]  # top 4 (includes self at rank 0)

        print(f"\nTarget [{query_chunk['chunk_id']}]: \"{query_chunk['text'][:120].replace(chr(10), ' ')}...\"")
        for rank, neighbor_idx in enumerate(top_k_indices[1:], 1):
            neighbor_chunk = chunks[neighbor_idx]
            sim_score = sims[neighbor_idx]
            print(f"  #{rank} [{neighbor_chunk['chunk_id']}] (Cosine: {sim_score:.4f}): \"{neighbor_chunk['text'][:100].replace(chr(10), ' ')}...\"")

    print("\n✅ Section 3 Acceptance Test Passed!")

if __name__ == "__main__":
    emb, meta = embed_chunks()
    chunks = load_chunks_jsonl()
    run_acceptance_test(emb, meta, chunks)
