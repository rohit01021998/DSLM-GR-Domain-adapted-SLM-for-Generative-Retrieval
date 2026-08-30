import json
import re
import random
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
from tqdm import tqdm

from genret import config
from genret.infer import Retriever
from genret.baselines import BM25Retriever, DenseRetriever

def load_eval_data(eval_path: Path) -> List[Dict[str, Any]]:
    """Load query-target evaluation pairs from JSONL or CSV."""
    records = []
    if not eval_path.exists():
        return records

    if eval_path.suffix == ".jsonl":
        with open(eval_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    query = item["input"].replace("retrieve: ", "").strip() if "input" in item else item.get("query", "")
                    records.append({
                        "query": query,
                        "gold_chunk_id": item["chunk_id"],
                        "target_id_str": item.get("target", "")
                    })
    elif eval_path.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(eval_path)
        for _, row in df.iterrows():
            records.append({
                "query": str(row["question"]).strip(),
                "gold_chunk_id": str(row["corpus_id"]).strip(),
                "target_id_str": ""
            })

    return records

def compute_retrieval_metrics(
    predictions: List[List[Dict[str, Any]]],
    gold_chunk_ids: List[str]
) -> Dict[str, float]:
    """Compute Hits@1, Hits@5, Hits@10, and MRR@10."""
    hits1 = 0
    hits5 = 0
    hits10 = 0
    mrr10 = 0.0

    n = len(gold_chunk_ids)
    if n == 0:
        return {"hits@1": 0.0, "hits@5": 0.0, "hits@10": 0.0, "mrr@10": 0.0}

    for preds, gold in zip(predictions, gold_chunk_ids):
        assert len(preds) > 0, "Retriever returned an empty prediction list!"
        pred_ids = [p["chunk_id"] for p in preds]

        if len(pred_ids) > 0 and pred_ids[0] == gold:
            hits1 += 1
        if gold in pred_ids[:5]:
            hits5 += 1
        if gold in pred_ids[:10]:
            hits10 += 1

        for rank, p_id in enumerate(pred_ids[:10], start=1):
            if p_id == gold:
                mrr10 += 1.0 / rank
                break

    return {
        "hits@1": hits1 / n,
        "hits@5": hits5 / n,
        "hits@10": hits10 / n,
        "mrr@10": mrr10 / n
    }

def compute_per_depth_accuracy(
    genret_results: List[List[Dict[str, Any]]],
    gold_chunk_ids: List[str],
    chunk_to_id: Dict[str, List[int]]
) -> Dict[str, float]:
    """Compute per-depth prefix accuracy of top beam."""
    depth_correct = {}
    depth_totals = {}

    for preds, gold_cid in zip(genret_results, gold_chunk_ids):
        if not preds or gold_cid not in chunk_to_id:
            continue
        
        gold_path = chunk_to_id[gold_cid]
        pred_path = preds[0].get("id_path", [])

        for depth in range(1, len(gold_path) + 1):
            d_key = f"depth_{depth}"
            depth_totals[d_key] = depth_totals.get(d_key, 0) + 1
            if len(pred_path) >= depth and pred_path[:depth] == gold_path[:depth]:
                depth_correct[d_key] = depth_correct.get(d_key, 0) + 1

    per_depth_acc = {}
    for d_key, total in depth_totals.items():
        per_depth_acc[d_key] = depth_correct.get(d_key, 0) / max(1, total)

    return per_depth_acc

def fuse_rrf(
    list_of_ranked_lists: List[List[Dict[str, Any]]], 
    k: int = 10, 
    rrf_k: int = 60
) -> List[Dict[str, Any]]:
    """
    Standard Reciprocal Rank Fusion (RRF):
    RRF_score(d) = sum_{retriever}( 1 / (rrf_k + rank(d)) )
    """
    scores = {}
    doc_map = {}
    for ranked_list in list_of_ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            cid = item["chunk_id"]
            if cid not in scores:
                scores[cid] = 0.0
                doc_map[cid] = item
            scores[cid] += 1.0 / (rrf_k + rank)

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    fused_results = []
    for rank, (cid, score) in enumerate(sorted_docs, start=1):
        fused_results.append({
            "rank": rank,
            "chunk_id": cid,
            "text": doc_map[cid].get("text", ""),
            "score": float(score)
        })
    return fused_results

def evaluate_all(
    run_name: Optional[str] = None,
    eval_path: Path = config.VAL_PATH,
    out_dir: Path = None
):
    """
    Full comparative evaluation: DSI Generative Retrieval vs BM25 vs Dense vs Hybrid.
    """
    if run_name is None:
        candidates = [
            "dsi_smollm2-1.7b-instruct",
            "dsi_qwen2.5-0.5b",
            "dsi_qwen_0.5b"
        ]
        for c in candidates:
            if (config.RUNS_DIR / c / "best").exists():
                run_name = c
                break
        if run_name is None:
            all_bests = sorted(list(config.RUNS_DIR.glob("*/best")), key=lambda p: p.stat().st_mtime, reverse=True)
            safe_name = config.MODEL_NAME.split("/")[-1].lower()
            run_name = all_bests[0].parent.name if all_bests else f"dsi_{safe_name}"

    run_dir = config.RUNS_DIR / run_name
    if out_dir is None:
        out_dir = run_dir

    eval_data = load_eval_data(eval_path)
    if not eval_data:
        raise FileNotFoundError(f"No evaluation queries found at {eval_path}.")

    queries = [item["query"] for item in eval_data]
    gold_ids = [item["gold_chunk_id"] for item in eval_data]

    print("\n" + "=" * 60)
    print(f"EVALUATING ON {len(queries)} QUERIES ({eval_path.name})")
    print("=" * 60)

    # 1. Evaluate BM25
    print("\n[1/6] Evaluating BM25 Baseline...")
    bm25 = BM25Retriever()
    bm25_preds = [bm25.retrieve(q, k=10) for q in tqdm(queries, desc="BM25")]
    bm25_metrics = compute_retrieval_metrics(bm25_preds, gold_ids)

    # 2. Evaluate Dense (BGE)
    print("\n[2/6] Evaluating Dense Retrieval Baseline (BGE)...")
    dense = DenseRetriever()
    dense_preds = [dense.retrieve(q, k=10) for q in tqdm(queries, desc="Dense BGE")]
    dense_metrics = compute_retrieval_metrics(dense_preds, gold_ids)

    # 3. Evaluate Standard DSI Generative Retriever (without lookahead)
    print("\n[3/6] Evaluating Standard DSI Generative Retriever (No Lookahead)...")
    dsi_retriever = Retriever(run_dir=run_dir / "best")
    dsi_preds = [dsi_retriever.retrieve(q, k=10, use_pag=False) for q in tqdm(queries, desc="Standard DSI")]
    dsi_metrics = compute_retrieval_metrics(dsi_preds, gold_ids)

    # 4. Evaluate PAG (Planning-Ahead Guided Beam Search)
    print("\n[4/6] Evaluating SIGIR 2024 GenRet + PAG (Planning-Ahead Guided Beam Search)...")
    pag_preds = [dsi_retriever.retrieve(q, k=10, use_pag=True, pag_alpha=2.0) for q in tqdm(queries, desc="GenRet + PAG")]
    pag_metrics = compute_retrieval_metrics(pag_preds, gold_ids)

    # 5. Evaluate Hybrid Fusion (PAG + BM25)
    print("\n[5/6] Evaluating Hybrid Search (PAG + BM25 RRF Fusion)...")
    hybrid_preds = [fuse_rrf([p_p, b_p], k=10) for p_p, b_p in zip(pag_preds, bm25_preds)]
    hybrid_metrics = compute_retrieval_metrics(hybrid_preds, gold_ids)

    # 6. Evaluate Tri-Hybrid Fusion (PAG + BM25 + Dense BGE)
    print("\n[6/6] Evaluating Tri-Hybrid Search (PAG + BM25 + Dense BGE RRF Fusion)...")
    tri_hybrid_preds = [fuse_rrf([p_p, b_p, den_p], k=10) for p_p, b_p, den_p in zip(pag_preds, bm25_preds, dense_preds)]
    tri_hybrid_metrics = compute_retrieval_metrics(tri_hybrid_preds, gold_ids)

    with open(config.IDS_PATH, "r", encoding="utf-8") as f:
        ids_data = json.load(f)
    per_depth_acc = compute_per_depth_accuracy(pag_preds, gold_ids, ids_data["chunk_to_id"])

    # Summary Table
    print("\n" + "=" * 78)
    print(f"{'Method':<30} | {'Hits@1':<8} | {'Hits@5':<8} | {'Hits@10':<8} | {'MRR@10':<8}")
    print("-" * 78)
    print(f"{'BM25 (Lexical)':<30} | {bm25_metrics['hits@1']:<8.2%} | {bm25_metrics['hits@5']:<8.2%} | {bm25_metrics['hits@10']:<8.2%} | {bm25_metrics['mrr@10']:<8.4f}")
    print(f"{'Dense (BGE)':<30} | {dense_metrics['hits@1']:<8.2%} | {dense_metrics['hits@5']:<8.2%} | {dense_metrics['hits@10']:<8.2%} | {dense_metrics['mrr@10']:<8.4f}")
    print(f"{'GenRet (Standard DSI)':<30} | {dsi_metrics['hits@1']:<8.2%} | {dsi_metrics['hits@5']:<8.2%} | {dsi_metrics['hits@10']:<8.2%} | {dsi_metrics['mrr@10']:<8.4f}")
    print(f"{'🚀 GenRet + PAG (SIGIR 2024)':<30} | {pag_metrics['hits@1']:<8.2%} | {pag_metrics['hits@5']:<8.2%} | {pag_metrics['hits@10']:<8.2%} | {pag_metrics['mrr@10']:<8.4f}")
    print(f"{'⭐ Hybrid (PAG + BM25)':<30} | {hybrid_metrics['hits@1']:<8.2%} | {hybrid_metrics['hits@5']:<8.2%} | {hybrid_metrics['hits@10']:<8.2%} | {hybrid_metrics['mrr@10']:<8.4f}")
    print(f"{'🏆 Tri-Hybrid (PAG+BM25+Dense)':<30} | {tri_hybrid_metrics['hits@1']:<8.2%} | {tri_hybrid_metrics['hits@5']:<8.2%} | {tri_hybrid_metrics['hits@10']:<8.2%} | {tri_hybrid_metrics['mrr@10']:<8.4f}")
    print("=" * 78)

    print(f"\nGenRet + PAG Per-Depth Prefix Accuracy: {per_depth_acc}")

    # Save metrics.json
    metrics_payload = {
        "dataset": str(eval_path),
        "num_queries": len(queries),
        "bm25": bm25_metrics,
        "dense": dense_metrics,
        "genret_dsi": dsi_metrics,
        "genret_pag": pag_metrics,
        "hybrid_pag_bm25": hybrid_metrics,
        "tri_hybrid_all3": tri_hybrid_metrics,
        "per_depth_accuracy": per_depth_acc
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    # Save comprehensive failure analysis (Task 9)
    failures_file = out_dir / "failures.txt"
    all_failures = []
    chunk_fail_counts = Counter()

    for q, preds, gold in zip(queries, pag_preds, gold_ids):
        pred_id = preds[0]["chunk_id"] if preds else "NONE"
        if pred_id != gold:
            all_failures.append({
                "query": q,
                "pred_id": pred_id,
                "pred_id_str": preds[0].get("id_str", "") if preds else "",
                "gold_id": gold
            })
            chunk_fail_counts[gold] += 1

    rng = random.Random(config.SEED)
    sample_failures = rng.sample(all_failures, min(30, len(all_failures)))

    with open(failures_file, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write(f"FAILURE ANALYSIS SUMMARY (Total Errors: {len(all_failures)} / {len(queries)})\n")
        f.write("=" * 65 + "\n\n")
        f.write("Top 15 Most Frequently Failing Gold Chunks:\n")
        for cid, cnt in chunk_fail_counts.most_common(15):
            f.write(f"  • Chunk {cid}: {cnt} error(s)\n")
        f.write("\n" + "-" * 65 + "\n")
        f.write(f"Random Representative Sample of {len(sample_failures)} Failures (Across Full Corpus):\n")
        f.write("-" * 65 + "\n\n")
        for item in sample_failures:
            f.write(f"Query:           \"{item['query']}\"\n")
            f.write(f"Predicted Chunk: {item['pred_id']} (Semantic ID: {item['pred_id_str']})\n")
            f.write(f"Gold Chunk:      {item['gold_id']}\n")
            f.write("-" * 50 + "\n")

    print(f"\nSaved metrics to {out_dir / 'metrics.json'}")
    print(f"Saved comprehensive failure analysis ({len(all_failures)} total errors, {len(sample_failures)} sampled) to {failures_file}")

if __name__ == "__main__":
    evaluate_all()
