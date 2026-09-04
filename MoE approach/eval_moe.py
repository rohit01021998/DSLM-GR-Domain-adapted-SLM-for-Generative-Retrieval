import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import re
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict
import torch
from tqdm import tqdm

# Ensure MoE approach directory and repo root are on path
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config_moe
from convert_to_moe import load_moe_checkpoint, load_and_convert_model
from moe_layer import SharedAndRoutedMoEBlock, ClassicSparseMoEBlock, SubDenseSparseMoEBlock
from genret.trie import IDTrie, make_prefix_allowed_tokens_fn


def load_eval_queries(eval_path: Path) -> List[Dict[str, Any]]:
    """Loads evaluation items from JSONL."""
    records = []
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_path}")

    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                query = item["input"].replace("retrieve: ", "").replace("index: ", "").strip()
                task = item.get("task", "retrieve")
                records.append({
                    "raw_input": item["input"],
                    "query": query,
                    "task": task,
                    "target": item["target"],
                    "gold_chunk_id": item.get("chunk_id", None)
                })
    return records


def compute_retrieval_metrics(
    predictions: List[List[Dict[str, Any]]],
    gold_chunk_ids: List[str]
) -> Dict[str, float]:
    """Computes Hits@1, Hits@5, Hits@10, MRR@10."""
    hits1 = 0
    hits5 = 0
    hits10 = 0
    mrr10 = 0.0

    valid_pairs = [(preds, gold) for preds, gold in zip(predictions, gold_chunk_ids) if gold is not None]
    n = len(valid_pairs)
    if n == 0:
        return {"hits@1": 0.0, "hits@5": 0.0, "hits@10": 0.0, "mrr@10": 0.0}

    for preds, gold in valid_pairs:
        pred_ids = [p["chunk_id"] for p in preds if "chunk_id" in p]

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


def collect_routing_stats(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Collects expert selection frequency across all MoE layers.
    """
    layer_stats = {}
    total_expert_counts = Counter()

    for idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        if isinstance(mlp, (SharedAndRoutedMoEBlock, ClassicSparseMoEBlock, SubDenseSparseMoEBlock)):
            counts = mlp.router.expert_counts.cpu().tolist()
            layer_stats[f"layer_{idx}"] = {
                f"expert_{i}": counts[i] for i in range(len(counts))
            }
            for i, c in enumerate(counts):
                total_expert_counts[f"expert_{i}"] += c

    return {
        "total_expert_counts": dict(total_expert_counts),
        "layer_breakdown": layer_stats
    }


def evaluate_moe(
    checkpoint_dir: Optional[Path] = None,
    eval_path: Path = config_moe.VAL_PATH,
    sample_limit: Optional[int] = None,
    num_beams: int = 10
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 65)
    print("EVALUATING MoE DSI RETRIEVAL MODEL")
    print("=" * 65)

    # 1. Load model
    if checkpoint_dir and (
        (checkpoint_dir / "model.safetensors").exists() or
        (checkpoint_dir / "moe_weights.pt").exists()
    ):
        tokenizer, model = load_moe_checkpoint(checkpoint_dir, device=device)
    else:
        print(f"Loading and converting directly from base checkpoint ({config_moe.SOURCE_CHECKPOINT_DIR})...")
        tokenizer, model = load_and_convert_model(device=device)

    model.eval()

    # Reset expert count buffers for evaluation
    for module in model.modules():
        if hasattr(module, "expert_counts"):
            module.expert_counts.zero_()

    # 2. Load ID maps and build Trie
    with open(config_moe.IDS_PATH, "r", encoding="utf-8") as f:
        ids_data = json.load(f)

    chunk_to_id = ids_data["chunk_to_id"]
    id_to_chunk = ids_data.get("id_to_chunk", {})
    trie = IDTrie(chunk_to_id)

    # 3. Setup prefix allowed tokens function
    digit_token_ids = {i: tokenizer.convert_tokens_to_ids(f"<d{i}>") for i in range(10)}
    id_start_token_id = tokenizer.convert_tokens_to_ids(config_moe.ID_START_TOKEN)
    id_end_token_id = tokenizer.convert_tokens_to_ids(config_moe.ID_END_TOKEN)

    prefix_fn = make_prefix_allowed_tokens_fn(
        trie=trie,
        tokenizer=tokenizer,
        digit_token_ids=digit_token_ids,
        id_start_token_id=id_start_token_id,
        id_end_token_id=id_end_token_id
    )

    # 4. Load evaluation dataset
    records = load_eval_queries(eval_path)
    if sample_limit:
        random.seed(config_moe.SEED)
        records = random.sample(records, min(sample_limit, len(records)))
        print(f"Subsampled {len(records)} queries for evaluation.")
    else:
        print(f"Evaluating full set of {len(records)} queries.")

    predictions = []
    gold_ids = []

    print("\nRunning prefix-constrained beam search generation...")
    for item in tqdm(records, desc="MoE Evaluation"):
        prompt = item["raw_input"] + "\n"
        gold_ids.append(item["gold_chunk_id"])

        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                num_beams=num_beams,
                num_return_sequences=min(10, num_beams),
                prefix_allowed_tokens_fn=prefix_fn,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        query_preds = []
        input_len = inputs["input_ids"].shape[1]
        for seq in outputs:
            gen_tokens = seq[input_len:]
            id_str = tokenizer.decode(gen_tokens, skip_special_tokens=False).strip()
            digits = tuple(int(d) for d in re.findall(r"<d(\d)>", id_str))
            
            # Map digits to chunk_id (keys are dash-separated like '2-3-0-4-0')
            digits_key = "-".join(str(d) for d in digits)
            chunk_id = id_to_chunk.get(digits_key, None)
            query_preds.append({
                "id_str": id_str,
                "digits": digits,
                "chunk_id": chunk_id
            })

        predictions.append(query_preds)

    # 5. Compute metrics
    metrics = compute_retrieval_metrics(predictions, gold_ids)
    routing_stats = collect_routing_stats(model)

    # Print summary table
    print("\n" + "=" * 65)
    print("MoE RETRIEVAL PERFORMANCE SUMMARY")
    print("=" * 65)
    print(f"Hits@1:   {metrics['hits@1']:.2%}")
    print(f"Hits@5:   {metrics['hits@5']:.2%}")
    print(f"Hits@10:  {metrics['hits@10']:.2%}")
    print(f"MRR@10:   {metrics['mrr@10']:.4f}")
    print("=" * 65)

    print("\n--- MoE Expert Utilization Statistics ---")
    total_tokens = sum(routing_stats["total_expert_counts"].values())
    if total_tokens > 0:
        for exp, count in sorted(routing_stats["total_expert_counts"].items()):
            pct = count / total_tokens
            print(f"• {exp}: {count:,} token activations ({pct:.1%})")

    # Save metrics
    out_file = config_moe.CHECKPOINTS_DIR / "eval_results.json"
    payload = {
        "metrics": metrics,
        "routing_stats": routing_stats,
        "evaluated_queries": len(records)
    }
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved evaluation metrics and routing stats -> {out_file}")

    return metrics, routing_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MoE Model on Retrieval")
    parser.add_argument("--checkpoint", type=str, default=str(config_moe.BEST_CHECKPOINT_DIR), help="Path to MoE checkpoint")
    parser.add_argument("--eval-path", type=str, default=str(config_moe.VAL_PATH), help="Path to evaluation dataset")
    parser.add_argument("--limit", type=int, default=None, help="Sample limit for fast eval")
    parser.add_argument("--beams", type=int, default=10, help="Beam search width")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else None
    evaluate_moe(
        checkpoint_dir=ckpt,
        eval_path=Path(args.eval_path),
        sample_limit=args.limit,
        num_beams=args.beams
    )
