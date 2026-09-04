import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import re
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import Counter
import torch

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
from transformers import LogitsProcessorList
from genret.trie import IDTrie, load_trie, make_prefix_allowed_tokens_fn
from genret.embed import load_chunks_jsonl
from genret.baselines import BM25Retriever
from genret.infer import PAGLogitsProcessor, rewrite_query_for_retrieval


def load_corpus_chunks() -> Dict[str, Dict[str, Any]]:
    """Loads text chunks for display."""
    chunks_map = {}
    if config_moe.CHUNKS_PATH.exists():
        with open(config_moe.CHUNKS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    cid = item.get("chunk_id", item.get("id"))
                    chunks_map[cid] = item
    return chunks_map


def reset_routing_stats(model: torch.nn.Module):
    """Zeroes out the routing statistics counters."""
    for layer in model.model.layers:
        mlp = layer.mlp
        if hasattr(mlp, "router") and hasattr(mlp.router, "expert_counts"):
            mlp.router.expert_counts.zero_()


def inspect_query_routing(model: torch.nn.Module) -> List[Dict[str, Any]]:
    """Inspects the aggregated routing distribution since the last reset."""
    routing_profile = []
    for idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        if hasattr(mlp, "router") and hasattr(mlp.router, "expert_counts"):
            counts = mlp.router.expert_counts.cpu().tolist()
            if sum(counts) > 0:
                expert_dict = {str(i): c for i, c in enumerate(counts) if c > 0}
                primary_expert = max(expert_dict.keys(), key=lambda k: expert_dict[k])
                routing_profile.append({
                    "layer": idx,
                    "primary_expert": int(primary_expert),
                    "expert_counts": expert_dict
                })
    return routing_profile



class MoERetriever:
    """
    MoE Generative Retrieval Inference Engine (Top-2 Sub-Dense MoE with PAG & Hybrid BM25).
    Given a query, generates prefix-constrained semantic IDs using the trained MoE model.
    """
    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        chunks_path: Path = config_moe.CHUNKS_PATH,
        ids_path: Path = config_moe.IDS_PATH,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if checkpoint_dir is None:
            if config_moe.BEST_CHECKPOINT_DIR.exists() and (
                (config_moe.BEST_CHECKPOINT_DIR / "model.safetensors").exists() or
                (config_moe.BEST_CHECKPOINT_DIR / "moe_weights.pt").exists()
            ):
                checkpoint_dir = config_moe.BEST_CHECKPOINT_DIR
            elif (config_moe.CHECKPOINTS_DIR / "last").exists():
                checkpoint_dir = config_moe.CHECKPOINTS_DIR / "last"
            else:
                checkpoint_dir = config_moe.BEST_CHECKPOINT_DIR

        print(f"Loading MoE DSI Retriever from: {checkpoint_dir}")
        self.tokenizer, self.model = load_moe_checkpoint(checkpoint_dir, device=self.device)
        self.model.eval()

        with open(ids_path, "r", encoding="utf-8") as f:
            ids_data = json.load(f)
        self.id_to_chunk = ids_data.get("id_to_chunk", {})
        self.chunk_to_id = ids_data.get("chunk_to_id", {})
        self.trie = load_trie(ids_path)

        chunks_list = load_chunks_jsonl(chunks_path)
        self.chunks_dict = {c["chunk_id"]: c for c in chunks_list}

        self.bm25 = BM25Retriever(chunks_path)

        self.digit_token_ids = {i: self.tokenizer.convert_tokens_to_ids(f"<d{i}>") for i in range(10)}
        self.id_start_token_id = self.tokenizer.convert_tokens_to_ids(config_moe.ID_START_TOKEN)
        self.id_end_token_id = self.tokenizer.convert_tokens_to_ids(config_moe.ID_END_TOKEN)

        self.prefix_allowed_fn = make_prefix_allowed_tokens_fn(
            trie=self.trie,
            tokenizer=self.tokenizer,
            digit_token_ids=self.digit_token_ids,
            id_start_token_id=self.id_start_token_id,
            id_end_token_id=self.id_end_token_id
        )
        self.last_routing: List[Dict[str, Any]] = []

    def retrieve(
        self,
        query: str,
        k: int = 10,
        beam_width: int = 25,
        use_pag: bool = True,
        pag_alpha: float = 10.0,
        early_stopping: bool = False,
        use_hybrid: bool = True
    ) -> List[Dict[str, Any]]:
        reset_routing_stats(self.model)
        prompt = f"retrieve: {query}\n"
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        input_len = inputs["input_ids"].shape[1]

        candidate_pool_size = max(k * 2, 20)
        num_beams = max(beam_width, candidate_pool_size)
        num_return = min(candidate_pool_size * 2, num_beams)

        logits_processor = None
        if use_pag:
            doc_priors = self.bm25.get_all_scores(query)
            pag_proc = PAGLogitsProcessor(
                trie=self.trie,
                digit_token_ids=self.digit_token_ids,
                id_start_token_id=self.id_start_token_id,
                id_end_token_id=self.id_end_token_id,
                doc_priors=doc_priors,
                alpha=pag_alpha
            )
            logits_processor = LogitsProcessorList([pag_proc])

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                num_beams=num_beams,
                num_return_sequences=num_return,
                prefix_allowed_tokens_fn=self.prefix_allowed_fn,
                logits_processor=logits_processor,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=early_stopping
            )

        self.last_routing = inspect_query_routing(self.model)

        sequences = outputs.sequences
        seq_scores = outputs.sequences_scores if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None else [0.0] * len(sequences)

        dsi_ranked_chunks = []
        seen_dsi_ids = set()

        for (seq, score) in zip(sequences, seq_scores):
            gen_tokens = seq[input_len:]
            gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=False)

            digits = [int(d) for d in re.findall(r"<d(\d)>", gen_text)]
            id_str = "-".join(map(str, digits))

            chunk_id = self.id_to_chunk.get(id_str)
            if chunk_id and chunk_id not in seen_dsi_ids and chunk_id in self.chunks_dict:
                seen_dsi_ids.add(chunk_id)
                chunk_obj = self.chunks_dict[chunk_id]
                raw_id = self.chunk_to_id.get(chunk_id, [])
                if isinstance(raw_id, list):
                    id_digits = raw_id
                    id_string = "-".join(map(str, id_digits))
                else:
                    id_string = str(raw_id)
                    id_digits = [int(x) for x in id_string.split("-")] if id_string else []

                dsi_ranked_chunks.append({
                    "chunk_id": chunk_id,
                    "id_path": id_digits,
                    "id_str": id_string,
                    "score": float(score),
                    "text": chunk_obj.get("text", chunk_obj.get("page_content", ""))
                })
                if len(dsi_ranked_chunks) >= candidate_pool_size:
                    break

        candidate_chunks = dsi_ranked_chunks[:candidate_pool_size]
        if not candidate_chunks:
            return []

        if not use_hybrid:
            results = []
            for rank, c in enumerate(candidate_chunks[:k], start=1):
                c_copy = dict(c)
                c_copy["rank"] = rank
                results.append(c_copy)
            return results

        # Stage 2: In-Candidate Lexical Reranker (O(k) complexity)
        lexical_scores = self.bm25.score_candidates(query, candidate_chunks)
        sorted_by_lexical = sorted(candidate_chunks, key=lambda c: lexical_scores.get(c["chunk_id"], 0.0), reverse=True)
        in_candidate_bm25_rank = {c["chunk_id"]: rank for rank, c in enumerate(sorted_by_lexical, start=1)}
        dsi_rank_map = {c["chunk_id"]: rank for rank, c in enumerate(candidate_chunks, start=1)}

        rrf_k = 10
        w_dsi = 1.0
        w_bm25 = 1.0

        fused_scores = {}
        for c in candidate_chunks:
            cid = c["chunk_id"]
            fused_score = (w_dsi / (rrf_k + dsi_rank_map[cid])) + (w_bm25 / (rrf_k + in_candidate_bm25_rank[cid]))
            fused_scores[cid] = fused_score

        sorted_candidates = sorted(candidate_chunks, key=lambda c: fused_scores[c["chunk_id"]], reverse=True)[:k]

        results = []
        for rank, c in enumerate(sorted_candidates, start=1):
            cid = c["chunk_id"]
            c_copy = dict(c)
            c_copy["rank"] = rank
            c_copy["score"] = round(fused_scores[cid], 5)
            results.append(c_copy)
        return results


def infer_single_query(
    query: str,
    checkpoint_dir: Optional[Path] = None,
    num_beams: int = 10,
    top_k: int = 3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 65)
    print(f"MoE DSI INFERENCE FOR QUERY: \"{query}\"")
    print("=" * 65)

    def _has_trained_weights(d: Path) -> bool:
        return (d / "model.safetensors").exists() or (d / "moe_weights.pt").exists()

    # 1. Load model
    if checkpoint_dir and _has_trained_weights(checkpoint_dir):
        tokenizer, model = load_moe_checkpoint(checkpoint_dir, device=device)
    elif _has_trained_weights(config_moe.BEST_CHECKPOINT_DIR):
        tokenizer, model = load_moe_checkpoint(config_moe.BEST_CHECKPOINT_DIR, device=device)
    else:
        print(f"Loading and dynamically upcycling base checkpoint ({config_moe.SOURCE_CHECKPOINT_DIR})...")
        tokenizer, model = load_and_convert_model(device=device)

    model.eval()

    # 2. Setup Trie
    with open(config_moe.IDS_PATH, "r", encoding="utf-8") as f:
        ids_data = json.load(f)

    chunk_to_id = ids_data["chunk_to_id"]
    id_to_chunk = ids_data.get("id_to_chunk", {})
    trie = IDTrie(chunk_to_id)
    chunks_map = load_corpus_chunks()

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

    prompt = f"retrieve: {query}\n"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            num_beams=num_beams,
            num_return_sequences=min(top_k, num_beams),
            prefix_allowed_tokens_fn=prefix_fn,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode results
    input_len = inputs["input_ids"].shape[1]
    print(f"\nTop-{len(outputs)} Retrieved Semantic IDs:")
    for rank, seq in enumerate(outputs, 1):
        gen_tokens = seq[input_len:]
        id_str = tokenizer.decode(gen_tokens, skip_special_tokens=False).strip()
        digits = tuple(int(d) for d in re.findall(r"<d(\d)>", id_str))
        digits_key = "-".join(str(d) for d in digits)
        chunk_id = id_to_chunk.get(digits_key, "UNKNOWN")

        print(f"\n[{rank}] Semantic ID: {id_str} -> Chunk ID: {chunk_id}")
        chunk_data = chunks_map.get(chunk_id)
        if chunk_data:
            section = chunk_data.get("section", "N/A")
            content = chunk_data.get("text", chunk_data.get("page_content", ""))[:200]
            print(f"    Section: {section}")
            print(f"    Snippet: \"{content}...\"")

    # Layer-by-layer routing inspection
    routing = inspect_query_routing(model)
    if routing:
        print("\n--- MoE Layer Routing Profile for Query ---")
        layer_summary = [f"L{r['layer']}:E{r['primary_expert']}" for r in routing]
        print(" -> ".join(layer_summary[:12]))
        if len(layer_summary) > 12:
            print(" -> ".join(layer_summary[12:]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single query through MoE model")
    parser.add_argument("--query", "-q", type=str, default=None, help="Query string")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to MoE checkpoint")
    parser.add_argument("--beams", type=int, default=10, help="Number of beams")
    parser.add_argument("--top-k", type=int, default=3, help="Number of returned candidates")
    args = parser.parse_args()

    q = args.query
    if not q:
        q = input("Enter query: ").strip()

    if q:
        ckpt = Path(args.checkpoint) if args.checkpoint else None
        infer_single_query(q, checkpoint_dir=ckpt, num_beams=args.beams, top_k=args.top_k)
