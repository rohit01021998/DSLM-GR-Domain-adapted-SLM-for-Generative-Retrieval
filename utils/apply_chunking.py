import os
import json
from pathlib import Path
from genret import config
from genret.chunking import chunk_corpus, get_tokenizer, run_acceptance_test

def get_winning_strategy(results_path: Path = config.EVAL_RESULTS_PATH) -> str:
    """
    Read benchmark evaluation report and determine the winning chunking strategy.
    """
    if not results_path.exists():
        print(f"No prior evaluation results found at {results_path}. Defaulting to Document Native Chunker.")
        return "Document Native Chunker (OCRPostProcessor)"
    
    with open(results_path, "r") as f:
        results = json.load(f)
    
    if not results:
        return "Document Native Chunker (OCRPostProcessor)"
    
    # Winner based on combined Precision Omega and IoU score
    winner = max(results.keys(), key=lambda k: results[k].get("precision_omega_mean", 0) + results[k].get("iou_mean", 0))
    print(f"📊 Validation Report loaded from {results_path}")
    print(f"🏆 Winning Strategy Identified: '{winner}' (Precision Omega: {results[winner].get('precision_omega_mean', 0):.2%}, IoU: {results[winner].get('iou_mean', 0):.2%})")
    return winner

def main():
    print("--- Applying Winning Chunking Strategy to Corpus ---")
    winner = get_winning_strategy()
    print(f"Using strategy: {winner}")

    # Generate standardized data/chunks.jsonl
    chunks = chunk_corpus(raw_dir=config.RAW_DIR, out_path=config.CHUNKS_PATH)
    
    # Run acceptance verification
    tokenizer = get_tokenizer()
    run_acceptance_test(chunks, tokenizer)

if __name__ == "__main__":
    main()
