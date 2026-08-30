import argparse
import sys
from pathlib import Path

def run_ocr_postprocessing():
    import json
    import os
    from utils.ocr_postprocessing import OCRPostProcessor
    
    input_file = "data/raw/ocr_extraction.json"
    output_file = "data/raw/chunks.json"
    
    print("\n" + "=" * 60)
    print("STEP 1: OCR POST-PROCESSING")
    print("=" * 60)
    print(f"Loading and processing {input_file}...")
    
    processor = OCRPostProcessor(max_chunk_size=1500)
    chunks = processor.process_file(input_file)
    print(f"Successfully generated {len(chunks)} chunks.")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved chunks to {output_file}.\n")

def run_prepare_corpus():
    from utils.prepare_corpus import main as prepare_corpus_main
    print("\n" + "=" * 60)
    print("STEP 2: PREPARE CORPUS TEXT")
    print("=" * 60)
    prepare_corpus_main()
    print()

def run_chunking_evaluation():
    from utils.chunking_evalution import main as evaluate_main
    print("\n" + "=" * 60)
    print("STEP 3: CHUNKING STRATEGY BENCHMARK (vLLM)")
    print("=" * 60)
    evaluate_main()
    print()

def run_apply_chunking():
    from utils.apply_chunking import main as chunking_main
    print("\n" + "=" * 60)
    print("STEP 4: VALIDATION-DRIVEN CHUNKING (data/chunks.jsonl)")
    print("=" * 60)
    chunking_main()
    print()

def run_embed():
    from genret.embed import embed_chunks, load_chunks_jsonl, run_acceptance_test
    print("\n" + "=" * 60)
    print("STEP 5: DENSE EMBEDDING GENERATION (data/embeddings.npy)")
    print("=" * 60)
    emb, meta = embed_chunks()
    chunks = load_chunks_jsonl()
    run_acceptance_test(emb, meta, chunks)
    print()

def run_semantic_ids():
    from genret.semantic_ids import main as semantic_ids_main
    print("\n" + "=" * 60)
    print("STEP 6: HIERARCHICAL SEMANTIC IDS (data/ids.json)")
    print("=" * 60)
    semantic_ids_main()
    print()

def run_gen_queries(limit=None):
    from genret.gen_queries import run_query_generation, run_acceptance_test
    print("\n" + "=" * 60)
    print("STEP 7: SYNTHETIC QUERY GENERATION (data/queries.jsonl via vLLM)")
    print("=" * 60)
    run_query_generation(limit=limit)
    run_acceptance_test()
    print()

def run_build_dataset():
    from genret.build_dataset import build_datasets, load_chunks_jsonl, run_acceptance_test
    print("\n" + "=" * 60)
    print("STEP 8: DUAL-TASK DATASET CREATION (data/train.jsonl & val.jsonl)")
    print("=" * 60)
    chunks = load_chunks_jsonl()
    train_data, val_data = build_datasets()
    run_acceptance_test(train_data, val_data, chunks)
    print()

def run_train():
    from genret.train import train_dsi
    print("\n" + "=" * 60)
    print("STEP 9: QWEN DSI MODEL TRAINING")
    print("=" * 60)
    train_dsi()
    print()

def run_eval():
    from genret.eval import evaluate_all
    print("\n" + "=" * 60)
    print("STEP 10: COMPARATIVE EVALUATION (DSI vs BM25 vs Dense)")
    print("=" * 60)
    evaluate_all()
    print()

def run_ask(query: str = None):
    from genret.infer import Retriever
    if not query:
        query = input("Enter your question: ").strip()
    if not query:
        return
    retriever = Retriever()
    qa_result = retriever.answer(query, top_k=10, use_pag=True, rewrite_query=True)
    print("\n" + "=" * 65)
    print(f"ORIGINAL QUERY: \"{query}\"")
    if qa_result.get("search_query") and qa_result["search_query"] != query:
        print(f"OPTIMIZED SEARCH QUERY: \"{qa_result['search_query']}\"")
    print("=" * 65)
    print(f"\n💡 ANSWER:\n{qa_result['answer']}\n")
    print("-" * 65)
    print("📚 RETRIEVED GROUNDING SOURCES (DSI + PAG):")
    for src in qa_result["sources"]:
        print(f"  • Rank #{src['rank']} | Chunk {src['chunk_id']} [ID: {src['id_str']}] (Score: {src['score']:.4f})")
        print(f"    \"{src['snippet']}\"")
    print("=" * 65 + "\n")

def run_chat():
    from genret.infer import Retriever
    retriever = Retriever()
    print("\n" + "=" * 65)
    print("🤖 DSLM-GR Interactive Assistant [DSI + PAG Enabled] (Type 'exit' to quit)")
    print("=" * 65)
    while True:
        try:
            print("\n🧑 You (Type your question, then press Enter TWICE to submit, or type 'exit'):")
            lines = []
            while True:
                try:
                    line = input("   | " if lines else "   > ")
                    if line.strip() == "" and lines:
                        break
                    if line.strip() == "" and not lines:
                        continue
                    if line.strip().lower() in ["exit", "quit", "q"]:
                        if not lines:
                            print("Goodbye!")
                            return
                    lines.append(line)
                except EOFError:
                    break
            
            query = "\n".join(lines).strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                return

            qa_result = retriever.answer(query, top_k=10, use_pag=True, rewrite_query=True)
            if qa_result.get("search_query") and qa_result["search_query"] != query:
                print(f"🔍 [Query Expanded for DSI+PAG: \"{qa_result['search_query']}\"]")
            print(f"\n💡 DSLM-GR:\n{qa_result['answer']}\n")
            print(f"[Citations: {', '.join([f'Chunk {s['chunk_id']} (ID: {s['id_str']})' for s in qa_result['sources']])}]")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

def run_dsi_pipeline():
    """Run full DSI generative retrieval pipeline from chunks to evaluation."""
    print("\n🚀 Starting End-to-End DSI Generative Retrieval Pipeline...")
    run_apply_chunking()
    run_embed()
    run_semantic_ids()
    run_gen_queries()
    run_build_dataset()
    run_train()
    run_eval()
    print("\n🎉 Complete DSI Pipeline Finished Successfully!")

def run_all():
    """Run full repository pipeline from raw OCR to final DSI evaluation."""
    print("\n🚀 Starting Full DSLM-GR Pipeline (Raw OCR -> Trained DSI Evaluator)...")
    run_ocr_postprocessing()
    run_prepare_corpus()
    run_chunking_evaluation()
    run_dsi_pipeline()

def main():
    parser = argparse.ArgumentParser(description="DSLM-GR Generative Retrieval Orchestrator")
    parser.add_argument(
        "step", 
        nargs="?",
        default="chat",
        choices=[
            "ocr", "corpus", "evaluate", "chunk", "embed", 
            "semantic-ids", "ids", "queries", "dataset", "train", "eval",
            "dsi", "pipeline", "all", "ask", "chat"
        ], 
        help=(
            "Pipeline step to run: "
            "'ask' (ask single question with grounding), "
            "'chat' (interactive conversation terminal), "
            "'dsi' / 'pipeline' (run complete DSI pipeline), "
            "'all' (run everything from raw OCR to final trained model), "
            "or individual steps: 'ocr', 'corpus', 'evaluate', 'chunk', 'embed', 'semantic-ids', 'queries', 'dataset', 'train', 'eval'."
        )
    )
    parser.add_argument("--query", "-q", type=str, default=None, help="Question query for 'ask' command")
    
    args = parser.parse_args()
    
    if args.step == "ask":
        run_ask(args.query)
        return
    elif args.step == "chat":
        run_chat()
        return

    dispatch = {
        "ocr": run_ocr_postprocessing,
        "corpus": run_prepare_corpus,
        "evaluate": run_chunking_evaluation,
        "chunk": run_apply_chunking,
        "embed": run_embed,
        "semantic-ids": run_semantic_ids,
        "ids": run_semantic_ids,
        "queries": run_gen_queries,
        "dataset": run_build_dataset,
        "train": run_train,
        "eval": run_eval,
        "dsi": run_dsi_pipeline,
        "pipeline": run_dsi_pipeline,
        "all": run_all
    }

    action = dispatch.get(args.step)
    if action:
        action()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
