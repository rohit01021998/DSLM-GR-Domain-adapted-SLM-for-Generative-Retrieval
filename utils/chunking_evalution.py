import os
import json
from chunking_evaluation import BaseChunker, SyntheticEvaluation, GeneralEvaluation
from chunking_evaluation.chunking import ClusterSemanticChunker
from chromadb.utils import embedding_functions

class FixedSizeChunker(BaseChunker):
    def __init__(self, chunk_size=1500, overlap=150):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str) -> list[str]:
        chunks = []
        i = 0
        while i < len(text):
            chunks.append(text[i:i + self.chunk_size])
            i += self.chunk_size - self.overlap
        return chunks

class DocumentNativeChunker(BaseChunker):
    def __init__(self, delimiter='\n\n<---CHUNK_BOUNDARY--->\n\n'):
        self.delimiter = delimiter

    def split_text(self, text: str) -> list[str]:
        # Splits perfectly along the boundaries we created during corpus prep
        # recreating the output from OCRPostProcessor
        return text.split(self.delimiter)

def main():
    # Configure for local vLLM server
    vllm_base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    vllm_gen_model = os.environ.get("VLLM_GEN_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
    vllm_emb_model = os.environ.get("VLLM_EMB_MODEL", "BAAI/bge-large-en-v1.5")
    
    # The OpenAI client under the hood reads OPENAI_BASE_URL
    os.environ["OPENAI_BASE_URL"] = vllm_base_url
    
    # We still need a dummy key so the OpenAI client doesn't crash
    dummy_key = "sk-fake-vllm-key"
    os.environ["OPENAI_API_KEY"] = dummy_key

    from genret import config

    corpus_path = config.CORPUS_PATH if config.CORPUS_PATH.exists() else (config.DATA_DIR / "corpus.txt")
    corpora_paths = [str(corpus_path)]
    queries_csv_path = str(config.DATA_DIR / "generated_queries_excerpts.csv")
    
    print(f"Initializing SyntheticEvaluation with local vLLM model: {vllm_gen_model}")
    print(f"Base URL: {vllm_base_url}")
    
    evaluation = SyntheticEvaluation(
        corpora_paths=corpora_paths, 
        queries_csv_path=queries_csv_path, 
        openai_api_key=dummy_key,
        model=vllm_gen_model
    )
    
    if not os.path.exists(queries_csv_path):
        print("Generating synthetic queries and excerpts (25 queries total)...")
        evaluation.generate_queries_and_excerpts(approximate_excerpts=True, num_rounds=5, queries_per_corpus=5)
        # Apply recommended filters
        try:
            evaluation.filter_poor_excerpts(threshold=0.36)
            evaluation.filter_duplicates(threshold=0.6)
        except Exception as e:
            print(f"Skipping OpenAI-specific filters: {e}")
    else:
        print(f"Using existing queries from {queries_csv_path}")

    print(f"Setting up local SentenceTransformer Embedding function ({vllm_emb_model})...")
    default_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=vllm_emb_model
    )

    chunkers = {
        "Fixed Size Chunker (1500 chars)": FixedSizeChunker(chunk_size=1500, overlap=150),
        "Cluster Semantic Chunker (max 1500 chars)": ClusterSemanticChunker(default_ef, max_chunk_size=1500),
        "Document Native Chunker (OCRPostProcessor)": DocumentNativeChunker()
    }

    print("\n--- Running Evaluations ---")
    results_summary = {}
    for name, chunker in chunkers.items():
        print(f"\nEvaluating {name}...")
        try:
            res = evaluation.run(chunker, embedding_function=default_ef)
            results_summary[name] = {
                "iou_mean": float(res["iou_mean"]),
                "recall_mean": float(res["recall_mean"]),
                "precision_mean": float(res["precision_mean"]),
                "precision_omega_mean": float(res["precision_omega_mean"])
            }
            print(f"Results for {name}: {results_summary[name]}")
        except Exception as e:
            print(f"Error evaluating {name}: {e}")

    results_path = str(config.EVAL_RESULTS_PATH)
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved evaluation metrics report to {results_path}")

    # Determine winner based on combined Precision Omega and IoU
    if results_summary:
        winner = max(results_summary.keys(), key=lambda k: results_summary[k]["precision_omega_mean"] + results_summary[k]["iou_mean"])
        print(f"🏆 Winning Chunking Strategy: {winner}")

if __name__ == "__main__":
    main()
