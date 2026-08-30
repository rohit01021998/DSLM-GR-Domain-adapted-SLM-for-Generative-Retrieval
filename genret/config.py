from pathlib import Path

# ==============================================================================
# PATHS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directory
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
EMB_PATH = DATA_DIR / "embeddings.npy"
EMB_META_PATH = DATA_DIR / "embeddings_meta.json"
IDS_PATH = DATA_DIR / "ids.json"
QUERIES_PATH = DATA_DIR / "queries.jsonl"
TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH = DATA_DIR / "val.jsonl"
TEST_MANUAL_PATH = DATA_DIR / "test_manual.jsonl"

# AutoSLM Corpus & Validation Paths
CORPUS_PATH = RAW_DIR / "corpus.txt"
OCR_CHUNKS_JSON_PATH = RAW_DIR / "chunks.json"
EVAL_RESULTS_PATH = DATA_DIR / "evaluation_results.json"

# Checkpoints and runs
RUNS_DIR = BASE_DIR / "runs"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# CHUNKING CONFIGURATION
# ==============================================================================
TARGET_TOKENS = 300
MIN_TOKENS = 120
MAX_TOKENS = 400
OVERLAP_RATIO = 0.15

# ==============================================================================
# EMBEDDING CONFIGURATION
# ==============================================================================
ENCODER_NAME = "BAAI/bge-large-en-v1.5"
EMB_BATCH_SIZE = 64
NORMALIZE = True

# ==============================================================================
# SEMANTIC IDS (HIERARCHICAL K-MEANS)
# ==============================================================================
BRANCHING = 5   # k in k-means (Depth 3-4 optimal tree)
LEAF_MAX = 2    # near-singleton leaves so every digit is a real semantic cluster decision (Task 5)
MAX_DEPTH = 6   # hard cap
DIGIT_TOKENS = [f"<d{i}>" for i in range(10)]  # ["<d0>", ..., "<d9>"]
ID_START_TOKEN = "<id>"
ID_END_TOKEN = "</id>"

# ==============================================================================
# QUERY GENERATION
# ==============================================================================
QUERIES_PER_CHUNK = 24
QUERY_STYLE_MIX = {"question": 0.4, "acronym_grounded": 0.3, "keyword": 0.15, "long": 0.15}
GEN_BASE_URL = "http://localhost:8000/v1"
GEN_MODEL_NAME = "sasa2000/cosmos-reason2-2b-text-only"

def get_active_vllm_model(base_url: str = GEN_BASE_URL, default: str = GEN_MODEL_NAME) -> str:
    """Auto-detect the active model currently served by the local vLLM instance."""
    try:
        import requests
        r = requests.get(f"{base_url}/models", timeout=1.5)
        if r.status_code == 200:
            data = r.json().get("data", [])
            if data and "id" in data[0]:
                return data[0]["id"]
    except Exception:
        pass
    return default

# ==============================================================================
# DATASET CREATION
# ==============================================================================
INDEX_TO_QUERY_RATIO = 0.5  # 1 indexing example per 2 query examples
VAL_QUERY_FRACTION = 0.2

# ==============================================================================
# MODEL & TRAINING HYPERPARAMETERS
# ==============================================================================
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
LR = 2e-4
BATCH_SIZE = 4
GRAD_ACCUM = 16
EPOCHS = 20
WARMUP_RATIO = 0.05
MAX_INPUT_TOKENS = 384
SEED = 42

# ==============================================================================
# INFERENCE CONFIGURATION
# ==============================================================================
BEAM_WIDTH = 25
TOP_K = 10
PAG_ALPHA = 10.0
