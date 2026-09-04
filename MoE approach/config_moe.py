from pathlib import Path
import torch

# Base paths
MOE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MOE_DIR.parent

# Base trained dense model checkpoint
SOURCE_CHECKPOINT_DIR = REPO_ROOT / "runs" / "dsi_smollm2-1.7b-instruct" / "best"
BASE_MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# MoE Checkpoints and outputs
CHECKPOINTS_DIR = MOE_DIR / "checkpoints"
INITIAL_MOE_DIR = CHECKPOINTS_DIR / "initial_moe"
BEST_CHECKPOINT_DIR = CHECKPOINTS_DIR / "best"
LAST_CHECKPOINT_DIR = CHECKPOINTS_DIR / "last"

# Datasets
DATA_DIR = REPO_ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH = DATA_DIR / "val.jsonl"
IDS_PATH = DATA_DIR / "ids.json"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"

# Ensure output directories exist
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
INITIAL_MOE_DIR.mkdir(parents=True, exist_ok=True)
BEST_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# MoE ARCHITECTURE CONFIGURATION
# ==============================================================================
# Architecture type:
# - "sub_dense_sparse" (Narrower expert MLPs + Top-1 routing -> 1.21B Active Params < 1.71B dense!)
# - "classic_sparse" (Standard 8192-dim MLPs + Top-1 routing -> 1.71B Active Params = 1.71B dense)
# - "shared_and_routed" (1 Shared + 3 Routed -> 2.31B Active Params)
MOE_STYLE = "sub_dense_sparse"

# Sub-dense settings (1.71B Active Parameters matching Dense Baseline):
NUM_EXPERTS = 4                  # 4 experts per MoE layer
TOP_K = 2                        # Top-2 routing -> 2 * 4096 = 8192 active intermediate size = 1.71B active params!
EXPERT_INTERMEDIATE_SIZE = 4096  # Half of dense 8192 intermediate size
MOE_LAYER_INDICES = None         # None = all 24 layers converted (1.71B active params per token)

# Compatibility aliases
NUM_ROUTED_EXPERTS = NUM_EXPERTS
TOP_K_ROUTED = TOP_K
CLASSIC_NUM_EXPERTS = NUM_EXPERTS
CLASSIC_TOP_K = TOP_K

# Symmetry breaking initialization noise std for duplicated expert weights
EXPERT_INIT_NOISE_STD = 0.01

# Router settings
ROUTER_JITTER_NOISE = 0.01  # Noise added to router logits during training for exploration
AUX_LOSS_COEF = 0.01        # Coefficient for load-balancing loss (\lambda_aux)

# ==============================================================================
# TRAINING HYPERPARAMETERS (Blackwell RTX 5060 Ti Optimized)
# ==============================================================================
SEED = 42
BATCH_SIZE = 4              # Micro-batch size 4 frees ~3 GB activation VRAM
GRADIENT_ACCUMULATION = 16  # Effective batch size = 4 * 16 = 64 (exact same math, fraction of VRAM)
EPOCHS = 25                 # 25 epochs (5,600 steps) for full convergence and router specialization
LEARNING_RATE_EXPERTS = 2e-4    # Match dense baseline LR (was 5e-5)
LEARNING_RATE_ROUTER = 2e-4     # Align router LR with experts
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05             # Match dense baseline warmup ratio
MAX_INPUT_TOKENS = 384
MAX_GRAD_NORM = 1.0

# Precision
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
USE_GRADIENT_CHECKPOINTING = True
USE_8BIT_ADAM = True        # Uses bitsandbytes.optim.PagedAdamW8bit for minimal VRAM overhead

# Special tokens
ID_START_TOKEN = "<id>"
ID_END_TOKEN = "</id>"
DIGIT_TOKENS = [f"<d{i}>" for i in range(10)]
