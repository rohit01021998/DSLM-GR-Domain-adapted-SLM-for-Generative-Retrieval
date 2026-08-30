# DSLM-GR: Domain-adapted SLM for Generative Retrieval

**DSLM-GR** is an end-to-end framework for domain-adapted Small Language Models (SLMs) with generative retrieval and grounded question-answering for technical document corpora.

---

## 🚀 Quick Start

### 1. Environment Setup

Ensure you have a Python 3.10+ virtual environment with PyTorch and CUDA support:

```bash
git clone https://github.com/your-username/DSLM-GR.git
cd DSLM-GR
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Start the Answering & Synthesis Server (vLLM)

DSLM-GR pairs with a local inference engine for synthesis and query optimization:

```bash
# Example: Running a 2B Reasoning Model (cosmos-reason2 / Qwen2.5 / SmolLM2)
VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 .venv/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model sasa2000/cosmos-reason2-2b-text-only \
  --max-model-len 4096 \
  --port 8000 \
  --gpu-memory-utilization 0.50 \
  --chat-template-content-format string
```

*(See `commands.txt` for additional model launch configurations).*

---

### 3. Launch the Interactive Webapp & Benchmark Suite

Launch the unified FastAPI application:

```bash
python app.py
```
Open your browser at **`http://localhost:8080`** to access:
- **Interactive Chat**: Ask complex multi-hop domain questions with step-by-step ReAct reasoning.
- **Benchmark Suite**: Upload test suites, evaluate retrieval accuracy, and inspect traces.

---

### 4. Command-Line Interface (CLI)

You can also interact directly from your terminal:

```bash
# Single grounded question answering
python main.py ask -q "What is the minimum speed tolerance for SOV in the cut-out test?"

# Interactive terminal chat
python main.py chat
```

---

## 📋 Full Training & Indexing Pipeline

To reproduce the indexing and training stages from raw technical documents:

| Stage | Command | Description |
| :---: | :--- | :--- |
| **1. Chunking** | `python main.py chunk` | Generates document-native chunks with structural table retention. |
| **2. Embeddings** | `python main.py embed` | Extracts dense semantic representations (`BAAI/bge-large-en-v1.5`). |
| **3. Semantic IDs** | `python main.py semantic-ids` | Constructs the deterministic hierarchical K-Means prefix tree. |
| **4. Synthetic Queries** | `python main.py queries` | Generates multi-perspective technical queries via LLM. |
| **5. Dataset Build** | `python main.py dataset` | Prepares the dual-task SFT dataset (`index:` and `retrieve:`). |
| **6. SLM Training** | `python main.py train` | Trains the SLM with Trie-constrained loss and custom `<id>` tokens. |
| **7. Evaluation** | `python main.py eval` | Runs comparative evaluation against BM25, Dense, and Hybrid baselines. |

---

## 📁 Repository Structure

```text
DSLM-GR/
├── app.py                      # FastAPI Webapp & ReAct Agentic Harness
├── main.py                     # Central CLI Pipeline Orchestrator
├── commands.txt                # Reference server launch commands
├── genret/
│   ├── config.py               # Centralized hyperparameters & model auto-detection
│   ├── chunking.py             # Section-aware structural chunker
│   ├── embed.py                # Dense vector extraction
│   ├── semantic_ids.py         # Hierarchical K-Means semantic ID generator
│   ├── trie.py                 # Prefix Trie for constrained beam search
│   ├── gen_queries.py          # Synthetic multi-perspective query generator
│   ├── build_dataset.py        # SFT dataset builder with leakage prevention
│   ├── train.py                # SLM training loop (TF32 + SDPA acceleration)
│   ├── infer.py                # Live Trie + PAG generative retriever
│   ├── baselines.py            # Lexical and Dense retrieval baselines
│   └── eval.py                 # Multi-metric comparative evaluation engine
├── utils/
│   ├── ocr_postprocessing.py   # OCR table and structure cleaner
│   ├── prepare_corpus.py       # Raw corpus text aggregator
│   └── chunking_evalution.py   # Chunking strategy verification
├── data/
│   ├── raw/                    # Source extracted document files
│   ├── chunks.jsonl            # Segmented document chunks
│   ├── ids.json                # Generated hierarchical semantic IDs
│   ├── queries.jsonl           # Synthetic query pairs
│   ├── train.jsonl             # SFT training split
│   └── val.jsonl               # SFT validation split
└── runs/
    └── dsi_smollm2-1.7b-instruct/  # Trained DSI checkpoint artifacts
```

---

## ⚡ Hardware Optimizations

- **TensorFloat-32 (TF32)**: Fast matrix operations on modern Tensor Cores.
- **Hardware-Fused SDPA**: Scaled dot-product attention for low-memory footprint.
- **Dynamic vLLM Auto-Discovery**: Automatic runtime model alignment with zero manual re-configuration.

---

## 📄 License & Attribution
Internal research and development codebase. All rights reserved.
