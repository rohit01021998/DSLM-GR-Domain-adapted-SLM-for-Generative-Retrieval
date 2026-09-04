# Mixture of Experts (MoE) for Generative Retrieval (DSI)

This directory contains the complete framework for converting, fine-tuning, and evaluating a **Mixture of Experts (MoE)** model derived from the pretrained **1.7B DSI model** (`runs/dsi_smollm2-1.7b-instruct/best`).

---

## 1. Motivation & Architecture

In Generative Retrieval / Differentiable Search Index (DSI), a single language model performs two fundamentally distinct objectives:
1. **`retrieve:`** High-level semantic reasoning, question parsing, and query-to-document mapping.
2. **`index:`** Dense structural memorization of exact document snippets, tables, and boundary context.

By upcycling the dense Llama architecture into a **Mixture of Experts (MoE)**, different sub-networks (experts) dynamically specialize across tasks, semantic clusters, and tree depths without multiplying inference compute.

### Architecture: Shared-and-Routed (DeepSeek / Qwen-MoE Style)
Rather than naive sparse routing where all experts are equal:
- **1 Shared Expert (Always Active)**: Inherits the exact pretrained 1.7B MLP weights. Guarantees that foundational retrieval and language generation capabilities are never degraded.
- **$M$ Routed Experts (Top-$k$ Active)**: Initialized from the pretrained weights with small symmetry-breaking Gaussian noise ($\sigma = 0.01$). Learn specialized representations for specific query patterns and semantic subtrees.
- **Top-$k$ Router with GShard Auxiliary Loss**: Computes routing probabilities with a load-balancing loss:
  $$\mathcal{L}_{\text{aux}} = \lambda \cdot N \sum_{i=1}^N f_i \cdot P_i$$

---

## 2. Directory Structure

```
MoE approach/
├── config_moe.py       # Centralized hyperparameters, model paths, and layer configurations
├── moe_layer.py        # PyTorch implementations of TopKRouter, SharedAndRoutedMoEBlock, ClassicSparseMoEBlock
├── convert_to_moe.py   # Upcycling script: converts 1.7B dense checkpoint into MoE
├── train_moe.py        # Fine-tuning loop with DSI depth-weighted loss + MoE load-balancing loss
├── eval_moe.py         # Evaluation script (Hits@1/5/10, MRR@10, Trie beam search, routing analytics)
├── infer_moe.py        # Single-query CLI inference with interactive layer routing inspection
├── README.md           # This documentation
└── checkpoints/        # Saved MoE checkpoints (initial, best, last) and metrics
```

---

## 3. Quickstart & Workflow

### Step 1: Upcycle Dense Checkpoint to MoE
Converts the dense checkpoint in `runs/dsi_smollm2-1.7b-instruct/best` into an initialized MoE model:
```bash
python "MoE approach/convert_to_moe.py"
```
This saves the initialized MoE model to `MoE approach/checkpoints/initial_moe/`.

### Step 2: Fine-Tuning with DSI Data

#### Pilot Test Run (e.g., 200 steps to verify memory & convergence)
```bash
python "MoE approach/train_moe.py" --max-steps 200 --val-interval 50
```

#### Full Fine-Tuning (5 epochs with PagedAdamW8bit & Gradient Checkpointing)
```bash
python "MoE approach/train_moe.py" --epochs 5 --batch-size 16 --grad-accum 4
```
The best checkpoint (highest validation Hits@1) is saved automatically to:
`MoE approach/checkpoints/best/`

### Step 3: Comprehensive Evaluation
Evaluate retrieval metrics (Hits@1, Hits@5, Hits@10, MRR@10) with constrained Trie beam search and inspect expert routing distributions:
```bash
python "MoE approach/eval_moe.py" --checkpoint "MoE approach/checkpoints/best"
```
Results and expert activation counts are saved to `MoE approach/checkpoints/eval_results.json`.

### Step 4: Interactive / Single-Query Inference
Test specific queries and inspect which experts were selected at each layer:
```bash
python "MoE approach/infer_moe.py" --query "What does VUT stand for in the Assisted Driving protocol?"
```

### Step 5: Launch MoE ReAct Web App & Benchmark Suite
Run the full interactive web application powered by the MoE DSI retrieval model:
```bash
python "MoE approach/app.py"
```
Or with custom port:
```bash
PORT=8081 python "MoE approach/app.py"
```
Open `http://localhost:8080` to access the interactive Q&A assistant and continuous-batching benchmark runner.

---

## 4. Hardware & Memory Optimization

To fit comfortably within a **16 GB VRAM** GPU (e.g. RTX 5060 Ti):
- **PagedAdamW8bit**: Uses `bitsandbytes.optim.PagedAdamW8bit` to store optimizer states in 8-bit, reducing optimizer VRAM footprint by ~75%.
- **Gradient Checkpointing**: Drastically reduces activation memory during training.
- **Selective Layer MoE (Optional)**: In `config_moe.py`, set `MOE_LAYER_INDICES = list(range(0, 24, 2))` to convert alternate layers, reducing parameter count while preserving MoE capacity.
