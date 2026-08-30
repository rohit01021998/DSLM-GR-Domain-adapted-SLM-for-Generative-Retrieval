import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import re
import json
import random
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup
)
from tqdm import tqdm

from genret import config
from genret.trie import IDTrie, load_trie, make_prefix_allowed_tokens_fn

# Enable TensorFloat-32 (TF32) for RTX Blackwell Tensor Cores
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

def setup_tokenizer_and_model(
    model_name: str = config.MODEL_NAME,
    run_dir: Path = None
) -> tuple:
    """
    Load base model & tokenizer and add custom DSI special tokens:
    <id>, </id>, <d0>, <d1>, ..., <d9>.
    Resizes model embeddings to match the extended vocabulary.
    """
    print(f"Loading base tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add DSI special tokens
    special_tokens = [config.ID_START_TOKEN, config.ID_END_TOKEN] + config.DIGIT_TOKENS
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    print(f"Added {num_added} DSI special tokens to tokenizer. Vocab size: {len(tokenizer)}")

    # Device & dtype setup (Blackwell Native BF16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    print(f"Loading base model: {model_name} in {dtype} with SDPA on {device}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            local_files_only=True,
            trust_remote_code=True
        ).to(device)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            trust_remote_code=True
        ).to(device)

    # Resize token embeddings for new special tokens
    model.resize_token_embeddings(len(tokenizer))

    # Enable gradient checkpointing to drastically reduce activation memory
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        print("Enabled gradient checkpointing for VRAM efficiency.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save tokenizer immediately to run_dir if provided
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(run_dir)
        print(f"Saved extended tokenizer -> {run_dir}")

    return tokenizer, model, device

class DSIDataset(Dataset):
    def __init__(self, data_path: Path):
        self.examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class DSICollator:
    """
    Causal LM Collator for DSI:
    Concatenates `input + '\n' + target + eos_token`.
    Masks the prompt portion with -100 in labels so loss is computed STRICTLY on ID target tokens.
    """
    def __init__(self, tokenizer, max_length: int = config.MAX_INPUT_TOKENS):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []

        for item in batch:
            prompt_str = item["input"] + "\n"
            target_str = item["target"]

            prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
            target_ids = self.tokenizer.encode(target_str, add_special_tokens=False) + [self.tokenizer.eos_token_id]

            # Protect target tokens: truncate prompt from left if combined exceeds max_length
            if len(prompt_ids) + len(target_ids) > self.max_length:
                max_prompt_len = max(1, self.max_length - len(target_ids))
                prompt_ids = prompt_ids[-max_prompt_len:]

            full_ids = prompt_ids + target_ids
            prompt_len = len(prompt_ids)

            # Label masking: -100 for prompt tokens
            labels = [-100] * prompt_len + full_ids[prompt_len:]

            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        # Pad to longest in batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=-100
        )
        
        # Build attention mask strictly from real sequence lengths (Task 10b)
        batch_size = len(input_ids_list)
        max_seq_len = input_ids.shape[1]
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        for b, seq_tensor in enumerate(input_ids_list):
            attention_mask[b, :len(seq_tensor)] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def compute_depth_weighted_loss(
    logits: torch.Tensor, 
    labels: torch.Tensor,
    digit_ids_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Computes depth-weighted CrossEntropy loss strictly on hierarchical digit tokens (<d0>..<d9>).
    Deterministic prefix/suffix tokens (<id>, </id>, <eos>) receive zero loss weight.
    Deeper digit tokens receive progressively higher weights (d1: 1.0, d2: 1.5, d3: 2.0, ...)
    to sharpen leaf discrimination and resolve sibling ambiguities.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size())

    # Create digit mask: True where shift_labels is a valid <d0>..<d9> token
    is_digit = torch.isin(shift_labels, digit_ids_tensor)
    
    # Vectorized progressive depth weighting for digits
    digit_order = torch.cumsum(is_digit.long(), dim=-1)
    digit_weights = torch.where(
        is_digit,
        1.0 + 0.5 * (digit_order - 1).float(),
        torch.zeros_like(loss)
    )

    weighted_loss = (loss * digit_weights).sum() / digit_weights.sum().clamp(min=1e-6)
    return weighted_loss

def evaluate_quick(
    model,
    tokenizer,
    trie: IDTrie,
    val_examples: List[Dict[str, Any]],
    device,
    limit: int = 200
) -> float:
    """
    Quick evaluation of Hits@1 using constrained beam search on validation examples.
    Samples deterministically across the entire corpus using config.SEED and evaluates exact digit matches.
    """
    model.eval()
    rng = random.Random(config.SEED)
    val_sample = rng.sample(val_examples, min(limit, len(val_examples)))
    
    # Map special token IDs
    digit_token_ids = {i: tokenizer.convert_tokens_to_ids(f"<d{i}>") for i in range(10)}
    id_start_token_id = tokenizer.convert_tokens_to_ids(config.ID_START_TOKEN)
    id_end_token_id = tokenizer.convert_tokens_to_ids(config.ID_END_TOKEN)

    prefix_fn = make_prefix_allowed_tokens_fn(
        trie=trie,
        tokenizer=tokenizer,
        digit_token_ids=digit_token_ids,
        id_start_token_id=id_start_token_id,
        id_end_token_id=id_end_token_id
    )

    hits_1 = 0
    with torch.inference_mode():
        for item in val_sample:
            prompt = item["input"] + "\n"
            target = item["target"]

            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.MAX_DEPTH + 3,
                num_beams=10,
                num_return_sequences=1,
                prefix_allowed_tokens_fn=prefix_fn,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            pred_id_str = tokenizer.decode(gen_tokens, skip_special_tokens=False).strip()
            
            # Exact digit sequence matching
            target_digits = re.findall(r"<d\d>", target)
            pred_digits = re.findall(r"<d\d>", pred_id_str)
            if pred_digits == target_digits and len(target_digits) > 0:
                hits_1 += 1

            del inputs, outputs, gen_tokens

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.train()
    return hits_1 / max(1, len(val_sample))

def train_dsi(
    model_name: str = config.MODEL_NAME,
    run_name: Optional[str] = None,
    train_path: Path = config.TRAIN_PATH,
    val_path: Path = config.VAL_PATH,
    ids_path: Path = config.IDS_PATH,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    grad_accum: int = config.GRAD_ACCUM,
    lr: float = config.LR
):
    """
    Main training loop for DSI Generative Retrieval with checkpointing and validation.
    """
    if run_name is None:
        safe_name = model_name.split("/")[-1].lower()
        run_name = f"dsi_{safe_name}"

    run_dir = config.RUNS_DIR / run_name
    best_dir = run_dir / "best"
    last_dir = run_dir / "last"

    tokenizer, model, device = setup_tokenizer_and_model(model_name=model_name, run_dir=run_dir)
    trie = load_trie(ids_path)

    # Pre-cache tokenized dataset directly in DDR5 RAM for zero disk/CPU latency
    print("⚡ Pre-tokenizing dataset into DDR5 system RAM...")
    raw_train = DSIDataset(train_path)
    train_cached = []
    for item in raw_train:
        enc_in = tokenizer.encode(item["input"], add_special_tokens=False)
        enc_tgt = tokenizer.encode(item["target"], add_special_tokens=False)
        train_cached.append((enc_in, enc_tgt))

    class FastRAMDataset(torch.utils.data.Dataset):
        def __init__(self, data): self.data = data
        def __len__(self): return len(self.data)
        def __getitem__(self, idx): return self.data[idx]

    def fast_dynamic_collator(batch):
        # Dynamic length padding: only pad to max length of the current batch
        max_in = max(len(b[0]) for b in batch)
        max_tgt = max(len(b[1]) for b in batch)
        max_len = min(config.MAX_INPUT_TOKENS, max_in + max_tgt + 1)
        pad_id = tokenizer.pad_token_id
        input_ids, labels, attn_mask = [], [], []
        
        for enc_in, enc_tgt in batch:
            full = enc_in + enc_tgt + [tokenizer.eos_token_id]
            if len(full) > max_len:
                full = full[:max_len]
            pad_len = max_len - len(full)
            input_ids.append(full + [pad_id] * pad_len)
            attn_mask.append([1] * len(full) + [0] * pad_len)
            prompt_len = min(len(enc_in), len(full))
            labels.append([-100] * prompt_len + full[prompt_len:] + [-100] * pad_len)
            
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long)
        }

    train_dataset = FastRAMDataset(train_cached)
    val_dataset = DSIDataset(val_path)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=fast_dynamic_collator,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
        prefetch_factor=2,
        persistent_workers=True
    )

    # Optimizer and Cosine Scheduler (Paged 8-bit AdamW for low VRAM footprint)
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=lr, weight_decay=0.01)
        print("Using 8-bit Paged AdamW optimizer (bnb.optim.PagedAdamW8bit) for maximum VRAM efficiency.")
    except Exception:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = (len(train_loader) // grad_accum) * epochs
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    print("\n" + "=" * 60)
    print(f"STARTING DSI TRAINING RUN: {run_name}")
    print(f"Base Model: {model_name}")
    print(f"Total Steps: {total_steps} | Epochs: {epochs} | Batch Size: {batch_size * grad_accum}")
    print("=" * 60)

    # Digit token IDs tensor for vectorized loss computation (Task 4)
    digit_ids_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(f"<d{i}>") for i in range(10)], device=device)

    best_hits1 = -1.0
    val_examples = [val_dataset[i] for i in range(len(val_dataset))]

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = compute_depth_weighted_loss(outputs.logits, labels, digit_ids_tensor) / grad_accum
            loss_val = loss.item() * grad_accum
            loss.backward()

            epoch_loss += loss_val

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                pbar.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

            del input_ids, attention_mask, labels, outputs, loss

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation evaluation
        val_hits1 = evaluate_quick(model, tokenizer, trie, val_examples, device)
        print(f"Epoch {epoch} Summary: Train Loss = {avg_train_loss:.4f} | Val Hits@1 = {val_hits1:.2%}")

        # Checkpointing
        if val_hits1 > best_hits1:
            best_hits1 = val_hits1
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"⭐ New Best Model Checkpointed (Val Hits@1 = {best_hits1:.2%}) -> {best_dir}")

        # Save last checkpoint
        model.save_pretrained(last_dir)
        tokenizer.save_pretrained(last_dir)

    # Save run configuration metadata
    run_meta = {
        "run_name": run_name,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size * grad_accum,
        "best_val_hits1": best_hits1,
        "model_name": model_name,
        "encoder_name": config.ENCODER_NAME
    }
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"\n🎉 DSI Training Complete! Best Val Hits@1 = {best_hits1:.2%}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train DSI Generative Retrieval Model")
    parser.add_argument("--model_name", type=str, default=config.MODEL_NAME, help="Base HF model name")
    parser.add_argument("--run_name", type=str, default=None, help="Run name under runs/")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Micro batch size")
    parser.add_argument("--grad_accum", type=int, default=config.GRAD_ACCUM, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=config.LR, help="Peak learning rate")
    args = parser.parse_args()

    train_dsi(
        model_name=args.model_name,
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr
    )
