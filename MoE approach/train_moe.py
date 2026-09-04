import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import re
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

# Ensure MoE approach directory and repo root are on path
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config_moe
from moe_layer import collect_moe_aux_loss
from convert_to_moe import load_and_convert_model, save_moe_checkpoint
from genret.trie import IDTrie, load_trie, make_prefix_allowed_tokens_fn

# Blackwell Tensor Core (TF32/BF16) and Attention Optimizations
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Prefer FlashAttention and Memory-Efficient Attention kernels on Blackwell
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)


class DSIDataset(Dataset):
    """Dataset for DSI dual-task retrieval and indexing pairs."""
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
    Collator for Causal LM:
    Input: prompt + '\n' + target + eos_token
    Masks prompt tokens with -100 in labels.
    """
    def __init__(self, tokenizer, max_length: int = config_moe.MAX_INPUT_TOKENS):
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

            if len(prompt_ids) + len(target_ids) > self.max_length:
                max_prompt_len = max(1, self.max_length - len(target_ids))
                prompt_ids = prompt_ids[-max_prompt_len:]

            full_ids = prompt_ids + target_ids
            prompt_len = len(prompt_ids)

            labels = [-100] * prompt_len + full_ids[prompt_len:]

            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

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
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size())

    is_digit = torch.isin(shift_labels, digit_ids_tensor)
    digit_order = torch.cumsum(is_digit.long(), dim=-1)
    digit_weights = torch.where(
        is_digit,
        1.0 + 0.5 * (digit_order - 1).float(),
        torch.zeros_like(loss)
    )

    weighted_loss = (loss * digit_weights).sum() / digit_weights.sum().clamp(min=1e-6)
    return weighted_loss


def evaluate_quick_hits1(
    model: nn.Module,
    tokenizer,
    trie: IDTrie,
    val_examples: List[Dict[str, Any]],
    device: torch.device,
    limit: int = 200
) -> float:
    """Evaluates quick Hits@1 with constrained Trie beam search."""
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    rng = random.Random(config_moe.SEED)
    val_sample = rng.sample(val_examples, min(limit, len(val_examples)))

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

    hits_1 = 0
    with torch.inference_mode():
        for item in val_sample:
            prompt = item["input"] + "\n"
            target = item["target"]

            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                num_beams=10,
                num_return_sequences=1,
                prefix_allowed_tokens_fn=prefix_fn,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            pred_id_str = tokenizer.decode(gen_tokens, skip_special_tokens=False).strip()

            target_digits = re.findall(r"<d\d>", target)
            pred_digits = re.findall(r"<d\d>", pred_id_str)
            if pred_digits == target_digits and len(target_digits) > 0:
                hits_1 += 1

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    model.train()
    return hits_1 / len(val_sample) if val_sample else 0.0


def train_moe(
    epochs: int = config_moe.EPOCHS,
    batch_size: int = config_moe.BATCH_SIZE,
    grad_accum_steps: int = config_moe.GRADIENT_ACCUMULATION,
    lr_experts: float = config_moe.LEARNING_RATE_EXPERTS,
    lr_router: float = config_moe.LEARNING_RATE_ROUTER,
    max_steps: Optional[int] = None,
    val_interval: int = 200,
    aux_loss_coef: float = config_moe.AUX_LOSS_COEF,
    use_8bit_adam: bool = config_moe.USE_8BIT_ADAM,
    resume_checkpoint: Optional[Path] = None
):
    print("\n" + "=" * 65)
    print("STARTING MoE DSI TRAINING / FINE-TUNING")
    print("=" * 65)
    print(f"• Epochs: {epochs} | Batch size: {batch_size} (Grad Accum: {grad_accum_steps})")
    print(f"• LR Experts: {lr_experts} | LR Router: {lr_router}")
    print(f"• Auxiliary Loss Coef: {aux_loss_coef}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load checkpoint: explicit resume, initial converted, or fresh convert
    from convert_to_moe import load_moe_checkpoint
    if resume_checkpoint and (Path(resume_checkpoint) / "model.safetensors").exists():
        print(f"Resuming training from checkpoint -> {resume_checkpoint}")
        tokenizer, model = load_moe_checkpoint(Path(resume_checkpoint), device=device)
    elif (config_moe.INITIAL_MOE_DIR / "model.safetensors").exists():
        print(f"Loading initial converted MoE from -> {config_moe.INITIAL_MOE_DIR}")
        tokenizer, model = load_moe_checkpoint(config_moe.INITIAL_MOE_DIR, device=device)
    else:
        tokenizer, model = load_and_convert_model(
            checkpoint_dir=config_moe.SOURCE_CHECKPOINT_DIR,
            device=device,
            dtype=config_moe.DTYPE
        )

    # 2. Enable gradient checkpointing for VRAM efficiency
    if config_moe.USE_GRADIENT_CHECKPOINTING and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        print("Enabled gradient checkpointing for VRAM optimization.")

    # 3. Setup Trie and dataset
    print(f"Loading IDs from {config_moe.IDS_PATH}...")
    with open(config_moe.IDS_PATH, "r", encoding="utf-8") as f:
        ids_data = json.load(f)
    trie = IDTrie(ids_data["chunk_to_id"])

    train_dataset = DSIDataset(config_moe.TRAIN_PATH)
    val_dataset = DSIDataset(config_moe.VAL_PATH)
    print(f"Dataset loaded: {len(train_dataset)} train samples, {len(val_dataset)} val samples.")

    collator = DSICollator(tokenizer, max_length=config_moe.MAX_INPUT_TOKENS)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True
    )

    # 4. Digits tensor for loss calculation
    digit_ids = [tokenizer.convert_tokens_to_ids(f"<d{i}>") for i in range(10)]
    digit_ids_tensor = torch.tensor(digit_ids, dtype=torch.long, device=device)

    # 5. Parameter groups: Router vs Experts/Base
    router_params = []
    base_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "router" in name or "gate" in name:
            router_params.append(param)
        else:
            base_params.append(param)

    param_groups = [
        {"params": base_params, "lr": lr_experts, "weight_decay": config_moe.WEIGHT_DECAY},
        {"params": router_params, "lr": lr_router, "weight_decay": 0.0}
    ]

    # 6. Optimizer setup
    optimizer = None
    if use_8bit_adam and torch.cuda.is_available():
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.PagedAdamW8bit(param_groups)
            print("Using bitsandbytes PagedAdamW8bit optimizer.")
        except Exception as e:
            print(f"Warning: bitsandbytes failed ({e}), falling back to standard AdamW.")

    if optimizer is None:
        optimizer = torch.optim.AdamW(param_groups)
        print("Using standard PyTorch AdamW optimizer.")

    total_steps = len(train_loader) // grad_accum_steps * epochs
    if max_steps:
        total_steps = min(total_steps, max_steps)

    warmup_steps = int(total_steps * config_moe.WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_hits1 = 0.0
    global_step = 0
    model.train()

    print(f"\n--- Training for {total_steps} steps (Warmup: {warmup_steps}) ---")
    progress = tqdm(total=total_steps, desc="MoE Training")

    for epoch in range(epochs):
        epoch_dsi_loss = 0.0
        epoch_aux_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type="cuda", dtype=config_moe.DTYPE):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                dsi_loss = compute_depth_weighted_loss(logits, labels, digit_ids_tensor)
                aux_loss = collect_moe_aux_loss(model)
                total_batch_loss = (dsi_loss + aux_loss) / grad_accum_steps

            total_batch_loss.backward()

            epoch_dsi_loss += dsi_loss.item()
            epoch_aux_loss += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config_moe.MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.update(1)

                progress.set_postfix({
                    "dsi_loss": f"{dsi_loss.item():.4f}",
                    "aux_loss": f"{aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                })

                # Validation interval
                if global_step % val_interval == 0 or global_step == total_steps:
                    val_hits1 = evaluate_quick_hits1(
                        model=model,
                        tokenizer=tokenizer,
                        trie=trie,
                        val_examples=val_dataset.examples,
                        device=device,
                        limit=150
                    )
                    print(f"\n[Step {global_step}/{total_steps}] Val Hits@1: {val_hits1:.2%}")

                    if val_hits1 > best_hits1:
                        best_hits1 = val_hits1
                        print(f"🌟 New best Hits@1: {best_hits1:.2%}! Saving checkpoint -> {config_moe.BEST_CHECKPOINT_DIR}")
                        save_moe_checkpoint(
                            model=model,
                            tokenizer=tokenizer,
                            output_dir=config_moe.BEST_CHECKPOINT_DIR,
                            metadata={
                                "best_val_hits1": best_hits1,
                                "step": global_step,
                                "epoch": epoch + 1
                            }
                        )

                if max_steps and global_step >= max_steps:
                    break

        if max_steps and global_step >= max_steps:
            print(f"Reached max_steps ({max_steps}). Finishing training.")
            break

    progress.close()

    # Save final checkpoint
    print(f"\nSaving final checkpoint to {config_moe.LAST_CHECKPOINT_DIR}...")
    save_moe_checkpoint(
        model=model,
        tokenizer=tokenizer,
        output_dir=config_moe.LAST_CHECKPOINT_DIR,
        metadata={"final_step": global_step, "best_val_hits1": best_hits1}
    )

    print("\n" + "=" * 65)
    print(f"TRAINING COMPLETE! Best Validation Hits@1: {best_hits1:.2%}")
    print(f"Checkpoints: {config_moe.BEST_CHECKPOINT_DIR} (best), {config_moe.LAST_CHECKPOINT_DIR} (last)")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train / Fine-tune MoE DSI Model")
    parser.add_argument("--epochs", type=int, default=config_moe.EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=config_moe.BATCH_SIZE, help="Batch size per step")
    parser.add_argument("--grad-accum", type=int, default=config_moe.GRADIENT_ACCUMULATION, help="Gradient accumulation steps")
    parser.add_argument("--lr-experts", type=float, default=config_moe.LEARNING_RATE_EXPERTS, help="Expert learning rate")
    parser.add_argument("--lr-router", type=float, default=config_moe.LEARNING_RATE_ROUTER, help="Router learning rate")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum training steps (useful for pilot runs)")
    parser.add_argument("--val-interval", type=int, default=200, help="Steps between validation evaluations")
    parser.add_argument("--no-8bit", action="store_true", help="Disable 8-bit AdamW")
    parser.add_argument("--resume", action="store_true", help="Resume from last/best checkpoint instead of initial_moe")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint directory to resume from")
    
    args = parser.parse_args()

    resume_path = None
    if args.checkpoint:
        resume_path = Path(args.checkpoint)
    elif args.resume:
        if (config_moe.BEST_CHECKPOINT_DIR / "model.safetensors").exists():
            resume_path = config_moe.BEST_CHECKPOINT_DIR
        elif (config_moe.LAST_CHECKPOINT_DIR / "model.safetensors").exists():
            resume_path = config_moe.LAST_CHECKPOINT_DIR

    train_moe(
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr_experts=args.lr_experts,
        lr_router=args.lr_router,
        max_steps=args.max_steps,
        val_interval=args.val_interval,
        use_8bit_adam=not args.no_8bit,
        resume_checkpoint=resume_path
    )
