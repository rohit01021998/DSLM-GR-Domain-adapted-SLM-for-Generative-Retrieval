import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import gc
import json
import argparse
from pathlib import Path
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from safetensors.torch import save_file, load_file

# Ensure MoE approach is on path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import config_moe
from moe_layer import SharedAndRoutedMoEBlock, ClassicSparseMoEBlock, SubDenseSparseMoEBlock


def upcycle_llama_to_moe(
    model: nn.Module,
    moe_style: str = config_moe.MOE_STYLE,
    num_routed_experts: int = config_moe.NUM_ROUTED_EXPERTS,
    top_k: int = config_moe.TOP_K_ROUTED,
    noise_std: float = config_moe.EXPERT_INIT_NOISE_STD,
    layer_indices: Optional[List[int]] = config_moe.MOE_LAYER_INDICES,
    aux_loss_coef: float = config_moe.AUX_LOSS_COEF,
    jitter_noise: float = config_moe.ROUTER_JITTER_NOISE
) -> nn.Module:
    """
    Converts standard Llama MLP blocks to MoE blocks in-place with minimal memory overhead.
    """
    total_layers = len(model.model.layers)
    target_layers = set(range(total_layers)) if layer_indices is None else set(layer_indices)

    print(f"\n--- Upcycling Model to MoE ({moe_style}) ---")
    print(f"Total layers: {total_layers} | Converting layers: {sorted(list(target_layers))}")

    converted_count = 0
    config = model.config
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    for layer_idx in range(total_layers):
        if layer_idx not in target_layers:
            continue

        orig_mlp = model.model.layers[layer_idx].mlp

        if moe_style == "shared_and_routed":
            moe_block = SharedAndRoutedMoEBlock(
                config=config,
                num_routed_experts=num_routed_experts,
                top_k=top_k,
                jitter_noise=jitter_noise,
                aux_loss_coef=aux_loss_coef
            ).to(device=model_device, dtype=model_dtype)

            # 1. Initialize Shared Expert with exact original weights
            moe_block.shared_expert.gate_proj.weight.data.copy_(orig_mlp.gate_proj.weight.data)
            moe_block.shared_expert.up_proj.weight.data.copy_(orig_mlp.up_proj.weight.data)
            moe_block.shared_expert.down_proj.weight.data.copy_(orig_mlp.down_proj.weight.data)

            # 2. Initialize Routed Experts with slight noise added to break symmetry
            for exp in moe_block.routed_experts:
                noise_gate = torch.randn_like(orig_mlp.gate_proj.weight.data) * noise_std
                exp.gate_proj.weight.data.copy_(orig_mlp.gate_proj.weight.data + noise_gate)

                noise_up = torch.randn_like(orig_mlp.up_proj.weight.data) * noise_std
                exp.up_proj.weight.data.copy_(orig_mlp.up_proj.weight.data + noise_up)

                noise_down = torch.randn_like(orig_mlp.down_proj.weight.data) * noise_std
                exp.down_proj.weight.data.copy_(orig_mlp.down_proj.weight.data + noise_down)

            # 3. Router gate projection initialization
            nn.init.normal_(moe_block.router.gate.weight, mean=0.0, std=0.02)

        elif moe_style == "classic_sparse":
            num_experts = config_moe.CLASSIC_NUM_EXPERTS
            classic_top_k = config_moe.CLASSIC_TOP_K
            moe_block = ClassicSparseMoEBlock(
                config=config,
                num_experts=num_experts,
                top_k=classic_top_k,
                jitter_noise=jitter_noise,
                aux_loss_coef=aux_loss_coef
            ).to(device=model_device, dtype=model_dtype)

            moe_block.experts[0].gate_proj.weight.data.copy_(orig_mlp.gate_proj.weight.data)
            moe_block.experts[0].up_proj.weight.data.copy_(orig_mlp.up_proj.weight.data)
            moe_block.experts[0].down_proj.weight.data.copy_(orig_mlp.down_proj.weight.data)

            for exp in moe_block.experts[1:]:
                noise_gate = torch.randn_like(orig_mlp.gate_proj.weight.data) * noise_std
                exp.gate_proj.weight.data.copy_(orig_mlp.gate_proj.weight.data + noise_gate)

                noise_up = torch.randn_like(orig_mlp.up_proj.weight.data) * noise_std
                exp.up_proj.weight.data.copy_(orig_mlp.up_proj.weight.data + noise_up)

                noise_down = torch.randn_like(orig_mlp.down_proj.weight.data) * noise_std
                exp.down_proj.weight.data.copy_(orig_mlp.down_proj.weight.data + noise_down)

        elif moe_style == "sub_dense_sparse":
            expert_intermediate_size = config_moe.EXPERT_INTERMEDIATE_SIZE
            num_experts = num_routed_experts
            sub_top_k = top_k
            moe_block = SubDenseSparseMoEBlock(
                config=config,
                num_experts=num_experts,
                top_k=sub_top_k,
                expert_intermediate_size=expert_intermediate_size,
                jitter_noise=jitter_noise,
                aux_loss_coef=aux_loss_coef
            ).to(device=model_device, dtype=model_dtype)

            half = expert_intermediate_size  # 4096
            # Expert 0: First functional half of dense neurons [0:4096]
            moe_block.experts[0].gate_proj.weight.data.copy_(orig_mlp.gate_proj.weight.data[0:half, :])
            moe_block.experts[0].up_proj.weight.data.copy_(orig_mlp.up_proj.weight.data[0:half, :])
            moe_block.experts[0].down_proj.weight.data.copy_(orig_mlp.down_proj.weight.data[:, 0:half])

            # Expert 1: Second functional half of dense neurons [4096:8192]
            moe_block.experts[1].gate_proj.weight.data.copy_(orig_mlp.gate_proj.weight.data[half:2*half, :])
            moe_block.experts[1].up_proj.weight.data.copy_(orig_mlp.up_proj.weight.data[half:2*half, :])
            moe_block.experts[1].down_proj.weight.data.copy_(orig_mlp.down_proj.weight.data[:, half:2*half])

            # Expert 2: Center window [2048:6144] + noise
            q1 = half // 2
            noise2 = torch.randn(half, config.hidden_size, device=model_device, dtype=model_dtype) * noise_std
            moe_block.experts[2].gate_proj.weight.data.copy_(orig_mlp.gate_proj.weight.data[q1:q1+half, :] + noise2)
            moe_block.experts[2].up_proj.weight.data.copy_(orig_mlp.up_proj.weight.data[q1:q1+half, :] + noise2)
            noise_down2 = torch.randn(config.hidden_size, half, device=model_device, dtype=model_dtype) * noise_std
            moe_block.experts[2].down_proj.weight.data.copy_(orig_mlp.down_proj.weight.data[:, q1:q1+half] + noise_down2)

            # Expert 3: Interleaved stride [0::2] + noise
            idx_stride = torch.arange(0, 2 * half, 2, device=model_device)
            noise3 = torch.randn(half, config.hidden_size, device=model_device, dtype=model_dtype) * noise_std
            moe_block.experts[3].gate_proj.weight.data.copy_(orig_mlp.gate_proj.weight.data[idx_stride, :] + noise3)
            moe_block.experts[3].up_proj.weight.data.copy_(orig_mlp.up_proj.weight.data[idx_stride, :] + noise3)
            noise_down3 = torch.randn(config.hidden_size, half, device=model_device, dtype=model_dtype) * noise_std
            moe_block.experts[3].down_proj.weight.data.copy_(orig_mlp.down_proj.weight.data[:, idx_stride] + noise_down3)

            nn.init.normal_(moe_block.router.gate.weight, mean=0.0, std=0.02)

        else:
            raise ValueError(f"Unknown MOE_STYLE: {moe_style}")

        # Replace dense MLP with MoE block and free old MLP
        model.model.layers[layer_idx].mlp = moe_block
        del orig_mlp
        converted_count += 1

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.moe_metadata = {
        "moe_style": moe_style,
        "num_experts": config_moe.NUM_EXPERTS if moe_style == "sub_dense_sparse" else (num_routed_experts if moe_style == "shared_and_routed" else config_moe.CLASSIC_NUM_EXPERTS),
        "top_k": config_moe.TOP_K if moe_style == "sub_dense_sparse" else (top_k if moe_style == "shared_and_routed" else config_moe.CLASSIC_TOP_K),
        "expert_intermediate_size": config_moe.EXPERT_INTERMEDIATE_SIZE if moe_style == "sub_dense_sparse" else config.intermediate_size,
        "converted_layers": sorted(list(target_layers)),
        "base_model": str(config_moe.SOURCE_CHECKPOINT_DIR)
    }

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Successfully converted {converted_count} layers to MoE!")
    print(f"Total Parameters: {total_params / 1e9:.2f}B | Trainable: {trainable_params / 1e9:.2f}B")

    return model


def load_and_convert_model(
    checkpoint_dir: Path = config_moe.SOURCE_CHECKPOINT_DIR,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = config_moe.DTYPE,
    layer_indices: Optional[List[int]] = config_moe.MOE_LAYER_INDICES
) -> Tuple[AutoTokenizer, nn.Module]:
    """
    Loads base 1.7B DSI model directly on target device and converts to MoE in-place.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading tokenizer from {checkpoint_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model directly to {device} in {dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=True
    ).to(device)

    model = upcycle_llama_to_moe(model, layer_indices=layer_indices)
    return tokenizer, model


from safetensors.torch import save_model, load_model

def save_moe_checkpoint(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    output_dir: Path,
    metadata: Optional[dict] = None
):
    """
    Saves MoE model weights efficiently with safetensors to prevent CPU RAM spikes
    and correctly handle tied word embeddings.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving MoE checkpoint to {output_dir}...")

    # Save tokenizer & config
    tokenizer.save_pretrained(output_dir)
    model.config.save_pretrained(output_dir)

    # Save MoE metadata (merge architecture metadata with step/epoch stats)
    meta = dict(getattr(model, "moe_metadata", {}))
    if metadata:
        meta.update(metadata)
    with open(output_dir / "moe_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save weights using safetensors save_model (handles tied embeddings and zero-copy streaming)
    weights_path = output_dir / "model.safetensors"
    save_model(model, str(weights_path))
    gc.collect()

    print(f"Checkpoint successfully saved -> {weights_path}")


def load_moe_checkpoint(
    checkpoint_dir: Path,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = config_moe.DTYPE,
    top_k: Optional[int] = None
) -> Tuple[AutoTokenizer, nn.Module]:
    """
    Loads a saved MoE checkpoint using safetensors load_model.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading MoE checkpoint from {checkpoint_dir}...")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)

    with open(checkpoint_dir / "moe_config.json", "r") as f:
        moe_meta = json.load(f)

    config = AutoConfig.from_pretrained(checkpoint_dir)
    
    # 1. Instantiate skeleton and convert on CPU to avoid GPU allocation duplication
    model = AutoModelForCausalLM.from_config(config, dtype=dtype, attn_implementation="sdpa")

    effective_top_k = top_k if top_k is not None else moe_meta.get("top_k", config_moe.TOP_K)

    model = upcycle_llama_to_moe(
        model,
        moe_style=moe_meta.get("moe_style", config_moe.MOE_STYLE),
        num_routed_experts=moe_meta.get("num_experts", config_moe.NUM_EXPERTS),
        top_k=effective_top_k,
        layer_indices=moe_meta.get("converted_layers", None)
    )

    # 2. Load weights on CPU
    safetensors_path = checkpoint_dir / "model.safetensors"
    pt_path = checkpoint_dir / "moe_weights.pt"

    if safetensors_path.exists():
        load_model(model, str(safetensors_path), device="cpu")
    elif pt_path.exists():
        state_dict = torch.load(pt_path, map_location="cpu")
        model.load_state_dict(state_dict)
        del state_dict
    else:
        raise FileNotFoundError(f"No weights file found in {checkpoint_dir}")

    # 3. Move cleanly to target device in a single transfer
    gc.collect()
    model.to(device)
    model.eval()

    print(f"Loaded MoE model successfully on {device} ({dtype}).")
    return tokenizer, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert pretrained 1.7B dense model to Sub-Dense MoE")
    parser.add_argument(
        "--layers", 
        type=str, 
        choices=["all", "alternate"], 
        default="all" if config_moe.MOE_LAYER_INDICES is None else "alternate",
        help="'all' converts all 24 layers (1.21B active params); 'alternate' converts 10 middle layers (1.46B active params)"
    )
    args = parser.parse_args()

    layer_indices = None if args.layers == "all" else list(range(2, 22, 2))

    print("=" * 65)
    print("CONVERTING PRETRAINED 1.7B DSI MODEL TO SUB-DENSE MoE")
    print(f"Mode: {args.layers} layers (Active parameters strictly < 1.71B)")
    print("=" * 65)
    
    tokenizer, model = load_and_convert_model(layer_indices=layer_indices)
    
    save_moe_checkpoint(
        model=model,
        tokenizer=tokenizer,
        output_dir=config_moe.INITIAL_MOE_DIR
    )
    print("\nInitial MoE conversion complete!")
