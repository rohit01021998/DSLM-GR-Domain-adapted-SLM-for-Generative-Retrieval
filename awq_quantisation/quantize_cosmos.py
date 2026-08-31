import argparse
import json
from pathlib import Path
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

def load_benchmark_calib_data(data_path: Path, tokenizer) -> list:
    """
    Loads benchmark.json and formats user/assistant QA pairs using the model's chat template
    for AWQ calibration.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        bench_items = json.load(f)
    
    calib_data = []
    for item in bench_items:
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()
        if not q:
            continue
        
        messages = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a}
        ]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            except Exception:
                text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
        else:
            text = f"User: {q}\nAssistant: {a}"
        calib_data.append(text)
    
    return calib_data

def main():
    parser = argparse.ArgumentParser(description="AWQ Quantization for Cosmos-Reason2-2B using benchmark.json")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="sasa2000/cosmos-reason2-2b-text-only",
        help="HuggingFace model ID or local directory"
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        default=str(REPO_ROOT / "benchmark.json"),
        help="Path to benchmark.json"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=str(SCRIPT_DIR / "quantised_weights_cosmos"),
        help="Output directory to save quantized model"
    )
    parser.add_argument("--w_bit", type=int, default=4, help="Quantization bit width")
    parser.add_argument("--q_group_size", type=int, default=128, help="Quantization group size")
    args = parser.parse_args()

    model_path = args.model_name
    quant_path = Path(args.output_dir)
    data_path = Path(args.data_path)

    quant_config = {
        "zero_point": True,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": "GEMM"
    }

    print("=" * 65)
    print("🚀 Cosmos Reasoner AWQ 4-Bit Quantization")
    print(f"• Model:       {model_path}")
    print(f"• Calibration: {data_path}")
    print(f"• Output:      {quant_path}")
    print(f"• Config:      {quant_config}")
    print("=" * 65)

    print(f"\n[1/4] Loading model and tokenizer for {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        safetensors=True,
        **{"low_cpu_mem_usage": True, "use_cache": False}
    )

    print(f"\n[2/4] Loading calibration data from {data_path}...")
    calib_data = load_benchmark_calib_data(data_path, tokenizer)
    print(f"Loaded {len(calib_data)} calibration samples formatted with model chat template.")

    print("\n[3/4] Starting AWQ Quantization (weights are being calibrated & packed)...")
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calib_data
    )

    print(f"\n[4/4] Saving quantized model & tokenizer package -> {quant_path}...")
    quant_path.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(quant_path))
    tokenizer.save_pretrained(str(quant_path))

    print("\n🎉 Quantization Finished Successfully!")
    print(f"Quantized model package ready at: {quant_path.resolve()}")
    print("You can run it with vLLM using:")
    print(f"  VLLM_USE_FLASHINFER_SAMPLER=0 python -m vllm.entrypoints.openai.api_server --model {quant_path.resolve()} --port 8000")

if __name__ == "__main__":
    main()
