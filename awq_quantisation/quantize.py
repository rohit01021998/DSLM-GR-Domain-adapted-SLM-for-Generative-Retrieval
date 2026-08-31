import argparse
import json
from pathlib import Path
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

def load_calib_data(data_path: Path, tokenizer, model_type: str = "dsi"):
    """
    Loads custom calibration data from JSONL (train.jsonl) or JSON (benchmark.json).
    """
    data = []
    if str(data_path).endswith(".jsonl"):
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                text = item["input"] + "\n" + item["target"]
                data.append(text)
    elif str(data_path).endswith(".json"):
        with open(data_path, "r", encoding="utf-8") as f:
            bench_items = json.load(f)
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
            data.append(text)
    return data

def main():
    parser = argparse.ArgumentParser(description="AWQ Quantization for DSI or Reasoning Models")
    parser.add_argument(
        "--target",
        type=str,
        choices=["dsi", "cosmos"],
        default="cosmos",
        help="Target preset to quantize: 'cosmos' (Reasoning 2B) or 'dsi' (SmolLM2 DSI finetuned)"
    )
    parser.add_argument("--model_path", type=str, default=None, help="Custom model path / repo id")
    parser.add_argument("--data_path", type=str, default=None, help="Custom calibration data path")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--w_bit", type=int, default=4, help="Bit width (default: 4)")
    parser.add_argument("--q_group_size", type=int, default=128, help="Group size (default: 128)")
    parser.add_argument("--max_calib_samples", type=int, default=512, help="Max calibration samples")
    args = parser.parse_args()

    if args.target == "cosmos":
        model_path = args.model_path or "sasa2000/cosmos-reason2-2b-text-only"
        data_path = Path(args.data_path) if args.data_path else REPO_ROOT / "benchmark.json"
        output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "quantised_weights_cosmos"
    else:
        model_path = args.model_path or str(REPO_ROOT / "runs/dsi_smollm2-1.7b-instruct/best")
        data_path = Path(args.data_path) if args.data_path else REPO_ROOT / "data/train.jsonl"
        output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "quantised_weights"

    quant_config = {
        "zero_point": True,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": "GEMM"
    }

    print("=" * 65)
    print(f"🚀 AWQ 4-Bit Quantization [{args.target.upper()}]")
    print(f"• Model:       {model_path}")
    print(f"• Calibration: {data_path}")
    print(f"• Output:      {output_dir}")
    print(f"• Config:      {quant_config}")
    print("=" * 65)

    print(f"\n[1/4] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(
        model_path, 
        safetensors=True,
        **{"low_cpu_mem_usage": True, "use_cache": False}
    )

    print(f"\n[2/4] Loading calibration data from {data_path}...")
    calib_data = load_calib_data(data_path, tokenizer, model_type=args.target)
    if len(calib_data) > args.max_calib_samples:
        calib_data = calib_data[:args.max_calib_samples]

    print(f"Loaded {len(calib_data)} calibration samples.")
    print("\n[3/4] Starting AWQ Quantization...")
    model.quantize(
        tokenizer, 
        quant_config=quant_config, 
        calib_data=calib_data
    )

    print(f"\n[4/4] Saving quantized model to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("\n🎉 Quantization complete!")
    print(f"Saved package to: {output_dir.resolve()}")

if __name__ == "__main__":
    main()
