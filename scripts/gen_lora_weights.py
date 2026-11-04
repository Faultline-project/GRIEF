#!/usr/bin/env python3
"""
Generate two random LoRA adapters for fuzzing.

Each adapter has independent random weights (different seeds), so they
produce maximally different outputs from the same base model. This makes
cross-adapter KV contamination immediately visible in the oracle.

Usage:
    python3 scripts/gen_lora_weights.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --output-dir /tmp/lora_weights \\
        --rank 8

Produces:
    /tmp/lora_weights/lora_a/   (seed=1)
    /tmp/lora_weights/lora_b/   (seed=2)

Then start vLLM with:
    python -m vllm.entrypoints.openai.api_server \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --enable-lora \\
        --lora-modules lora_a=/tmp/lora_weights/lora_a \\
                       lora_b=/tmp/lora_weights/lora_b
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fuzzer.adapters.vllm_lora.weights import generate_random_lora


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate random LoRA adapters for fuzzing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",        default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir",   default="/tmp/lora_weights")
    parser.add_argument("--rank",         type=int, default=8)
    parser.add_argument("--num-adapters", type=int, default=6,
                        help="Number of adapters to generate (1–6, named lora_a..lora_f)")
    args = parser.parse_args()

    _NAMES = ["lora_a", "lora_b", "lora_c", "lora_d", "lora_e", "lora_f"]
    if not (1 <= args.num_adapters <= len(_NAMES)):
        print(f"--num-adapters must be between 1 and {len(_NAMES)}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    adapters = [(_NAMES[i], i + 1) for i in range(args.num_adapters)]
    for name, seed in adapters:
        print(f"Generating {name} (seed={seed}) ...", flush=True)
        path = generate_random_lora(
            base_model_name=args.model,
            output_dir=args.output_dir,
            adapter_name=name,
            rank=args.rank,
            seed=seed,
        )
        print(f"  → {path}")

    print()
    print("Done.")
    print()
    print("Start vLLM with:")
    print(f"  --enable-lora \\")
    for name, _ in adapters:
        path = os.path.join(args.output_dir, name)
        print(f"  --lora-modules {name}={path} \\")
    print()
    print("Start SGLang with:")
    lora_paths = " ".join(
        f"{name}={os.path.join(args.output_dir, name)}"
        for name, _ in adapters
    )
    print(f"  --lora-paths {lora_paths} \\")
    print(f"  --max-loras-per-batch 2 \\")


if __name__ == "__main__":
    main()
