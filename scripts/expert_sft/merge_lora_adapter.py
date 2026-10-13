"""Merge a PEFT LoRA adapter into its base model for vLLM/verl inference.

The merged directory is a standalone Hugging Face model.  RL then creates a
fresh LoRA adapter on top of this SFT-initialized policy, so GRPO can
start from exactly the same weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_file = args.adapter / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise FileNotFoundError(f"Missing LoRA weights: {adapter_file}")

    args.output.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model = model.merge_and_unload(safe_merge=True)
    model.config.use_cache = True
    model.save_pretrained(
        args.output,
        safe_serialization=True,
        max_shard_size="4GB",
    )

    tokenizer_source = args.adapter if (args.adapter / "tokenizer.json").is_file() else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    tokenizer.save_pretrained(args.output)

    metadata = {
        "base_model": str(args.base_model.resolve()),
        "adapter": str(args.adapter.resolve()),
        "output": str(args.output.resolve()),
        "dtype": "bfloat16",
    }
    (args.output / "sft_merge_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
