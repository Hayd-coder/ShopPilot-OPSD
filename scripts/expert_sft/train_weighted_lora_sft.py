#!/usr/bin/env python3
"""LoRA SFT with prompt masking and separate Think/Action token weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--think-weight", type=float, default=0.2)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-eval-samples", type=int, default=256)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument(
        "--eval-strategy", choices=("steps", "epoch"), default="steps"
    )
    parser.add_argument(
        "--save-strategy", choices=("steps", "epoch"), default="steps"
    )
    parser.add_argument("--save-total-limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class WeightedSFTDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer,
        max_length: int,
        think_weight: float,
        action_weight: float,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.think_weight = think_weight
        self.action_weight = action_weight

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list]:
        record = self.records[index]
        prompt_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": record["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
        )
        think_text = f"<think>{record['generated_think']}</think>\n"
        action_text = f"<action>{record['expert_action']}</action>"
        think_ids = self.tokenizer.encode(think_text, add_special_tokens=False)
        action_ids = self.tokenizer.encode(action_text, add_special_tokens=False)
        response_ids = think_ids + action_ids + [self.tokenizer.eos_token_id]
        response_weights = (
            [self.think_weight] * len(think_ids)
            + [self.action_weight] * (len(action_ids) + 1)
        )

        prompt_budget = self.max_length - len(response_ids)
        if prompt_budget <= 0:
            raise ValueError(
                f"Response alone exceeds max length at row {index}: {len(response_ids)}"
            )
        if len(prompt_ids) > prompt_budget:
            head = max(1, prompt_budget // 4)
            tail = prompt_budget - head
            prompt_ids = prompt_ids[:head] + prompt_ids[-tail:]

        input_ids = prompt_ids + response_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + response_ids,
            "loss_weights": [0.0] * len(prompt_ids) + response_weights,
        }


class WeightedCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list]]) -> dict[str, torch.Tensor]:
        width = max(len(feature["input_ids"]) for feature in features)
        batch: dict[str, list[list]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "loss_weights": [],
        }
        for feature in features:
            padding = width - len(feature["input_ids"])
            batch["input_ids"].append(
                feature["input_ids"] + [self.pad_token_id] * padding
            )
            batch["attention_mask"].append(
                feature["attention_mask"] + [0] * padding
            )
            batch["labels"].append(feature["labels"] + [-100] * padding)
            batch["loss_weights"].append(
                feature["loss_weights"] + [0.0] * padding
            )
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "loss_weights": torch.tensor(batch["loss_weights"], dtype=torch.float32),
        }


class WeightedSFTTrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")
        loss_weights = inputs.pop("loss_weights")
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :].contiguous()
        target = labels[:, 1:].contiguous()
        weights = loss_weights[:, 1:].to(logits.device)
        valid = target.ne(-100)
        token_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(target)
        effective_weights = weights * valid
        loss = (token_loss * effective_weights).sum() / effective_weights.sum().clamp_min(1e-8)
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_records = load_jsonl(args.train_file)
    validation_records = load_jsonl(args.validation_file)
    if args.max_eval_samples > 0:
        validation_records = validation_records[: args.max_eval_samples]
    train_dataset = WeightedSFTDataset(
        train_records,
        tokenizer,
        args.max_length,
        args.think_weight,
        args.action_weight,
    )
    validation_dataset = WeightedSFTDataset(
        validation_records,
        tokenizer,
        args.max_length,
        args.think_weight,
        args.action_weight,
    )

    if args.dry_run:
        for index in range(min(3, len(train_dataset))):
            item = train_dataset[index]
            labels = item["labels"]
            weights = item["loss_weights"]
            print(
                json.dumps(
                    {
                        "row": index,
                        "tokens": len(item["input_ids"]),
                        "prompt_tokens": sum(label == -100 for label in labels),
                        "think_weight_tokens": sum(
                            abs(weight - args.think_weight) < 1e-8 for weight in weights
                        ),
                        "action_weight_tokens": sum(
                            abs(weight - args.action_weight) < 1e-8 for weight in weights
                        ),
                    }
                )
            )
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        # Keep enough intermediate LoRA points so WebShop behavior, rather
        # than only the final training loss, can select the policy.
        save_total_limit=args.save_total_limit,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        optim="adamw_torch_fused",
        seed=args.seed,
    )
    trainer = WeightedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=WeightedCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    trainer.save_model(str(args.output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "final_adapter"))
    metrics = dict(train_result.metrics)
    if torch.cuda.is_available():
        metrics["peak_gpu_allocated_gib"] = (
            torch.cuda.max_memory_allocated() / 1024**3
        )
        metrics["peak_gpu_reserved_gib"] = (
            torch.cuda.max_memory_reserved() / 1024**3
        )
    print(json.dumps({"final_training_metrics": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
