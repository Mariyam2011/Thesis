"""
finetune_lora.py
----------------
LoRA fine-tuning for GSM8K on causal language models.

This script:
  1) Loads GSM8K train split
  2) Formats examples with the project CoT prompt template
  3) Applies label masking so loss is computed only on the target response
  4) Fine-tunes with PEFT/LoRA
  5) Saves adapter weights (and optionally merged full model)
"""

import argparse
import inspect
import math
import re
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from prompts import cot_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning on GSM8K.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", type=str, default="lora_outputs/qwen3_gsm8k")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)

    # Training hyperparameters
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)

    # LoRA hyperparameters
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        help="Model module names to apply LoRA to.",
    )

    # Optional export
    parser.add_argument(
        "--save-merged",
        action="store_true",
        help="Also save merged full model (base + LoRA).",
    )
    return parser.parse_args()


def _to_final_answer_style(answer_text: str) -> str:
    """
    Convert GSM8K answer format into:
      rationale...
      Final Answer: <number>
    GSM8K usually ends with: #### <number>
    """
    text = answer_text.strip()
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if not match:
        return text

    final_number = match.group(1).replace(",", "")
    rationale = re.sub(r"####\s*-?[\d,]+(?:\.\d+)?", "", text).strip()
    if rationale:
        return f"{rationale}\n\nFinal Answer: {final_number}"
    return f"Final Answer: {final_number}"


def build_features(example: dict, tokenizer, max_length: int) -> dict:
    prompt = cot_prompt(example["question"])
    target = _to_final_answer_style(example["answer"])
    full_text = f"{prompt} {target}{tokenizer.eos_token or ''}"

    full_enc = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]
    labels = input_ids.copy()

    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    # If truncation removed all supervised tokens, skip by marking empty target.
    has_target_tokens = any(x != -100 for x in labels)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "has_target_tokens": has_target_tokens,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"
    print(f"[lora] Device: {device}")
    print(f"[lora] Loading model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16 if use_fp16 else torch.float32,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("[lora] Loading GSM8K train split...")
    ds = load_dataset("gsm8k", "main", split="train")

    # Keep a small validation split for monitoring.
    split = ds.train_test_split(test_size=0.02, seed=args.seed, shuffle=True)
    train_ds = split["train"]
    eval_ds = split["test"]

    print("[lora] Tokenizing and masking labels...")
    train_ds = train_ds.map(
        lambda ex: build_features(ex, tokenizer, args.max_length),
        remove_columns=train_ds.column_names,
        desc="Tokenizing train",
    ).filter(lambda x: x["has_target_tokens"])
    eval_ds = eval_ds.map(
        lambda ex: build_features(ex, tokenizer, args.max_length),
        remove_columns=eval_ds.column_names,
        desc="Tokenizing eval",
    ).filter(lambda x: x["has_target_tokens"])

    train_ds = train_ds.remove_columns(["has_target_tokens"])
    eval_ds = eval_ds.remove_columns(["has_target_tokens"])

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt",
        label_pad_token_id=-100,
    )

    ta_signature = inspect.signature(TrainingArguments.__init__).parameters
    steps_per_epoch = math.ceil(len(train_ds) / max(args.batch_size * args.grad_accum, 1))
    total_update_steps = max(1, int(steps_per_epoch * args.epochs))
    warmup_steps = int(total_update_steps * args.warmup_ratio)

    ta_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "bf16": False,
        "fp16": use_fp16,
        "report_to": "none",
        "dataloader_num_workers": 0,
        "gradient_checkpointing": True,
    }

    if "warmup_steps" in ta_signature:
        ta_kwargs["warmup_steps"] = warmup_steps
    elif "warmup_ratio" in ta_signature:
        ta_kwargs["warmup_ratio"] = args.warmup_ratio

    # Transformers API changed: some versions use evaluation_strategy,
    # newer builds may expose eval_strategy.
    if "evaluation_strategy" in ta_signature:
        ta_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in ta_signature:
        ta_kwargs["eval_strategy"] = "steps"
    else:
        ta_kwargs["do_eval"] = True

    if "save_strategy" in ta_signature:
        ta_kwargs["save_strategy"] = "steps"

    train_args = TrainingArguments(**ta_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": train_args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": collator,
    }

    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    print("[lora] Starting training...")
    trainer.train()

    adapter_dir = output_dir / "adapter"
    print(f"[lora] Saving LoRA adapter to {adapter_dir}")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    if args.save_merged:
        print("[lora] Merging adapter into base model...")
        merged = trainer.model.merge_and_unload()
        merged_dir = output_dir / "merged"
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"[lora] Merged model saved to {merged_dir}")

    print("[lora] Done.")


if __name__ == "__main__":
    main()

