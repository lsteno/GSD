#!/usr/bin/env python3
"""
Minimal QLoRA fine-tuning script with NVML energy logging.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from nvml_utils import gpu_energy_logger


def load_json_dataset(path: Path, tokenizer, max_length: int) -> Dataset:
    with path.open("r", encoding="utf-8") as fp:
        raw_items = json.load(fp)

    formatted = []
    for item in raw_items:
        messages = [
            {"role": "user", "content": item["input"]},
            {"role": "assistant", "content": item["output"]},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            text = f"<|user|>\n{item['input']}\n<|assistant|>\n{item['output']}"
        formatted.append({"text": text})

    dataset = Dataset.from_list(formatted)

    def tokenize(batch):
        tokenized = tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


class RelativeEarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience: int, min_relative_improvement: float):
        self.patience = patience
        self.min_relative_improvement = min_relative_improvement
        self.best_metric: float | None = None
        self.best_step: int | None = None
        self.wait_count = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric = metrics.get("eval_loss")
        if metric is None:
            return control

        if self.best_metric is None:
            self.best_metric = metric
            self.best_step = state.global_step
            control.should_save = True
            self.wait_count = 0
            return control

        improvement = (self.best_metric - metric) / max(abs(self.best_metric), 1e-8)

        if improvement >= self.min_relative_improvement:
            self.best_metric = metric
            self.best_step = state.global_step
            control.should_save = True
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                control.should_training_stop = True

        return control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal QLoRA fine-tuning")
    parser.add_argument("--base-model", required=True, help="Model name or local path")
    parser.add_argument("--dataset", required=True, type=Path, help="JSON file")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--energy-log", type=Path, default=None)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quant-bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Quantization bit width (default: 4)",
    )
    parser.add_argument(
        "--quant-type",
        default="nf4",
        choices=["nf4", "fp4"],
        help="4-bit quantization type (default: nf4)",
    )
    parser.add_argument(
        "--compute-dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Computation dtype for QLoRA (default: bfloat16)",
    )
    parser.add_argument(
        "--no-double-quant",
        action="store_true",
        help="Disable double quantization (enabled by default)",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing (calls model.enable_input_require_grads)",
    )
    parser.add_argument(
        "--dev-dataset",
        type=Path,
        help="Optional JSON file for validation (enables evaluation + early stopping)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Number of eval periods with no improvement before stopping (default: 3)",
    )
    parser.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=0.01,
        help="Minimum relative improvement (fraction) in eval metric to reset patience (default: 0.01 = 1%)",
    )
    parser.add_argument(
        "--evals-per-epoch",
        type=int,
        default=2,
        help="How many times to run evaluation per epoch (default: 2, i.e., every 50% of an epoch)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map[args.compute_dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_json_dataset(args.dataset, tokenizer, args.max_length)
    eval_dataset = None
    if args.dev_dataset:
        if not args.dev_dataset.exists():
            raise FileNotFoundError(f"Validation JSON not found at {args.dev_dataset}")
        eval_dataset = load_json_dataset(args.dev_dataset, tokenizer, args.max_length)

    # Configure quantization based on bit width
    if args.quant_bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=args.quant_type,
            bnb_4bit_use_double_quant=not args.no_double_quant,
        )
    elif args.quant_bits == 8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=compute_dtype,
        )
    else:
        raise ValueError(f"Unsupported quantization bits: {args.quant_bits}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model.enable_input_require_grads()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    per_step_batch = args.batch_size * args.grad_accum
    steps_per_epoch = max(
        1,
        math.ceil(len(dataset) / per_step_batch),
    )
    eval_steps = None
    if eval_dataset is not None:
        eval_frequency = max(1, args.evals_per_epoch)
        eval_steps = max(1, steps_per_epoch // eval_frequency)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_steps=eval_steps if eval_steps is not None else 50,
        save_strategy="steps",
        eval_strategy="steps" if eval_steps is not None else "no",
        eval_steps=eval_steps,
        save_total_limit=2,
        report_to=["none"],
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=1.0,
        seed=args.seed,
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    callbacks = None
    if eval_dataset is not None:
        callbacks = [
            RelativeEarlyStoppingCallback(
                patience=args.early_stopping_patience,
                min_relative_improvement=args.early_stopping_threshold,
            )
        ]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    if args.energy_log:
        with gpu_energy_logger(device_index=args.device_index) as stats:
            trainer.train()
        stats.update(
            {
                "model_name": args.base_model,
                "dataset": str(args.dataset),
                "output_dir": str(args.output_dir),
            }
        )
        args.energy_log.parent.mkdir(parents=True, exist_ok=True)
        args.energy_log.write_text(json.dumps(stats, indent=2))
    else:
        trainer.train()

    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
