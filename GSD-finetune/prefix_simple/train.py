#!/usr/bin/env python3
"""
Minimal prefix-tuning script with NVML energy logging.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import PrefixTuningConfig, get_peft_model

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
            return_attention_mask=True,
        )
        input_ids = np.array(tokenized["input_ids"])
        labels = input_ids.copy()
        pad_id = tokenizer.pad_token_id
        labels[labels == pad_id] = -100
        tokenized["labels"] = labels.tolist()
        return tokenized

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal prefix tuning")
    parser.add_argument("--base-model", required=True, help="Model name or local path")
    parser.add_argument("--dataset", required=True, type=Path, help="JSON file")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--energy-log", type=Path, default=None)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-virtual-tokens",
        type=int,
        default=128,
        help="Number of virtual tokens for prefix tuning (default: 128)",
    )
    parser.add_argument(
        "--prefix-projection",
        action="store_true",
        help="Enable prefix projection layer (recommended for small prefixes)",
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
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint directory to resume training from",
    )
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    # Handle checkpoint resumption for prefix tuning
    # Due to PEFT bugs with prefix tuning resume, we need special handling
    if args.resume_from_checkpoint and args.resume_from_checkpoint.exists():
        from peft import PeftConfig
        print(f"Loading model from checkpoint: {args.resume_from_checkpoint}")
        
        # Load the saved adapter config
        loaded_config = PeftConfig.from_pretrained(str(args.resume_from_checkpoint))
        
        # Create base model
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            cache_dir=args.cache_dir,
            trust_remote_code=True,
        )
        
        # Create a fresh PEFT model with the loaded config
        model = get_peft_model(base_model, loaded_config)
        
        # Manually load the adapter weights
        adapter_path = args.resume_from_checkpoint / "adapter_model.bin"
        if not adapter_path.exists():
            adapter_path = args.resume_from_checkpoint / "adapter_model.safetensors"
        
        if adapter_path.suffix == ".safetensors":
            from safetensors.torch import load_file
            adapter_weights = load_file(str(adapter_path))
        else:
            adapter_weights = torch.load(adapter_path, map_location="cpu")
        
        model.load_state_dict(adapter_weights, strict=False)
        
        # Let Trainer handle optimizer/scheduler/step restoration
        resume_from_checkpoint = str(args.resume_from_checkpoint)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            cache_dir=args.cache_dir,
            trust_remote_code=True,
        )
        prefix_cfg = PrefixTuningConfig(
            task_type="CAUSAL_LM",
            num_virtual_tokens=args.num_virtual_tokens,
            prefix_projection=args.prefix_projection,
        )
        model = get_peft_model(model, prefix_cfg)
        resume_from_checkpoint = None

    model.print_trainable_parameters()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
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
        bf16=torch.cuda.is_available(),
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
        load_best_model_at_end=False,  # Caused shape mismatch error when set to True
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
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
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
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    metadata = {"base_model": args.base_model}
    (args.output_dir / "prefix_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
