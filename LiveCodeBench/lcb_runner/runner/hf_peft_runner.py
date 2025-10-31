"""Runner for Hugging Face causal models fine-tuned with PEFT prefix-tuning adapters."""

from __future__ import annotations

import json
import os
from typing import Any

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "transformers is required to use the HuggingFacePrefix runner"
    ) from exc

try:
    from peft import PeftModel
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError("peft is required to load prefix-tuning adapters") from exc

from lcb_runner.runner.base_runner import BaseRunner


class HFPefTRunner(BaseRunner):
    """Runs text generation using a PEFT prefix-tuned Hugging Face model."""

    def __init__(self, args, model):
        super().__init__(args, model)
        if not args.peft_adapter_path:
            raise ValueError(
                "--peft_adapter_path must be provided when using the HuggingFacePrefix runner"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_path = (
            os.path.expanduser(args.local_model_path)
            if args.local_model_path
            else model.model_name
        )
        adapter_path = os.path.expanduser(args.peft_adapter_path)

        torch_dtype = self._resolve_dtype(args.dtype)
        if self.device.type == "cpu" and torch_dtype in (torch.float16, torch.bfloat16):
            torch_dtype = torch.float32
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_path,
                trust_remote_code=args.trust_remote_code,
            )
        except Exception:
            print(
                "[HFPefTRunner] Could not load tokenizer from base model; falling back to adapter resources."
            )
            # Fall back to loading the tokenizer from the adapter directory.
            self.tokenizer = AutoTokenizer.from_pretrained(
                adapter_path,
                trust_remote_code=args.trust_remote_code,
            )
        if self.tokenizer.pad_token is None:
            # Default to eos token for padding if none is defined
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id or self.pad_token_id
        self.tokenizer.padding_side = "left"

        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
        )
        if len(getattr(self.tokenizer, "added_tokens_decoder", {})) > 0:
            base_model.resize_token_embeddings(len(self.tokenizer))

        base_model.config.pad_token_id = self.pad_token_id
        base_model.config.eos_token_id = self.eos_token_id

        self.peft_model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            is_trainable=False,
        )
        self.peft_model.to(self.device)
        self.peft_model.eval()

        # Multiprocessing is not supported because the model lives on a single GPU process.
        if getattr(self.args, "multiprocess", 0) not in (0, 1):
            print(
                "[HFPefTRunner] Multiprocessing is not supported; falling back to single-process generation."
            )
            self.args.multiprocess = 0

        self.stop_sequences = [stop for stop in (args.stop or []) if stop]

    @staticmethod
    def _resolve_dtype(dtype_str: str | None) -> torch.dtype:
        if not dtype_str:
            return torch.float16 if torch.cuda.is_available() else torch.float32
        mapping: dict[str, torch.dtype] = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        return mapping.get(dtype_str.lower(), torch.float16)

    def _normalize_prompt(self, prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list):
            # Assume chat-style messages
            return "\n\n".join(
                f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in prompt
            )
        if isinstance(prompt, tuple) and len(prompt) == 2:
            prefix, metadata = prompt
            if isinstance(prefix, str):
                return prefix + json.dumps(metadata)
        raise TypeError(f"Unsupported prompt type for HF PEFT runner: {type(prompt)}")

    def _apply_stop_sequences(self, text: str) -> str:
        if not self.stop_sequences:
            return text.strip()
        truncated = text
        for stop in self.stop_sequences:
            position = truncated.find(stop)
            if position != -1:
                truncated = truncated[:position]
        return truncated.strip()

    def _run_single(self, prompt) -> list[str]:
        prompt_text = self._normalize_prompt(prompt)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=True,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        input_length = encoded["input_ids"].shape[-1]

        do_sample = self.args.temperature > 0
        num_return = self.args.n if do_sample else 1

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.args.max_tokens,
            "num_return_sequences": num_return,
            "do_sample": do_sample,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.args.temperature
            generation_kwargs["top_p"] = self.args.top_p

        with torch.no_grad():
            output_ids = self.peft_model.generate(
                **encoded,
                **generation_kwargs,
            )

        if output_ids.dim() == 1:
            output_ids = output_ids.unsqueeze(0)

        new_token_ids = output_ids[:, input_length:]
        generations = self.tokenizer.batch_decode(
            new_token_ids, skip_special_tokens=True
        )

        generations = [self._apply_stop_sequences(gen) for gen in generations]

        if not generations:
            generations = [""]

        if not do_sample and self.args.n > 1:
            generations = generations * self.args.n
        if len(generations) < self.args.n:
            generations = generations + [generations[-1]] * (self.args.n - len(generations))

        generations = generations[: self.args.n]
        assert len(generations) == self.args.n
        return generations

