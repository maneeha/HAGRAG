from __future__ import annotations

import os
from dataclasses import dataclass

from .errors import ConfigurationError


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.95


class HuggingFaceGenerator:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        use_4bit: bool = False,
        settings: GenerationSettings | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

        self.settings = settings or GenerationSettings()
        token = os.getenv("HF_TOKEN") or None
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_kwargs = {"torch_dtype": dtype}

        requested_device = device
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"

        if use_4bit:
            if not torch.cuda.is_available():
                raise ConfigurationError("4-bit loading requires a CUDA-capable environment")
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise ConfigurationError(
                    "4-bit loading requested but bitsandbytes is not installed; install hagrag[gpu]"
                ) from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto"
        elif requested_device == "cuda":
            model_kwargs["device_map"] = "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, token=token, **model_kwargs)

        pipeline_kwargs = {
            "task": "text-generation",
            "model": self.model,
            "tokenizer": self.tokenizer,
            "return_full_text": False,
        }
        if requested_device == "cpu" and "device_map" not in model_kwargs:
            pipeline_kwargs["device"] = -1
        self.pipeline = pipeline(**pipeline_kwargs)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        temp = self.settings.temperature if temperature is None else temperature
        kwargs = {
            "max_new_tokens": max_new_tokens or self.settings.max_new_tokens,
            "do_sample": temp > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temp > 0:
            kwargs["temperature"] = temp
            kwargs["top_p"] = self.settings.top_p if top_p is None else top_p
        output = self.pipeline(prompt, **kwargs)
        return output[0]["generated_text"].strip()
