"""
VLM Backend wrapper for judge.py.

Supports OpenAI-compatible APIs and local Qwen-VL inference.
Configuration is read from agent.config.run_config.
"""

import base64
from typing import List, Optional

from agent.config import run_config


class VLMBackend:
    """Unified interface for calling a Vision-Language Model."""

    def __init__(self) -> None:
        self._qwen_model = None
        self._qwen_processor = None

    # ── Public API ────────────────────────────────────────────────────────

    def call(self, prompt: str, image_paths: Optional[List[str]] = None) -> str:
        """Call the configured VLM backend and return raw text."""
        if run_config.BACKEND == "openai":
            return self._call_openai(prompt, image_paths)
        else:
            return self._call_qwen(prompt, image_paths)

    # ── OpenAI backend ────────────────────────────────────────────────────

    def _call_openai(self, prompt: str, image_paths: Optional[List[str]] = None) -> str:
        """Call an OpenAI-compatible API (GPT-4o, etc.)."""
        from openai import OpenAI

        client = OpenAI(
            api_key=run_config.OPENAI_API_KEY,
            base_url=run_config.OPENAI_BASE_URL,
        )

        content: list = []
        if image_paths:
            for p in image_paths[:4]:  # limit to 4 frames
                with open(p, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
        content.append({"type": "text", "text": prompt})

        resp = client.chat.completions.create(
            model=run_config.OPENAI_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=1024,
            temperature=0.0,
        )
        return resp.choices[0].message.content

    # ── Qwen backend ─────────────────────────────────────────────────────

    def _load_qwen(self):
        """Lazy-load the Qwen model and processor (singleton)."""
        if self._qwen_model is not None:
            return

        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        import torch

        self._qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
            run_config.QWEN_MODEL_PATH,
            torch_dtype="auto",
            device_map="auto",
        )
        self._qwen_model.eval()
        self._qwen_processor = AutoProcessor.from_pretrained(
            run_config.QWEN_MODEL_PATH,
        )

    def _call_qwen(self, prompt: str, image_paths: Optional[List[str]] = None) -> str:
        """Call local Qwen-VL model and return raw text."""
        self._load_qwen()

        model = self._qwen_model
        processor = self._qwen_processor

        content: list = []
        if image_paths:
            for p in image_paths[:4]:
                content.append({"type": "image", "image": p})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        tok = processor.tokenizer
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=1024,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
