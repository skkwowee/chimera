"""
VLM inference for CS2 screenshot analysis using Qwen3.5-35B-A3B MoE.
"""

import json
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers.generation import LogitsProcessor

from src.prompts import CS2_SYSTEM_PROMPT, CS2_USER_PROMPT


class ThinkingBudgetProcessor(LogitsProcessor):
    """Cap thinking tokens and force a clean </think> transition.

    Reference: Zach Mueller's ThinkingTokenBudgetProcessor
    (https://muellerzr.github.io/til/end_thinking.html)

    Behavior:
    - At 95% budget: soft boost to </think> and newline logits
    - At budget-1: force newline (clean line break)
    - At budget: force </think>, set stopped_thinking=True
    - After </think>: no intervention, model generates freely
    """

    def __init__(self, processor, max_thinking_tokens: int):
        self.max_thinking_tokens = max_thinking_tokens
        # Processor wraps tokenizer — use .tokenizer for encode
        tok = getattr(processor, "tokenizer", processor)
        self.think_end_token = tok.encode("</think>", add_special_tokens=False)[0]
        self.nl_token = tok.encode("\n", add_special_tokens=False)[0]
        self.tokens_generated = 0
        self.stopped_thinking = False
        self.neg_inf = float("-inf")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.tokens_generated += 1

        if self.stopped_thinking or self.max_thinking_tokens is None:
            return scores

        # Soft boost at 95% budget
        ratio = self.tokens_generated / self.max_thinking_tokens
        if ratio > 0.95:
            boost = 1 + ratio
            scores[0][self.nl_token] = scores[0][self.think_end_token] * boost
            scores[0][self.think_end_token] = scores[0][self.think_end_token] * boost

        # Hard cutoff: force \n then </think>
        if self.tokens_generated >= self.max_thinking_tokens - 1:
            if self.tokens_generated == self.max_thinking_tokens - 1:
                scores[:] = self.neg_inf
                scores[0][self.nl_token] = 0
            else:
                scores[:] = self.neg_inf
                scores[0][self.think_end_token] = 0
                self.stopped_thinking = True

        return scores


def parse_json_response(response: str) -> dict:
    """Try to extract JSON from model response."""
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(response[start:end])
    except json.JSONDecodeError:
        pass
    return {"raw_response": response}


class Qwen3VLInference:
    """
    Run inference using Qwen3.5-35B-A3B MoE VLM in bf16.

    Loads from HuggingFace Hub using Qwen3_5MoeForConditionalGeneration.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-35B-A3B",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = getattr(torch, torch_dtype)
        self.model = None
        self.processor = None

    def load_model(self):
        """Load the model and processor from HuggingFace Hub."""
        from transformers import Qwen3_5MoeForConditionalGeneration, AutoProcessor

        print(f"Loading {self.model_name}...")

        self.model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=self.dtype,
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(self.model_name)

        print(f"Model loaded | vision encoder: {hasattr(self.model.model, 'visual')}")

    def analyze(
        self,
        image_path: Path | str,
        prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        enable_thinking: bool = True,
        thinking_budget: int = 512,
    ) -> dict:
        """Analyze a CS2 screenshot."""
        if self.model is None:
            self.load_model()

        if max_new_tokens is None:
            max_new_tokens = 2048 if enable_thinking else 1024

        image_path = Path(image_path)
        prompt_text = prompt or CS2_USER_PROMPT

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Qwen3.5 message format — system prompt separate from user message
        messages = [
            {"role": "system", "content": [{"type": "text", "text": CS2_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]

        # Apply chat template with tokenization
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=enable_thinking,
        ).to(self.model.device)

        # Qwen3.5 recommended sampling params (from official model card)
        # Thinking + structured: temp=0.6, top_p=0.95
        # Non-thinking general:  temp=0.7, top_p=0.8
        # Never use greedy (do_sample=False) or repetition_penalty > 1.0
        if enable_thinking:
            temperature, top_p = 0.6, 0.95
        else:
            temperature, top_p = 0.7, 0.8

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=20,
        )
        if enable_thinking:
            gen_kwargs["logits_processor"] = [
                ThinkingBudgetProcessor(self.processor, thinking_budget)
            ]

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(**gen_kwargs)

        # Decode only the generated tokens (exclude input)
        generated_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return parse_json_response(response)

    def analyze_batch(
        self,
        image_paths: list[Path | str],
        output_dir: Optional[Path | str] = None,
    ) -> list[dict]:
        """Analyze multiple screenshots."""
        from tqdm import tqdm

        if self.model is None:
            self.load_model()

        results = []
        output_dir = Path(output_dir) if output_dir else None

        for image_path in tqdm(image_paths, desc="Analyzing screenshots"):
            image_path = Path(image_path)
            result = self.analyze(image_path)
            result["_source_image"] = str(image_path)
            results.append(result)

            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                result_path = output_dir / f"{image_path.stem}_analysis.json"
                with open(result_path, "w") as f:
                    json.dump(result, f, indent=2)

        return results
