"""
VLM inference for CS2 screenshot analysis using Qwen3-VL-8B.
"""

import json
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from src.prompts import CS2_SYSTEM_PROMPT


def _parse_json_response(response: str) -> dict:
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
    Run inference using Qwen3-VL-8B.

    Requires transformers from source:
        pip install git+https://github.com/huggingface/transformers

    For flash attention (faster, less memory):
        pip install flash-attn --no-build-isolation

    VRAM: ~18GB (full precision), ~6GB (4-bit quantized)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        use_flash_attention: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = getattr(torch, torch_dtype)
        self.use_flash_attention = use_flash_attention
        self.model = None
        self.processor = None

    def load_model(self):
        """Load the model and processor."""
        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError(
                "Qwen3-VL requires transformers from source:\n"
                "  pip install git+https://github.com/huggingface/transformers"
            )

        print(f"Loading {self.model_name}...")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        # Load model with optional flash attention
        model_kwargs = {
            "torch_dtype": self.dtype,
            "device_map": "auto",
        }

        # Use SDPA (PyTorch 2.0+ built-in efficient attention)
        # Flash attention can cause OOM issues on some systems
        model_kwargs["attn_implementation"] = "sdpa"
        print("Using PyTorch SDPA")

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        print(f"Model loaded on {self.device}")

    def analyze(
        self,
        image_path: Path | str,
        prompt: Optional[str] = None,
        max_new_tokens: int = 1024,
    ) -> dict:
        """Analyze a CS2 screenshot."""
        if self.model is None:
            self.load_model()

        image_path = Path(image_path)
        prompt_text = prompt or CS2_SYSTEM_PROMPT

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Qwen3-VL message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # Apply chat template with tokenization
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

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

        return _parse_json_response(response)

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
