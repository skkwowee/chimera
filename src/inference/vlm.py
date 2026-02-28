"""
VLM inference for CS2 screenshot analysis using Qwen3.5-27B.
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
    Run inference using Qwen3.5-27B.

    Requires transformers from source:
        pip install git+https://github.com/huggingface/transformers

    For flash attention (faster, less memory):
        pip install flash-attn --no-build-isolation
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-27B",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        use_4bit: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = getattr(torch, torch_dtype)
        self.use_4bit = use_4bit
        self.model = None
        self.processor = None

    def load_model(self):
        """Load the model and processor via Unsloth."""
        from unsloth import FastVisionModel
        from transformers import AutoProcessor

        print(f"Loading {self.model_name} with Unsloth...")
        print(f"  4-bit quantization: {self.use_4bit}")

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            self.model_name,
            load_in_4bit=self.use_4bit,
            use_gradient_checkpointing="unsloth",
        )
        FastVisionModel.for_inference(self.model)

        self.processor = AutoProcessor.from_pretrained(self.model_name)

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

        # Qwen3.5 message format
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
