"""
VLM inference for CS2 screenshot analysis.

Supports:
- DeepSeek-VL2 (requires deepseek_vl2 package from their repo)
- Qwen2-VL (uses transformers)
- Qwen3-VL (newest, requires transformers from source)
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


class DeepSeekVLInference:
    """
    Run inference on CS2 screenshots using DeepSeek-VL2.

    Requires installing deepseek_vl2 package:
        git clone https://github.com/deepseek-ai/DeepSeek-VL2
        cd DeepSeek-VL2
        pip install -e .

    Model sizes and VRAM requirements:
        - deepseek-vl2-tiny: ~3GB (1.0B active params)
        - deepseek-vl2-small: ~8GB (2.8B active params)
        - deepseek-vl2: ~15GB (4.5B active params)

    Note: For 4090 (24GB), deepseek-vl2-small or deepseek-vl2 should work fine.
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/deepseek-vl2-small",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = getattr(torch, torch_dtype)
        self.model = None
        self.processor = None
        self.tokenizer = None

    def load_model(self):
        """Load the model and processor."""
        try:
            from deepseek_vl2.models import DeepseekVLV2Processor
        except ImportError:
            raise ImportError(
                "DeepSeek-VL2 package not found. Install it with:\n"
                "  git clone https://github.com/deepseek-ai/DeepSeek-VL2\n"
                "  cd DeepSeek-VL2 && pip install -e ."
            )

        from transformers import AutoModelForCausalLM

        print(f"Loading {self.model_name}...")

        # Load processor
        self.processor = DeepseekVLV2Processor.from_pretrained(self.model_name)
        self.tokenizer = self.processor.tokenizer

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model = self.model.to(self.dtype).to(self.device).eval()

        print(f"Model loaded on {self.device}")

    def analyze(
        self,
        image_path: Path | str,
        prompt: Optional[str] = None,
        max_new_tokens: int = 1024,
    ) -> dict:
        """
        Analyze a CS2 screenshot.

        Args:
            image_path: Path to the screenshot
            prompt: Custom prompt (uses default CS2 prompt if not provided)
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dictionary with analysis results
        """
        if self.model is None:
            self.load_model()

        image_path = Path(image_path)
        prompt_text = prompt or CS2_SYSTEM_PROMPT

        # DeepSeek-VL2 conversation format
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n{prompt_text}",
                "images": [str(image_path)],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # Load and process images
        from deepseek_vl2.utils.io import load_pil_images
        pil_images = load_pil_images(conversation)

        # Prepare inputs
        prepare_inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt="",
        ).to(self.model.device)

        # Generate embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        # Generate response
        with torch.no_grad():
            outputs = self.model.language.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        # Decode response
        response = self.tokenizer.decode(
            outputs[0].cpu().tolist(),
            skip_special_tokens=True,
        )

        return _parse_json_response(response)

    def analyze_batch(
        self,
        image_paths: list[Path | str],
        output_dir: Optional[Path | str] = None,
    ) -> list[dict]:
        """
        Analyze multiple screenshots.

        Args:
            image_paths: List of paths to screenshots
            output_dir: If provided, save results as JSON files

        Returns:
            List of analysis dictionaries
        """
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


class Qwen3VLInference:
    """
    Run inference using Qwen3-VL (newest, recommended).

    Requires transformers from source:
        pip install git+https://github.com/huggingface/transformers

    For flash attention (faster, less memory):
        pip install flash-attn --no-build-isolation

    Model sizes:
        - Qwen3-VL-8B-Instruct: ~18GB VRAM
        - Qwen3-VL-30B-A3B-Instruct: ~8GB active (MoE, 30B total but 3B active)
        - Qwen3-VL-32B-Instruct: ~64GB (too large for 4090)

    Recommended for 4090: Qwen3-VL-8B-Instruct or Qwen3-VL-30B-A3B-Instruct (MoE)
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


class QwenVLInference:
    """
    Run inference using Qwen2-VL (legacy, use Qwen3VLInference for newer models).

    Uses the transformers library directly. Install with:
        pip install transformers>=4.45.0

    For flash attention (faster, less memory):
        pip install flash-attn --no-build-isolation

    Model sizes:
        - Qwen2-VL-2B-Instruct: ~5GB
        - Qwen2-VL-7B-Instruct: ~16GB
        - Qwen2.5-VL-7B-Instruct: ~16GB (newer version)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
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
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        print(f"Loading {self.model_name}...")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        # Load model with optional flash attention
        model_kwargs = {
            "torch_dtype": self.dtype,
            "device_map": "auto",
        }

        if self.use_flash_attention:
            try:
                import flash_attn  # noqa: F401
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("Using flash attention 2")
            except ImportError:
                # Fall back to SDPA (PyTorch 2.0+ built-in efficient attention)
                model_kwargs["attn_implementation"] = "sdpa"
                print("Flash attention not installed, using PyTorch SDPA")

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
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

        # Qwen2-VL message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path.absolute()}"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # Apply chat template and tokenize
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
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
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
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
