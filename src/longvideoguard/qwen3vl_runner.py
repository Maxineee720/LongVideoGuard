from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GenerationResult:
    raw_output: str
    latency_seconds: float
    input_token_count: int
    generated_token_count: int
    peak_gpu_memory_mb: float | None


class Qwen3VLRunner:
    """Minimal deterministic Qwen3-VL video inference wrapper.

    Heavy dependencies are imported lazily so data utilities and unit tests do
    not require PyTorch or Transformers.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        *,
        dtype: str = "auto",
        attn_implementation: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                'VLM dependencies are missing. Install with: '
                'pip install -e ".[vlm]"'
            ) from exc

        self._torch = torch
        supported_dtypes = {"auto", "float16", "bfloat16", "float32"}
        if dtype not in supported_dtypes:
            raise ValueError(
                f"dtype must be one of {sorted(supported_dtypes)}, got {dtype!r}"
            )
        resolved_dtype: Any = "auto" if dtype == "auto" else getattr(torch, dtype)
        model_kwargs: dict[str, Any] = {
            "dtype": resolved_dtype,
            "device_map": "auto",
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model_name = model_name
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(model_name)
        # Use a fixed frame budget instead of the processor's default FPS sampling.
        self.processor.video_processor.fps = None

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            **model_kwargs,
        )
        self.model.eval()

    def generate_video_answer(
        self,
        video_path: str | Path,
        prompt: str,
        *,
        num_frames: int,
        max_new_tokens: int = 8,
    ) -> GenerationResult:
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        path = Path(video_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"video file not found: {path}")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "url": str(path),
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            processor_kwargs={
                "text_kwargs": {
                    "return_tensors": "pt",
                },
                "videos_kwargs": {
                    "do_sample_frames": True,
                    "num_frames": num_frames,
                },
            },
        )

        inputs.pop("token_type_ids", None)
        inputs = inputs.to(self.model.device)
        

        torch = self._torch
        using_cuda = torch.cuda.is_available()
        if using_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        if using_cuda:
            torch.cuda.synchronize()
        latency = time.perf_counter() - start

        input_length = int(inputs.input_ids.shape[-1])
        trimmed = generated_ids[:, input_length:]
        raw_output = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        peak_memory = None
        if using_cuda:
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)

        return GenerationResult(
            raw_output=raw_output,
            latency_seconds=latency,
            input_token_count=input_length,
            generated_token_count=int(trimmed.shape[-1]),
            peak_gpu_memory_mb=peak_memory,
        )
