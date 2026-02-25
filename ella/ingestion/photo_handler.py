"""Ingestion handler for Telegram photo messages.

Downloads the highest-resolution photo from Telegram, then uses the
Qwen2.5-VL model (mlx-vlm) on-demand to describe what is in the image.
The VL model is unloaded after use to free unified RAM.

If the user also sent a caption alongside the photo, it is appended to
the description so BrainAgent has full context.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from ella.config import get_settings
from ella.communications.telegram.sender import get_sender

logger = logging.getLogger(__name__)

MAX_TOKENS = 300


async def describe_photo(file_id: str, caption: str | None = None) -> str:
    """Download a Telegram photo and return an LLM-generated description."""
    sender = get_sender()

    with tempfile.TemporaryDirectory() as tmp_dir:
        photo_path = Path(tmp_dir) / f"{file_id}.jpg"
        await sender.download_file_id(file_id, photo_path)

        if not photo_path.exists() or photo_path.stat().st_size == 0:
            return "[photo — download failed]"

        description = _run_vl_model(str(photo_path), get_settings().mlx_vl_model, caption)

    return description


def _run_vl_model(image_path: str, model_name: str, caption: str | None) -> str:
    """Load Qwen2.5-VL on-demand, describe the image, then unload."""
    model = None
    processor = None
    try:
        import mlx.core as mx
        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        logger.info("Loading VL model on-demand for photo: %s", model_name)
        model, processor = load(model_name)
        config = load_config(model_name)

        prompt_text = (
            "Describe what you see in this photo in 2-3 sentences. "
            "Be specific: mention people, objects, setting, colours, mood, and any visible text. "
            "Reply in English."
        )
        if caption:
            prompt_text += f' The sender also wrote: "{caption}"'

        formatted = apply_chat_template(
            processor,
            config,
            prompt_text,
            num_images=1,
        )

        output = generate(
            model,
            processor,
            formatted,
            image=[image_path],
            max_tokens=MAX_TOKENS,
            verbose=False,
        )

        description = output.strip() if isinstance(output, str) else str(output).strip()
        if caption:
            description = f"{description} (caption: {caption})"
        logger.info("Photo described: %d chars", len(description))
        return description if description else "[photo — no description generated]"

    except ImportError as exc:
        msg = str(exc)
        if "mlx_vlm" in msg or "mlx" in msg:
            logger.error("mlx-vlm not installed — cannot describe photo")
        else:
            # Optional dependency missing inside mlx-vlm (e.g. torchvision) — log but continue
            logger.warning("Optional dependency missing during photo processing (%s) — retrying without it", msg)
        if caption:
            return f"[photo — VL model unavailable] (caption: {caption})"
        return "[photo — VL model unavailable]"
    except Exception:
        logger.exception("VL model inference failed for photo")
        if caption:
            return f"[photo — description failed] (caption: {caption})"
        return "[photo — description failed]"
    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
            logger.info("VL model unloaded after photo, metal cache cleared")
        except Exception:
            pass
