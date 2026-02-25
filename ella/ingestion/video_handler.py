"""Ingestion handler for Telegram video messages.

Downloads the video, extracts key frames using OpenCV,
then uses Qwen2.5-VL (mlx-vlm) on-demand to produce a text summary.
The VL model is explicitly unloaded after use to free unified RAM.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from ella.config import get_settings
from ella.communications.telegram.sender import get_sender

logger = logging.getLogger(__name__)

FRAME_COUNT = 4      # number of evenly-spaced frames to sample
MAX_TOKENS = 256     # summary length cap


async def summarise_video(file_id: str) -> str:
    """Download a Telegram video and return an LLM-generated text summary."""
    settings = get_settings()
    sender = get_sender()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        video_path = tmp / f"{file_id}.mp4"
        await sender.download_file_id(file_id, video_path)

        frame_paths = _extract_frames(str(video_path), tmp, FRAME_COUNT)
        if not frame_paths:
            return "[video message — could not extract frames]"

        summary = _run_vl_model(frame_paths, settings.mlx_vl_model)

    return summary


def _extract_frames(video_path: str, dest_dir: Path, count: int) -> list[str]:
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []

        indices = [int(total * i / count) for i in range(count)]
        saved: list[str] = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            path = str(dest_dir / f"frame_{idx:06d}.jpg")
            cv2.imwrite(path, frame)
            saved.append(path)

        cap.release()
        return saved

    except ImportError:
        logger.error("opencv-python-headless not installed. Run: pip install opencv-python-headless")
        return []
    except Exception:
        logger.exception("Frame extraction failed")
        return []


def _run_vl_model(frame_paths: list[str], model_name: str) -> str:
    """Load Qwen2.5-VL on-demand, summarise frames, then unload."""
    model = None
    processor = None
    try:
        import mlx.core as mx
        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        logger.info("Loading VL model on-demand: %s", model_name)
        model, processor = load(model_name)
        config = load_config(model_name)

        prompt_text = (
            "Describe what is happening in these video frames in 2-3 sentences. "
            "Be concise and focus on the main action, people, and setting. "
            "If there is text visible, include it. "
            "Reply in the same language as any text visible in the frames, "
            "defaulting to English if unsure."
        )

        formatted = apply_chat_template(
            processor,
            config,
            prompt_text,
            num_images=len(frame_paths),
        )

        output = generate(
            model,
            processor,
            formatted,
            image=frame_paths,
            max_tokens=MAX_TOKENS,
            verbose=False,
        )

        summary = output.strip() if isinstance(output, str) else str(output).strip()
        logger.info("Video summarised: %d chars", len(summary))
        return summary if summary else "[video message — no summary generated]"

    except ImportError:
        logger.error("mlx-vlm not installed. Run: pip install mlx-vlm")
        return "[video message — mlx-vlm not available]"
    except Exception:
        logger.exception("VL model inference failed")
        return "[video message — summarisation failed]"
    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
            logger.info("VL model unloaded, metal cache cleared")
        except Exception:
            pass
