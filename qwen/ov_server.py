#!/usr/bin/env python3
"""Serve LLaVA-OneVision-2-8B-Instruct as a drop-in VQ/MQ/TA video reward model.

This mirrors ``server.py`` (the Qwen3-VL judge) byte-for-byte on the HTTP layer:
it reuses the exact same request handler, JSON parsing, winner parsing, scoring
helpers and ``/compute`` + ``/compare`` endpoints.  The ONLY thing that changes is
the underlying judge model, so the existing ``pairwise_eval.py`` /
``pairwise_acc.py`` (and the pointwise ``eval.py`` / ``acc.py``) work unchanged.

This makes the OneVision-2-8B vs Qwen3-VL comparison fair: identical videos,
identical MQ prompt, identical parsing, identical accuracy script -- only the
vision-language judge differs.

Run inside the OneVision-2 env (transformers>=5.7.0):

    export OV_MODEL_PATH="lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct"
    export CUDA_VISIBLE_DEVICES=0
    bash GRPO/qwen/ov_server.sh

Then point the existing client at it (any env, only needs requests):

    python GRPO/qwen/pairwise_eval.py --skip-generate \
        --output-dir <dir with videos_meta.csv + videos/> \
        --server-url http://127.0.0.1:18080
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

import torch

# Reuse the Qwen judge's HTTP + parsing machinery so behaviour is identical.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server import (  # noqa: E402
    DIMENSIONS,
    LRUCache,
    RewardHTTPServer,
    RewardRequestHandler,
    ScoreResult,
    build_dimension_prompt,
    extract_first_json_object,
    extract_score_from_json,
    parse_dtype,
    parse_winner,
    read_prompt_file,
)

logger = logging.getLogger("ov2_reward_server")


class OneVision2Judge:
    """Same public surface as server.Qwen3VLJudge, backed by LLaVA-OneVision-2."""

    def __init__(self, args: argparse.Namespace):
        model_path = args.model_path or os.environ.get("OV_MODEL_PATH", "")
        if not model_path:
            raise ValueError(
                "OneVision-2 model path is empty. Set --model-path or OV_MODEL_PATH."
            )

        self.model_path = model_path
        self.device = args.device
        self.dtype = parse_dtype(args.dtype)
        self.max_new_tokens = int(args.max_new_tokens)
        self.video_nframes = args.video_nframes
        self.video_max_pixels = args.video_max_pixels
        self.video_backend = args.video_backend
        self.snap_scores = bool(args.snap_scores)
        self.prompts = self._load_prompts(args)
        self.model_lock = threading.Lock()

        self.processor, self.model = self._load_model(args)
        logger.info(
            "OneVision-2 judge loaded model=%s device=%s dtype=%s backend=%s nframes=%s",
            self.model_path,
            self.device,
            self.dtype,
            self.video_backend,
            self.video_nframes,
        )

    def _load_prompts(self, args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
        prompt_dir = os.path.abspath(args.prompt_dir)
        include_overrides = {
            "VQ": args.vq_include_prompt,
            "MQ": args.mq_include_prompt,
            "TA": args.ta_include_prompt,
        }
        out: Dict[str, Dict[str, Any]] = {}
        for dim, spec in DIMENSIONS.items():
            out[dim] = {
                **spec,
                "template": read_prompt_file(prompt_dir, spec["prompt_file"]),
                "include_prompt": include_overrides[dim],
            }
        return out

    def _load_model(self, args: argparse.Namespace):
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "OneVision-2 reward server requires transformers>=5.7.0 with "
                "AutoModelForImageTextToText support."
            ) from e

        processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        if self.video_max_pixels is not None and hasattr(processor, "video_processor"):
            processor.video_processor.max_pixels = int(self.video_max_pixels)

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": self.dtype,
        }
        attn = self._resolve_attn_implementation(args.attn_implementation)
        if attn:
            model_kwargs["attn_implementation"] = attn
        device_map = self._resolve_device_map(args.device_map_mode)
        if device_map is not None:
            model_kwargs["device_map"] = device_map

        model = AutoModelForImageTextToText.from_pretrained(self.model_path, **model_kwargs)
        if device_map is None:
            model.to(self.device)
        model.eval()
        return processor, model

    def _resolve_device_map(self, mode: str):
        value = str(mode or "auto").lower()
        if value in {"none", "false", "off"}:
            return None
        if value in {"auto", "balanced", "balanced_low_0", "sequential"}:
            return value
        if value in {"single", "cuda", "device"}:
            return {"": self.device}
        raise ValueError(
            "--device-map-mode must be one of auto, single, none, balanced, "
            f"balanced_low_0, sequential; got {mode!r}"
        )

    def _resolve_attn_implementation(self, requested: str) -> Optional[str]:
        value = str(requested or "").lower()
        if value in {"", "none", "default"}:
            return None
        if value != "auto":
            return requested
        try:
            import flash_attn  # type: ignore  # noqa: F401

            logger.info("using flash_attention_2 attention backend")
            return "flash_attention_2"
        except Exception:
            logger.info("flash-attn unavailable; falling back to sdpa attention backend")
            return "sdpa"

    # ----- public surface identical to Qwen3VLJudge -----

    def score_video(
        self,
        video_path: str,
        prompt: str,
        *,
        vq_coef: float,
        mq_coef: float,
        ta_coef: float,
        score_scale: str,
        fallback_value: float,
        return_responses: bool,
        dimensions: Tuple[str, ...] = ("VQ", "MQ", "TA"),
    ) -> Dict[str, Any]:
        raw: Dict[str, float] = {}
        scaled: Dict[str, float] = {}
        responses: Dict[str, Any] = {}
        errors: Dict[str, str] = {}

        for dim in dimensions:
            result = self._score_dimension(
                dim=dim,
                video_path=video_path,
                prompt=prompt,
                score_scale=score_scale,
                fallback_value=fallback_value,
            )
            raw[dim] = result.raw_score
            scaled[dim] = result.scaled_score
            if result.error:
                errors[dim] = result.error
            if return_responses:
                responses[dim] = {
                    "text": result.response_text,
                    "json": result.parsed_json,
                    "error": result.error,
                }

        coef_map = {"VQ": float(vq_coef), "MQ": float(mq_coef), "TA": float(ta_coef)}
        composite = sum(coef_map[dim] * scaled[dim] for dim in dimensions)
        metric: Dict[str, Any] = {
            "score_scale": score_scale,
            "dimensions": list(dimensions),
        }
        for dim in dimensions:
            metric[dim] = scaled[dim]
            metric[f"{dim}_raw"] = raw[dim]
        metric["composite"] = composite
        if errors:
            metric["errors"] = errors
        if return_responses:
            metric["responses"] = responses
        return metric

    def compare_videos(
        self,
        video_a_path: str,
        video_b_path: str,
        instruction: str,
        *,
        prompt: Optional[str] = None,
        return_responses: bool = False,
    ) -> Dict[str, Any]:
        text_prompt = instruction.rstrip()
        if prompt:
            text_prompt = f"{text_prompt}\n{prompt.strip()}"
        result: Dict[str, Any] = {}
        response_text = ""
        try:
            response_text = self._generate_pair(video_a_path, video_b_path, text_prompt)
            parsed = extract_first_json_object(response_text)
            result["winner"] = parse_winner(parsed)
        except Exception as e:  # noqa: BLE001
            logger.exception("failed to compare %s vs %s: %s", video_a_path, video_b_path, e)
            result["winner"] = None
            result["error"] = str(e)
        if return_responses:
            result["response"] = response_text
        return result

    def _score_dimension(
        self,
        *,
        dim: str,
        video_path: str,
        prompt: str,
        score_scale: str,
        fallback_value: float,
    ) -> ScoreResult:
        spec = self.prompts[dim]
        judge_prompt = build_dimension_prompt(
            spec["template"],
            prompt,
            include_prompt=bool(spec["include_prompt"]),
        )
        try:
            response_text = self._generate(video_path, judge_prompt)
            parsed = extract_first_json_object(response_text)
            raw_score = extract_score_from_json(
                parsed,
                spec["json_key"],
                snap_scores=self.snap_scores,
            )
            return ScoreResult(
                raw_score=raw_score,
                scaled_score=self._scale_score(raw_score, score_scale),
                response_text=response_text,
                parsed_json=parsed,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("failed to score %s for %s: %s", dim, video_path, e)
            raw_score = float(fallback_value)
            return ScoreResult(
                raw_score=raw_score,
                scaled_score=self._scale_score(raw_score, score_scale),
                response_text="",
                parsed_json=None,
                error=str(e),
            )

    def _scale_score(self, raw_score: float, score_scale: str) -> float:
        if score_scale == "raw":
            return float(raw_score)
        if score_scale == "unit":
            return float(raw_score) / 5.0
        raise ValueError(f"score_scale must be 'raw' or 'unit', got {score_scale!r}")

    # ----- OneVision-2 generation -----

    def _generate(self, video_path: str, text_prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        return self._run_generation(messages, [video_path])

    def _generate_pair(self, video_a_path: str, video_b_path: str, text_prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "视频A："},
                    {"type": "video", "video": video_a_path},
                    {"type": "text", "text": "视频B："},
                    {"type": "video", "video": video_b_path},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        return self._run_generation(messages, [video_a_path, video_b_path])

    def _run_generation(self, messages: List[Dict[str, Any]], video_paths: List[str]) -> str:
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        proc_kwargs: Dict[str, Any] = {"video_backend": self.video_backend}
        if self.video_nframes is not None:
            proc_kwargs["num_frames"] = int(self.video_nframes)
        if self.video_max_pixels is not None:
            proc_kwargs["max_pixels"] = int(self.video_max_pixels)

        inputs = self.processor(
            text=[text],
            videos=video_paths,
            return_tensors="pt",
            padding=True,
            **proc_kwargs,
        )
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with self.model_lock, torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        trimmed = generated_ids[:, input_len:]
        decoded = self.processor.tokenizer.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0] if decoded else ""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=os.environ.get("OV_MODEL_PATH", "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct"),
        help="OneVision-2 model path or HF id.",
    )
    parser.add_argument(
        "--prompt-dir",
        default="/Users/siyuan.fu/fsy/posttrain/GRPO/prompts",
        help="Directory containing aesthetic_quality.txt, motion_quality_noprompt.txt, "
        "and instruction_following.txt.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
    )
    parser.add_argument(
        "--device-map-mode",
        default="single",
        choices=["auto", "single", "none", "balanced", "balanced_low_0", "sequential"],
        help="How transformers places weights. single maps the whole 8B model onto --device.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="auto",
        help="Attention backend: auto, flash_attention_2, sdpa, or none.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--video-backend",
        default="frames",
        choices=["frames", "codec"],
        help="frames = uniform sampling (paper/model-card default for SHORT clips like Wan "
        "outputs, and apples-to-apples vs Qwen). codec = OneVision canvas packing "
        "(paper's headline mode, for long videos; needs codec-video-prep + ffmpeg).",
    )
    parser.add_argument(
        "--video-nframes",
        type=int,
        default=16,
        help="Frames sampled per video. 16 matches the LLaVA-OneVision-2-8B model-card "
        "video demo and the encoder's 16-frame eval setting.",
    )
    parser.add_argument(
        "--video-max-pixels",
        type=int,
        default=200704,
        help="Per-frame pixel budget. 200704 = 448x448, the value used in the "
        "LLaVA-OneVision-2-8B model-card video example. Lower if you hit OOM.",
    )
    parser.add_argument("--snap-scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vq-include-prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mq-include-prompt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ta-include-prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-size", type=int, default=512)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    judge = OneVision2Judge(args)
    cache = LRUCache(args.cache_size)
    server = RewardHTTPServer((args.host, args.port), RewardRequestHandler, judge, cache)
    logger.info("serving OneVision-2 reward server at http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
