#!/usr/bin/env python3
"""Serve Qwen3-VL-72B as a VQ/MQ/TA video reward model.

The server loads the judge model once on a single GPU and exposes two endpoints:

  GET  /health
  POST /compute

``/compute`` expects JSON:

    {
      "prompts": ["text-to-video prompt", ...],
      "videos": ["<base64 mp4 bytes>", ...],
      "video_fps": 15,
      "vq_coef": 1.0,
      "mq_coef": 1.0,
      "ta_coef": 1.0,
      "score_scale": "raw",
      "fallback_value": 0.0,
      "return_responses": false
    }

It returns:

    {
      "error": null,
      "rewards": [7.0, ...],
      "metrics": [
        {"VQ": 3.0, "MQ": 1.0, "TA": 3.0, "composite": 7.0, ...}
      ]
    }

Model path is intentionally not hard-coded.  Pass it through ``--model-path`` or
``QWEN3VL_MODEL_PATH`` on the server.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import inspect
import json
import logging
import os
import re
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

logger = logging.getLogger("qwen3vl_reward_server")


DIMENSIONS: Dict[str, Dict[str, Any]] = {
    "VQ": {
        "prompt_file": "aesthetic_quality.txt",
        "json_key": "aesthetic_quality",
        "include_prompt": True,
    },
    "MQ": {
        "prompt_file": "motion_quality_noprompt.txt",
        "json_key": "motion_quality",
        "include_prompt": False,
    },
    "TA": {
        "prompt_file": "instruction_following.txt",
        "json_key": "instruction_following",
        "include_prompt": True,
    },
}

LEGAL_SCORES: Tuple[float, ...] = (0.0, 1.0, 3.0, 5.0)
PROMPT_MARKERS: Tuple[str, ...] = ("##视频prompt如下：", "##视频prompt如下:")


@dataclass
class ScoreResult:
    raw_score: float
    scaled_score: float
    response_text: str
    parsed_json: Optional[Dict[str, Any]]
    error: Optional[str] = None


class LRUCache:
    def __init__(self, max_size: int):
        self.max_size = max(0, int(max_size))
        self._data: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if self.max_size <= 0:
            return None
        with self._lock:
            value = self._data.get(key)
            if value is None:
                return None
            self._data.move_to_end(key)
            return value

    def put(self, key: str, value: Dict[str, Any]) -> None:
        if self.max_size <= 0:
            return
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


def parse_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype {name!r}; use bf16/fp16/fp32")


def read_prompt_file(prompt_dir: str, filename: str) -> str:
    path = os.path.join(prompt_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def strip_prompt_marker(template: str) -> str:
    text = template.rstrip()
    for marker in PROMPT_MARKERS:
        if text.endswith(marker):
            return text[: -len(marker)].rstrip()
    return text


def build_dimension_prompt(template: str, prompt: str, include_prompt: bool) -> str:
    if include_prompt:
        return f"{template.rstrip()}\n{prompt.strip()}"
    return strip_prompt_marker(template)


def remove_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def extract_first_json_object(text: str) -> Dict[str, Any]:
    cleaned = remove_markdown_fences(text)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        start = match.start()
        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"could not find a JSON object in response: {cleaned[:300]!r}")


def extract_numeric_score(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        raise TypeError(f"invalid score value: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            return float(match.group(0))
    raise TypeError(f"could not parse numeric score from {value!r}")


def nearest_legal_score(value: float) -> float:
    return min(LEGAL_SCORES, key=lambda item: abs(item - value))


def extract_score_from_json(
    payload: Mapping[str, Any],
    json_key: str,
    *,
    snap_scores: bool,
) -> float:
    section = payload.get(json_key)
    if isinstance(section, Mapping) and "score" in section:
        score = extract_numeric_score(section["score"])
    elif "score" in payload:
        score = extract_numeric_score(payload["score"])
    else:
        raise KeyError(f"missing score under key {json_key!r}")

    if snap_scores:
        return nearest_legal_score(score)
    if score not in LEGAL_SCORES:
        raise ValueError(
            f"score {score} is not one of {LEGAL_SCORES}; enable snap_scores to coerce"
        )
    return float(score)


def b64_to_file(data_b64: str, out_path: str) -> None:
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(data_b64))


def make_cache_key(
    video_b64: str,
    prompt: str,
    score_scale: str,
    vq_coef: float,
    mq_coef: float,
    ta_coef: float,
) -> str:
    h = hashlib.sha256()
    h.update(video_b64.encode("utf-8"))
    h.update(b"\0")
    h.update(prompt.encode("utf-8"))
    h.update(b"\0")
    h.update(score_scale.encode("utf-8"))
    h.update(b"\0")
    h.update(f"{vq_coef:.12g},{mq_coef:.12g},{ta_coef:.12g}".encode("utf-8"))
    return h.hexdigest()


class Qwen3VLJudge:
    def __init__(self, args: argparse.Namespace):
        model_path = args.model_path or os.environ.get("QWEN3VL_MODEL_PATH", "")
        if not model_path:
            raise ValueError(
                "Qwen3-VL model path is empty. Set --model-path or QWEN3VL_MODEL_PATH."
            )

        self.model_path = model_path
        self.device = args.device
        self.dtype = parse_dtype(args.dtype)
        self.max_new_tokens = int(args.max_new_tokens)
        self.video_fps = float(args.video_fps)
        self.video_min_pixels = args.video_min_pixels
        self.video_max_pixels = args.video_max_pixels
        self.video_total_pixels = args.video_total_pixels
        self.video_nframes = args.video_nframes
        self.snap_scores = bool(args.snap_scores)
        self.prompts = self._load_prompts(args)
        self.model_lock = threading.Lock()

        self.processor, self.model, self.process_vision_info = self._load_model(args)
        logger.info(
            "Qwen3-VL judge loaded model=%s device=%s dtype=%s prompt_dir=%s",
            self.model_path,
            self.device,
            self.dtype,
            args.prompt_dir,
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
                "Qwen3-VL reward server requires transformers with "
                "AutoModelForImageTextToText support."
            ) from e

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as e:
            raise ImportError(
                "Qwen3-VL video scoring requires qwen-vl-utils. "
                "Install the version recommended by Qwen3-VL, e.g. "
                "`pip install qwen-vl-utils`."
            ) from e

        processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=args.trust_remote_code,
        )

        model_kwargs: Dict[str, Any] = {
            "torch_dtype": self.dtype,
            "trust_remote_code": args.trust_remote_code,
        }
        attn_implementation = self._resolve_attn_implementation(args.attn_implementation)
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        device_map = self._resolve_device_map(args.device_map_mode)
        if device_map is not None:
            model_kwargs["device_map"] = device_map

        model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            **model_kwargs,
        )
        if device_map is None:
            model.to(self.device)
        model.eval()
        return processor, model, process_vision_info

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
    ) -> Dict[str, Any]:
        raw: Dict[str, float] = {}
        scaled: Dict[str, float] = {}
        responses: Dict[str, Any] = {}
        errors: Dict[str, str] = {}

        for dim in ("VQ", "MQ", "TA"):
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

        composite = (
            float(vq_coef) * scaled["VQ"]
            + float(mq_coef) * scaled["MQ"]
            + float(ta_coef) * scaled["TA"]
        )
        metric = {
            "VQ": scaled["VQ"],
            "MQ": scaled["MQ"],
            "TA": scaled["TA"],
            "VQ_raw": raw["VQ"],
            "MQ_raw": raw["MQ"],
            "TA_raw": raw["TA"],
            "composite": composite,
            "score_scale": score_scale,
        }
        if errors:
            metric["errors"] = errors
        if return_responses:
            metric["responses"] = responses
        return metric

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
            logger.warning("failed to score %s for %s: %s", dim, video_path, e)
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

    def _generate(self, video_path: str, text_prompt: str) -> str:
        video_item: Dict[str, Any] = {
            "type": "video",
            "video": video_path,
            "fps": self.video_fps,
        }
        if self.video_min_pixels is not None:
            video_item["min_pixels"] = int(self.video_min_pixels)
        if self.video_max_pixels is not None:
            video_item["max_pixels"] = int(self.video_max_pixels)
        if self.video_total_pixels is not None:
            video_item["total_pixels"] = int(self.video_total_pixels)
        if self.video_nframes is not None:
            video_item["nframes"] = int(self.video_nframes)

        messages = [
            {
                "role": "user",
                "content": [
                    video_item,
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs, video_kwargs = self._process_vision(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to(self.device)

        with self.model_lock, torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        input_ids = inputs["input_ids"]
        trimmed = [
            output_ids[len(input_ids[idx]) :]
            for idx, output_ids in enumerate(generated_ids)
        ]
        decoded = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0] if decoded else ""

    def _process_vision(self, messages: List[Dict[str, Any]]):
        sig = inspect.signature(self.process_vision_info)
        supported = sig.parameters
        kwargs: Dict[str, Any] = {}
        if "image_patch_size" in supported:
            kwargs["image_patch_size"] = 16
        if "return_video_kwargs" in supported:
            kwargs["return_video_kwargs"] = True
        if "return_video_metadata" in supported:
            kwargs["return_video_metadata"] = True

        result = self.process_vision_info(messages, **kwargs)
        if not isinstance(result, tuple):
            raise TypeError(
                "qwen_vl_utils.process_vision_info returned unexpected "
                f"{type(result).__name__}"
            )
        if len(result) == 2:
            image_inputs, video_inputs = result
            video_kwargs = {}
        elif len(result) >= 3:
            image_inputs, video_inputs = result[0], result[1]
            video_kwargs = result[2] if isinstance(result[2], dict) else {}
        else:
            raise ValueError(f"unexpected process_vision_info result length: {len(result)}")
        return image_inputs, video_inputs, video_kwargs


class RewardHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: Tuple[str, int],
        handler_cls,
        judge: Qwen3VLJudge,
        cache: LRUCache,
    ):
        super().__init__(server_address, handler_cls)
        self.judge = judge
        self.cache = cache


class RewardRequestHandler(BaseHTTPRequestHandler):
    server: RewardHTTPServer

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/health":
            self._send_json(
                {
                    "status": "ok",
                    "model_path": self.server.judge.model_path,
                    "device": self.server.judge.device,
                    "cache_size": len(self.server.cache),
                    "dimensions": {
                        dim: {
                            "prompt_file": spec["prompt_file"],
                            "json_key": spec["json_key"],
                            "include_prompt": spec["include_prompt"],
                        }
                        for dim, spec in self.server.judge.prompts.items()
                    },
                }
            )
            return
        self._send_json({"error": f"unknown endpoint: {self.path}"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/compute":
            self._send_json({"error": f"unknown endpoint: {self.path}"}, HTTPStatus.NOT_FOUND)
            return

        try:
            payload = self._read_json()
            result = self._handle_compute(payload)
            self._send_json(result)
        except Exception as e:  # noqa: BLE001
            logger.exception("request failed")
            self._send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        if not raw:
            raise ValueError("empty request body")
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise TypeError("request body must be a JSON object")
        return data

    def _handle_compute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompts = payload.get("prompts", payload.get("prompt"))
        videos = payload.get("videos", payload.get("video"))
        if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            raise TypeError("`prompts` must be a list[str]")
        if not isinstance(videos, list) or not all(isinstance(v, str) for v in videos):
            raise TypeError("`videos` must be a list[str] of base64-encoded mp4 bytes")
        if len(prompts) != len(videos):
            raise ValueError(
                f"len(prompts)={len(prompts)} != len(videos)={len(videos)}"
            )

        score_scale = str(payload.get("score_scale", "raw")).lower()
        if score_scale not in {"raw", "unit"}:
            raise ValueError("score_scale must be 'raw' or 'unit'")

        vq_coef = float(payload.get("vq_coef", 1.0))
        mq_coef = float(payload.get("mq_coef", 1.0))
        ta_coef = float(payload.get("ta_coef", 1.0))
        fallback_value = float(payload.get("fallback_value", 0.0))
        return_responses = bool(payload.get("return_responses", False))

        rewards: List[float] = []
        metrics: List[Dict[str, Any]] = []

        with tempfile.TemporaryDirectory(prefix="qwen3vl_reward_server_") as tmpdir:
            for idx, (prompt, video_b64) in enumerate(zip(prompts, videos)):
                cache_key = make_cache_key(
                    video_b64,
                    prompt,
                    score_scale,
                    vq_coef,
                    mq_coef,
                    ta_coef,
                )
                cached = self.server.cache.get(cache_key)
                if cached is not None:
                    metric = dict(cached)
                    if not return_responses:
                        metric.pop("responses", None)
                    metrics.append(metric)
                    rewards.append(float(metric["composite"]))
                    continue

                video_path = os.path.abspath(os.path.join(tmpdir, f"sample_{idx:04d}.mp4"))
                b64_to_file(video_b64, video_path)
                metric = self.server.judge.score_video(
                    video_path,
                    prompt,
                    vq_coef=vq_coef,
                    mq_coef=mq_coef,
                    ta_coef=ta_coef,
                    score_scale=score_scale,
                    fallback_value=fallback_value,
                    return_responses=return_responses,
                )
                if "errors" not in metric:
                    self.server.cache.put(cache_key, dict(metric))
                metrics.append(metric)
                rewards.append(float(metric["composite"]))

        return {
            "error": None,
            "rewards": rewards,
            "metrics": metrics,
        }

    def _send_json(
        self,
        payload: Dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=os.environ.get("QWEN3VL_MODEL_PATH", ""),
        help="Qwen3-VL model path. Leave empty here and set QWEN3VL_MODEL_PATH later.",
    )
    parser.add_argument(
        "--prompt-dir",
        default="/Users/siyuan.fu/fsy/posttrain/prompts",
        help="Directory containing aesthetic_quality.txt, motion_quality_noprompt.txt, "
        "and instruction_following.txt.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument(
        "--device-map-mode",
        default="auto",
        choices=["auto", "single", "none", "balanced", "balanced_low_0", "sequential"],
        help="How transformers places Qwen3-VL weights. auto is safest for large/meta-loaded models; "
        "single maps the whole model to --device; none loads on CPU then calls .to(--device).",
    )
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--attn-implementation",
        default="auto",
        help="Attention backend passed to transformers: auto, flash_attention_2, sdpa, or none.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--video-fps", type=float, default=1.0)
    parser.add_argument("--video-min-pixels", type=int, default=None)
    parser.add_argument("--video-max-pixels", type=int, default=None)
    parser.add_argument("--video-total-pixels", type=int, default=None)
    parser.add_argument("--video-nframes", type=int, default=None)
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
    judge = Qwen3VLJudge(args)
    cache = LRUCache(args.cache_size)
    server = RewardHTTPServer((args.host, args.port), RewardRequestHandler, judge, cache)
    logger.info("serving Qwen3-VL reward server at http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
