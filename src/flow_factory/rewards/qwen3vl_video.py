# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# src/flow_factory/rewards/qwen3vl_video.py
"""
HTTP client reward wrapper for a Qwen3-VL video judge server.

This module intentionally does not load Qwen3-VL in the training process.  The
72B judge should run as a single long-lived server on a dedicated GPU; every
Flow-Factory rank sends generated videos to that server and receives scalar
rewards.

YAML example:
    rewards:
      - name: "qwen3vl"
        reward_model: "qwen3vl_video"
        weight: 1.0
        batch_size: 1
        server_url: "http://127.0.0.1:18080"
        timeout: 600
        retry_attempts: 3
        video_fps: 15
        vq_coef: 1.0
        mq_coef: 1.0
        ta_coef: 1.0
        score_scale: "raw"  # raw: 0/1/3/5 per dim; unit: divide each dim by 5
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
from accelerate import Accelerator
from PIL import Image

from .abc import PointwiseRewardModel, RewardModelOutput
from ..hparams import RewardArguments

logger = logging.getLogger(__name__)


def _frames_to_mp4(frames: List[Image.Image], out_path: str, fps: int) -> None:
    """Write PIL frames to mp4 using diffusers first, imageio as fallback."""
    try:
        from diffusers.utils import export_to_video  # type: ignore

        export_to_video(frames, out_path, fps=fps)
        return
    except Exception:
        pass

    import imageio  # type: ignore
    import numpy as np

    arr = [np.array(frame.convert("RGB")) for frame in frames]
    imageio.mimsave(out_path, arr, fps=fps, codec="libx264", quality=8)


def _file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class Qwen3VLRewardClient:
    """Small retrying client for the Qwen3-VL reward server."""

    def __init__(
        self,
        server_url: str,
        timeout: float = 600.0,
        retries: int = 3,
        retry_sleep: float = 2.0,
    ):
        self.url = server_url.rstrip("/")
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.retry_sleep = float(retry_sleep)
        self._session = requests.Session()

    def health_check(self) -> Dict[str, Any]:
        r = self._session.get(f"{self.url}/health", timeout=min(self.timeout, 10.0))
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "ok":
            raise RuntimeError(f"unhealthy Qwen3-VL reward server: {data}")
        return data

    def compute(
        self,
        prompts: List[str],
        video_b64: List[str],
        *,
        video_fps: int,
        vq_coef: float,
        mq_coef: float,
        ta_coef: float,
        score_scale: str,
        fallback_value: float,
        return_responses: bool,
    ) -> Dict[str, Any]:
        payload = {
            "prompts": prompts,
            "videos": video_b64,
            "video_fps": video_fps,
            "vq_coef": vq_coef,
            "mq_coef": mq_coef,
            "ta_coef": ta_coef,
            "score_scale": score_scale,
            "fallback_value": fallback_value,
            "return_responses": return_responses,
        }

        last_err: Optional[BaseException] = None
        for attempt in range(self.retries):
            try:
                r = self._session.post(
                    f"{self.url}/compute",
                    json=payload,
                    timeout=self.timeout,
                )
                r.raise_for_status()
                data = r.json()
                if data.get("error"):
                    raise RuntimeError(str(data["error"]))
                return data
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt + 1 >= self.retries:
                    break
                sleep_s = self.retry_sleep * (2**attempt)
                logger.warning(
                    "Qwen3-VL reward server request failed on attempt %s/%s: %s; "
                    "retrying in %.1fs",
                    attempt + 1,
                    self.retries,
                    e,
                    sleep_s,
                )
                time.sleep(sleep_s)

        raise RuntimeError(
            f"failed to query Qwen3-VL reward server after {self.retries} attempt(s): "
            f"{last_err}"
        )


class Qwen3VLVideoRewardModel(PointwiseRewardModel):
    """
    Pointwise video reward backed by a Qwen3-VL judge server.

    The server returns VQ/MQ/TA scores and a composite scalar.  Flow-Factory
    consumes only ``rewards`` for GRPO; this wrapper logs the per-dimension
    means directly to wandb when a wandb run is active.
    """

    required_fields: Tuple[str, ...] = ("prompt", "video")
    use_tensor_inputs = False

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        server_url = getattr(config, "server_url", None) or os.environ.get(
            "QWEN3VL_REWARD_SERVER_URL", "http://127.0.0.1:18080"
        )
        self.video_fps = int(getattr(config, "video_fps", 15))
        self.vq_coef = float(getattr(config, "vq_coef", 1.0))
        self.mq_coef = float(getattr(config, "mq_coef", 1.0))
        self.ta_coef = float(getattr(config, "ta_coef", 1.0))
        self.score_scale = str(getattr(config, "score_scale", "raw")).lower()
        if self.score_scale not in {"raw", "unit"}:
            raise ValueError(
                f"score_scale must be 'raw' or 'unit', got {self.score_scale!r}"
            )
        self.fallback_value = float(getattr(config, "fallback_value", 0.0))
        self.return_responses = bool(getattr(config, "return_responses", False))
        self.log_dim_metrics = bool(getattr(config, "log_dim_metrics", True))

        self.client = Qwen3VLRewardClient(
            server_url=server_url,
            timeout=float(getattr(config, "timeout", 600.0)),
            retries=int(getattr(config, "retry_attempts", 3)),
            retry_sleep=float(getattr(config, "retry_sleep", 2.0)),
        )
        health = self.client.health_check()
        logger.info(
            "[Qwen3VLVideoReward] connected to %s model=%s video_fps=%s "
            "coefs=(%.3f,%.3f,%.3f) score_scale=%s",
            server_url,
            health.get("model_path", "<unknown>"),
            self.video_fps,
            self.vq_coef,
            self.mq_coef,
            self.ta_coef,
            self.score_scale,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        audio: Any = None,
        condition_images: Any = None,
        condition_videos: Any = None,
        **kwargs,
    ) -> RewardModelOutput:
        if video is None:
            raise ValueError("Qwen3VLVideoRewardModel requires video input.")
        if not isinstance(prompt, list):
            prompt = [prompt]
        if len(prompt) != len(video):
            raise ValueError(
                f"len(prompt)={len(prompt)} != len(video)={len(video)}"
            )

        video_b64: List[str] = []
        with tempfile.TemporaryDirectory(prefix="qwen3vl_reward_") as tmpdir:
            for idx, frames in enumerate(video):
                mp4_path = os.path.join(tmpdir, f"sample_{idx:04d}.mp4")
                _frames_to_mp4(frames, mp4_path, fps=self.video_fps)
                video_b64.append(_file_to_b64(mp4_path))

        data = self.client.compute(
            prompts=prompt,
            video_b64=video_b64,
            video_fps=self.video_fps,
            vq_coef=self.vq_coef,
            mq_coef=self.mq_coef,
            ta_coef=self.ta_coef,
            score_scale=self.score_scale,
            fallback_value=self.fallback_value,
            return_responses=self.return_responses,
        )

        rewards = data.get("rewards")
        if not isinstance(rewards, list) or len(rewards) != len(prompt):
            raise RuntimeError(
                "Qwen3-VL reward server returned invalid rewards: "
                f"expected list length {len(prompt)}, got {type(rewards).__name__} "
                f"with value {rewards!r}"
            )

        metrics = data.get("metrics") or []
        if self.log_dim_metrics and metrics:
            self._maybe_log_dim_metrics(metrics)

        extra_info = {
            "metrics": metrics,
            "score_scale": self.score_scale,
            "composite_coefs": {
                "vq": self.vq_coef,
                "mq": self.mq_coef,
                "ta": self.ta_coef,
            },
        }
        return RewardModelOutput(
            rewards=torch.tensor(rewards, dtype=torch.float32, device=self.device),
            extra_info=extra_info,
        )

    def _maybe_log_dim_metrics(self, metrics: List[Dict[str, Any]]) -> None:
        try:
            import wandb  # type: ignore
        except Exception:
            return
        if wandb.run is None:
            return

        def mean_key(key: str) -> Optional[float]:
            vals = []
            for item in metrics:
                value = item.get(key)
                if isinstance(value, (int, float)):
                    vals.append(float(value))
            if not vals:
                return None
            return sum(vals) / len(vals)

        payload: Dict[str, Any] = {}
        for key in ("VQ", "MQ", "TA", "composite"):
            val = mean_key(key)
            if val is not None:
                payload[f"reward/qwen3vl_{key}_mean"] = val
        if not payload:
            return

        # Use a content-derived monotonic-ish local call count without touching
        # the trainer's explicit step.  Each process owns its own client object.
        counter = getattr(self, "_reward_call_counter", 0) + 1
        self._reward_call_counter = counter
        try:
            if not getattr(self, "_wandb_metrics_defined", False):
                wandb.define_metric("reward/qwen3vl_call")
                wandb.define_metric("reward/qwen3vl_*", step_metric="reward/qwen3vl_call")
                self._wandb_metrics_defined = True
            payload["reward/qwen3vl_call"] = counter
            wandb.log(payload, commit=False)
        except Exception as e:  # noqa: BLE001
            logger.debug("Qwen3-VL dim-metric wandb log skipped: %s", e)


def video_hash_for_cache(
    video_b64: str,
    prompt: str,
    score_scale: str = "raw",
    vq_coef: float = 1.0,
    mq_coef: float = 1.0,
    ta_coef: float = 1.0,
) -> str:
    """Hash helper shared by tests and compatible with the server cache key."""
    h = hashlib.sha256()
    h.update(video_b64.encode("utf-8"))
    h.update(b"\0")
    h.update(prompt.encode("utf-8"))
    h.update(b"\0")
    h.update(score_scale.encode("utf-8"))
    h.update(b"\0")
    h.update(f"{vq_coef:.12g},{mq_coef:.12g},{ta_coef:.12g}".encode("utf-8"))
    return h.hexdigest()
