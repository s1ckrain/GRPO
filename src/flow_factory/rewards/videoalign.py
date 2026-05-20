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

# src/flow_factory/rewards/videoalign.py
"""
VideoAlign (KwaiVGI/VideoReward) reward model for Flow-Factory.

Strictly follows DanceGRPO official implementation
(`DanceGRPO/fastvideo/train_grpo_hunyuan.py:260-278`):

    reward = inferencer.reward([video_path], [prompt], use_norm=True)
    composite = vq_coef * VQ + mq_coef * MQ + ta_coef * TA
    # Default coefs (1.0, 1.0, 1.0) match:
    #   - SAGE-GRPO codebase default (rewards.py:383)
    #   - VideoAlign's built-in `Overall` = VQ + MQ + TA (inference.py:201)
    # On any reward computation failure -> sentinel -1.0 (DanceGRPO's fallback).

Usage in YAML config:
    rewards:
      - name: "videoalign"
        reward_model: "videoalign"
        batch_size: 1                # DanceGRPO calls reward() one video at a time
        # Extra kwargs:
        videoalign_dir: "/aigc/posttrain/siyuanfu/VideoAlign"
        checkpoint_dir: null         # default = {videoalign_dir}/checkpoints
        vq_coef: 1.0
        mq_coef: 1.0
        ta_coef: 1.0
        use_norm: true               # DanceGRPO uses True
        video_fps: 15                # fps for writing temp mp4s (match Wan2.1's fps)
        return_metrics: true         # expose VQ/MQ/TA in extra_info for logging
"""
from __future__ import annotations

import os
import sys
import logging
import tempfile
from typing import List, Optional

import torch
from PIL import Image
from accelerate import Accelerator

from .abc import PointwiseRewardModel, RewardModelOutput
from ..hparams import RewardArguments

logger = logging.getLogger(__name__)


def _import_videoalign(videoalign_dir: str):
    """
    Import VideoVLMRewardInference from a VideoAlign repo checkout.

    VideoAlign uses *top-level* imports (e.g. `from inference import ...` in
    `eval_videogen_rewardbench.py`), so we put its root on sys.path and import
    the class. This mirrors the DanceGRPO integration pattern.
    """
    videoalign_dir = os.path.abspath(videoalign_dir)
    if not os.path.isdir(videoalign_dir):
        raise FileNotFoundError(
            f"videoalign_dir does not exist: {videoalign_dir}. "
            "Set it via reward extra_kwargs (e.g. videoalign_dir: /path/to/VideoAlign)."
        )
    if not os.path.isfile(os.path.join(videoalign_dir, "inference.py")):
        raise FileNotFoundError(
            f"inference.py not found under {videoalign_dir}. "
            "Pass the VideoAlign repo root (the one with inference.py / checkpoints/)."
        )
    if videoalign_dir not in sys.path:
        sys.path.insert(0, videoalign_dir)

    # Re-import after path mutation. VideoAlign's `inference` module top-level-imports
    # `vision_process`, `data`, `utils`, etc., so all must resolve from the same dir.
    from inference import VideoVLMRewardInference  # type: ignore
    return VideoVLMRewardInference


def _frames_to_mp4(frames: List[Image.Image], out_path: str, fps: int) -> None:
    """
    Write a list of PIL frames to an mp4. Uses diffusers.utils.export_to_video
    when available (matches DanceGRPO's approach exactly), with imageio fallback.
    """
    try:
        from diffusers.utils import export_to_video  # type: ignore
        export_to_video(frames, out_path, fps=fps)
        return
    except Exception:  # noqa: BLE001
        pass
    # Fallback: imageio (only used if diffusers is too old to expose export_to_video).
    import numpy as np
    import imageio  # type: ignore
    arr = [np.array(f.convert("RGB")) for f in frames]
    imageio.mimsave(out_path, arr, fps=fps, codec="libx264", quality=8)


class VideoAlignRewardModel(PointwiseRewardModel):
    """
    Pointwise reward model wrapping KwaiVGI/VideoReward (VideoAlign).

    Composite reward = vq_coef * VQ + mq_coef * MQ + ta_coef * TA
    (defaults 1:1:1, equivalent to the model's built-in `Overall` field).
    """

    required_fields = ("prompt", "video")
    use_tensor_inputs = False  # We need PIL frames so we can dump them to mp4.

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        # ---- Read extra kwargs with sensible DanceGRPO-aligned defaults --------
        videoalign_dir: str = getattr(config, "videoalign_dir", None) or os.environ.get(
            "VIDEOALIGN_DIR", "/aigc/posttrain/siyuanfu/VideoAlign"
        )
        checkpoint_dir: Optional[str] = getattr(config, "checkpoint_dir", None)
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(videoalign_dir, "checkpoints")

        # Composite weights (default 1:1:1 = SAGE-GRPO codebase default)
        self.vq_coef: float = float(getattr(config, "vq_coef", 1.0))
        self.mq_coef: float = float(getattr(config, "mq_coef", 1.0))
        self.ta_coef: float = float(getattr(config, "ta_coef", 1.0))

        # Inference knobs
        self.use_norm: bool = bool(getattr(config, "use_norm", True))   # DanceGRPO uses True
        self.video_fps: int = int(getattr(config, "video_fps", 15))     # match Wan2.1 default fps
        self.return_metrics: bool = bool(getattr(config, "return_metrics", True))
        self.fallback_value: float = float(getattr(config, "fallback_value", -1.0))  # DanceGRPO sentinel

        # ---- Load VideoVLMRewardInference -------------------------------------
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(
                f"VideoAlign checkpoint_dir not found: {checkpoint_dir}. "
                f"Expected layout: {videoalign_dir}/checkpoints/{{model_config.json, checkpoint-*/}}"
            )
        VideoVLMRewardInference = _import_videoalign(videoalign_dir)

        # Match DanceGRPO call: VideoVLMRewardInference(load_from_pretrained, device=..., dtype=bf16)
        self.inferencer = VideoVLMRewardInference(
            load_from_pretrained=checkpoint_dir,
            device=str(self.device),
            dtype=self.dtype,
        )
        logger.info(
            f"[VideoAlign] loaded ckpt={checkpoint_dir} device={self.device} dtype={self.dtype} "
            f"composite coefs (VQ,MQ,TA)=({self.vq_coef},{self.mq_coef},{self.ta_coef}) "
            f"use_norm={self.use_norm} video_fps={self.video_fps}"
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        audio=None,
        condition_images=None,
        condition_videos=None,
        **kwargs,
    ) -> RewardModelOutput:
        if video is None:
            raise ValueError(
                "VideoAlignRewardModel requires `video` input (List[List[PIL.Image]]). "
                "Got video=None."
            )
        if not isinstance(prompt, list):
            prompt = [prompt]
        if len(video) != len(prompt):
            raise ValueError(
                f"len(video)={len(video)} != len(prompt)={len(prompt)}; the two must align."
            )

        device = self.device
        vq_list: List[float] = []
        mq_list: List[float] = []
        ta_list: List[float] = []
        composite_list: List[float] = []

        # DanceGRPO calls inferencer.reward([single_path], [single_prompt]) one at a time,
        # with try/except fallback to -1.0. We replicate that exactly to keep the reward
        # signal byte-identical with their training-time behaviour.
        with tempfile.TemporaryDirectory(prefix="videoalign_") as tmpdir:
            for i, (frames, pmt) in enumerate(zip(video, prompt)):
                mp4_path = os.path.join(tmpdir, f"sample_{i:04d}.mp4")
                try:
                    _frames_to_mp4(frames, mp4_path, fps=self.video_fps)
                    # Critical: pass absolute path (DanceGRPO uses os.path.abspath).
                    abs_path = os.path.abspath(mp4_path)
                    reward = self.inferencer.reward(
                        [abs_path],
                        [pmt],
                        use_norm=self.use_norm,
                    )
                    vq = float(reward[0]["VQ"])
                    mq = float(reward[0]["MQ"])
                    ta = float(reward[0]["TA"])
                except Exception as e:  # noqa: BLE001
                    # DanceGRPO behaviour: on failure assign -1.0 to all dimensions
                    # (train_grpo_hunyuan.py:274-278).
                    logger.warning(
                        f"[VideoAlign] reward computation failed for sample {i}: {e}; "
                        f"falling back to {self.fallback_value}."
                    )
                    vq = mq = ta = self.fallback_value

                vq_list.append(vq)
                mq_list.append(mq)
                ta_list.append(ta)
                composite_list.append(
                    self.vq_coef * vq + self.mq_coef * mq + self.ta_coef * ta
                )

        rewards = torch.tensor(composite_list, dtype=torch.float32, device=device)

        extra_info = None
        if self.return_metrics:
            extra_info = {
                "VQ": torch.tensor(vq_list, dtype=torch.float32, device=device),
                "MQ": torch.tensor(mq_list, dtype=torch.float32, device=device),
                "TA": torch.tensor(ta_list, dtype=torch.float32, device=device),
                "composite_coefs": {
                    "vq": self.vq_coef,
                    "mq": self.mq_coef,
                    "ta": self.ta_coef,
                },
                "use_norm": self.use_norm,
            }

        return RewardModelOutput(rewards=rewards, extra_info=extra_info)
