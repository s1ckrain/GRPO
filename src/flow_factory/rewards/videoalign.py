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

# src/flow_factory/rewards/videoalign.py
"""
VideoAlign (KwaiVGI/VideoReward) reward model for Flow-Factory.

Self-contained inference implementation. Previously this file imported
`VideoAlign/inference.py`, which transitively pulled in `train_reward.py` and
`trainer.py`; the latter relies on `DistributedTensorGatherer` (removed in
transformers >= 4.42). The Flow-Factory environment pins transformers >= 4.57.1,
so that path is unusable.

What we inline here (copied 1:1 from the VideoAlign repo so the LoRA weights
deserialize against the exact same module tree they were saved from):

    * `Qwen2VLRewardModelBT`        (VideoAlign/trainer.py:59-173)
    * `find_target_linear_names`    (VideoAlign/train_reward.py:43-63)
    * checkpoint loader             (VideoAlign/utils.py:136-200)
    * prompt templates              (VideoAlign/prompt_template.py)
    * minimal inference orchestrator (VideoAlign/inference.py:29-203)

What we still import from the VideoAlign repo on disk:

    * `vision_process.process_vision_info`
      (only depends on torch / torchvision / decord / PIL — no transformers
      contamination; safe across versions).

Reward call matches DanceGRPO exactly
(`DanceGRPO/fastvideo/train_grpo_hunyuan.py:260-278`):

    reward = inferencer.reward([abs_video_path], [prompt], use_norm=True)
    composite = vq_coef * VQ + mq_coef * MQ + ta_coef * TA
    # Defaults (1, 1, 1) match SAGE-GRPO codebase default + VideoAlign `Overall`.
    # On any failure -> sentinel -1.0 (DanceGRPO's fallback).

YAML usage:
    rewards:
      - name: "videoalign"
        reward_model: "videoalign"
        batch_size: 1
        videoalign_dir: "/aigc/posttrain/siyuanfu/VideoAlign"
        checkpoint_dir: null              # default = {videoalign_dir}/checkpoints
        vq_coef: 1.0
        mq_coef: 1.0
        ta_coef: 1.0
        use_norm: true                    # DanceGRPO uses True
        video_fps: 15                     # match Wan2.1 generation fps
        return_metrics: true              # expose VQ/MQ/TA in extra_info
"""
from __future__ import annotations

import glob
import json
import logging
import os
import sys
import tempfile
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple

import safetensors.torch
import torch
import torch.nn as nn
from accelerate import Accelerator
from PIL import Image

from .abc import PointwiseRewardModel, RewardModelOutput
from ..hparams import RewardArguments

logger = logging.getLogger(__name__)


# ============================================================================
# Section 1: prompt template (verbatim from VideoAlign/prompt_template.py)
# ============================================================================

_DIMENSION_DESCRIPTIONS = {
    "VQ": ["visual quality", "the quality of the video in terms of clearness, resolution, brightness, and color"],
    "TA": ["text-to-video alignment", "the alignment between the text prompt and the video content and motion"],
    "MQ": ["motion quality", "the quality of the motion in terms of consistency, smoothness, and completeness"],
    "Overall": ["Overall Performance", "the overall performance of the video in terms of visual quality, text-to-video alignment, and motion quality"],
}

_DETAILED_PROMPT_WITH_SPECIAL_TOKEN = """
You are tasked with evaluating a generated video based on three distinct criteria: Visual Quality, Motion Quality, and Text Alignment. Please provide a rating from 0 to 10 for each of the three categories, with 0 being the worst and 10 being the best. Each evaluation should be independent of the others.

**Visual Quality:**  
Evaluate the overall visual quality of the video, with a focus on static factors. The following sub-dimensions should be considered:
- **Reasonableness:** The video should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the video. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the video, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The video should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible. 

Please provide the ratings of Visual Quality: <|VQ_reward|>
END

**Motion Quality:**  
Assess the dynamic aspects of the video, with a focus on dynamic factors. Consider the following sub-dimensions:
- **Stability:** Evaluate the continuity and stability between frames. There should be no sudden, unnatural jumps, and the video should maintain stable attributes (e.g., no fluctuating colors, textures, or missing body parts).
- **Naturalness:** The movement should align with physical laws and be realistic. For example, clothing should flow naturally with motion, and facial expressions should change appropriately (e.g., blinking, mouth movements).
- **Aesthetic Quality:** The movement should be smooth and fluid. The transitions between different motions or camera angles should be seamless, and the overall dynamic feel should be visually pleasing.
- **Fusion:** Ensure that elements in motion (e.g., edges of the subject, hair, clothing) blend naturally with the background, without obvious artifacts or the feeling of cut-and-paste effects.
- **Clarity of Motion:** The video should be clear and smooth in motion. Pay attention to any areas where the video might have blurry or unsteady sections that hinder visual continuity.
- **Amplitude:** If the video is largely static or has little movement, assign a low score for motion quality.

Please provide the ratings of Motion Quality: <|MQ_reward|>
END

**Text Alignment:**  
Assess how well the video matches the textual prompt across the following sub-dimensions:
- **Subject Relevance** Evaluate how accurately the subject(s) in the video (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
- **Motion Relevance:** Evaluate if the dynamic actions (e.g., gestures, posture, facial expressions like talking or blinking) align with the described prompt. The motion should match the prompt in terms of type, scale, and direction.
- **Environment Relevance:** Assess whether the background and scene fit the prompt. This includes checking if real-world locations or scenes are accurately represented, though some stylistic adaptation is acceptable.  
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the video adheres to this style.
- **Camera Movement Relevance:** Check if the camera movements (e.g., following the subject, focus shifts) are consistent with the expected behavior from the prompt.

Textual prompt - {text_prompt}
Please provide the ratings of Text Alignment: <|TA_reward|>
END
"""


def _build_prompt(prompt: str, dimension, template_type: str) -> str:
    """Mirror of VideoAlign/prompt_template.py:build_prompt for the only path we use."""
    if template_type != "detailed_special":
        raise NotImplementedError(
            f"Inlined VideoAlign reward only supports template_type='detailed_special' "
            f"(the one used by the released checkpoint); got '{template_type}'."
        )
    return _DETAILED_PROMPT_WITH_SPECIAL_TOKEN.format(text_prompt=prompt)


# ============================================================================
# Section 2: Qwen2-VL reward model with rm_head
# (verbatim from VideoAlign/trainer.py:59-173, no behaviour change)
# ============================================================================


def _load_qwen2vl_class():
    """
    Import Qwen2VLForConditionalGeneration. The class lives in different
    sub-modules across transformers versions; AutoClass would also work but the
    direct import keeps the inheritance explicit and lets `from_pretrained`
    return *our* subclass without auto_map shenanigans.
    """
    from transformers import Qwen2VLForConditionalGeneration  # noqa: WPS433
    return Qwen2VLForConditionalGeneration


def _make_qwen2vl_reward_model_class():
    """
    Build `Qwen2VLRewardModelBT` lazily so that we don't hard-fail at module
    import time when transformers happens to be missing Qwen2-VL support.
    """
    Qwen2VLForConditionalGeneration = _load_qwen2vl_class()

    class Qwen2VLRewardModelBT(Qwen2VLForConditionalGeneration):
        """Copied 1:1 from VideoAlign/trainer.py:59-173 so saved LoRA targets line up."""

        def __init__(self, config, output_dim=4, reward_token="last", special_token_ids=None):
            super().__init__(config)
            self.output_dim = output_dim
            self.rm_head = nn.Linear(config.hidden_size, output_dim, bias=False)
            self.reward_token = reward_token

            self.special_token_ids = special_token_ids
            if self.special_token_ids is not None:
                self.reward_token = "special"

        def forward(  # noqa: PLR0912, PLR0915
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            rope_deltas: Optional[torch.LongTensor] = None,
        ):
            output_attentions = (
                output_attentions if output_attentions is not None else self.config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if inputs_embeds is None:
                inputs_embeds = self.model.embed_tokens(input_ids)
                if pixel_values is not None:
                    pixel_values = pixel_values.type(self.visual.get_dtype())
                    image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                    image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

                if pixel_values_videos is not None:
                    pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                    video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                    video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                    video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

                if attention_mask is not None:
                    attention_mask = attention_mask.to(inputs_embeds.device)

            outputs = self.model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]  # [B, L, D]
            logits = self.rm_head(hidden_states)  # [B, L, N]

            batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

            if self.config.pad_token_id is None and batch_size != 1:
                raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:
                    sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                    sequence_lengths = sequence_lengths % input_ids.shape[-1]
                    sequence_lengths = sequence_lengths.to(logits.device)
                else:
                    sequence_lengths = -1

            if self.reward_token == "last":
                pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
            elif self.reward_token == "mean":
                valid_lengths = torch.clamp(sequence_lengths, min=0, max=logits.size(1) - 1)
                pooled_logits = torch.stack(
                    [logits[i, : valid_lengths[i]].mean(dim=0) for i in range(batch_size)]
                )
            elif self.reward_token == "special":
                special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                for special_token_id in self.special_token_ids:
                    special_token_mask = special_token_mask | (input_ids == special_token_id)
                pooled_logits = logits[special_token_mask, ...]
                pooled_logits = pooled_logits.view(batch_size, 3, -1)  # [B, 3, N], 3 attrs
                if self.output_dim == 3:
                    pooled_logits = pooled_logits.diagonal(dim1=1, dim2=2)
                pooled_logits = pooled_logits.view(batch_size, -1)
            else:
                raise ValueError("Invalid reward_token")

            return {"logits": pooled_logits}

    return Qwen2VLRewardModelBT


# ============================================================================
# Section 3: checkpoint loader (verbatim from VideoAlign/utils.py:136-200)
# ============================================================================


def _insert_adapter_name_into_state_dict(
    state_dict: Dict[str, torch.Tensor], adapter_name: str, parameter_prefix: str
) -> Dict[str, torch.Tensor]:
    peft_model_state_dict = {}
    for key, val in state_dict.items():
        if parameter_prefix in key:
            suffix = key.split(parameter_prefix)[1]
            if "." in suffix:
                suffix_to_replace = ".".join(suffix.split(".")[1:])
                key = key.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
            else:
                key = f"{key}.{adapter_name}"
            peft_model_state_dict[key] = val
        else:
            peft_model_state_dict[key] = val
    return peft_model_state_dict


def _load_videoalign_checkpoint(
    model: nn.Module, checkpoint_dir: str, checkpoint_step: Optional[int]
) -> Tuple[nn.Module, str]:
    """Mirror of VideoAlign/utils.py:load_model_from_checkpoint."""
    checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No 'checkpoint-*' subdir found under {checkpoint_dir}. "
            "Expected layout: {checkpoint_dir}/checkpoint-<step>/{adapter_model.safetensors, non_lora_state_dict.pth}"
        )
    checkpoint_paths.sort(key=lambda x: int(x.split("-")[-1]), reverse=True)

    if checkpoint_step is None or checkpoint_step == -1:
        checkpoint_path = checkpoint_paths[0]
    else:
        candidate = os.path.join(checkpoint_dir, f"checkpoint-{checkpoint_step}")
        checkpoint_path = candidate if candidate in checkpoint_paths else checkpoint_paths[0]

    resolved_step = checkpoint_path.split("checkpoint-")[-1].split("/")[0]
    logger.info(f"[VideoAlign] using checkpoint: {checkpoint_path}")

    full_ckpt = os.path.join(checkpoint_path, "model.pth")
    lora_ckpt = os.path.join(checkpoint_path, "adapter_model.safetensors")
    non_lora_ckpt = os.path.join(checkpoint_path, "non_lora_state_dict.pth")

    if os.path.exists(full_ckpt):
        model_state_dict = torch.load(full_ckpt, map_location="cpu")
        model.load_state_dict(model_state_dict)
    else:
        lora_state_dict = safetensors.torch.load_file(lora_ckpt)
        non_lora_state_dict = torch.load(non_lora_ckpt, map_location="cpu")
        lora_state_dict = _insert_adapter_name_into_state_dict(
            lora_state_dict, adapter_name="default", parameter_prefix="lora_"
        )
        model_state_dict = model.state_dict()
        model_state_dict.update(non_lora_state_dict)
        model_state_dict.update(lora_state_dict)
        model.load_state_dict(model_state_dict)

    return model, resolved_step


# ============================================================================
# Section 4: LoRA target discovery (verbatim from VideoAlign/train_reward.py:43-63)
# ============================================================================


def _find_target_linear_names(
    model: nn.Module,
    num_lora_modules: int = -1,
    lora_namespan_exclude: Optional[List[str]] = None,
) -> List[str]:
    lora_namespan_exclude = lora_namespan_exclude or []
    lora_module_names: List[str] = []
    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (nn.Linear, nn.Embedding)):
            lora_module_names.append(name)
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    return lora_module_names


# ============================================================================
# Section 5: inferencer (replaces VideoAlign/inference.py:VideoVLMRewardInference)
# ============================================================================


class _VideoAlignInferencer:
    """
    Drop-in replacement for `VideoVLMRewardInference` that builds the model
    in-process (no dependency on `VideoAlign/train_reward.py` or `trainer.py`)
    and exposes the same `reward(video_paths, prompts, ..., use_norm=True)` API
    DanceGRPO calls.
    """

    def __init__(
        self,
        load_from_pretrained: str,
        vision_process_module,
        load_from_pretrained_step: Optional[int] = -1,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.dtype = dtype
        self._process_vision_info = vision_process_module.process_vision_info

        # ---- read model_config.json (saved by VideoAlign trainer) ----------
        config_path = os.path.join(load_from_pretrained, "model_config.json")
        with open(config_path, "r") as f:
            cfg = json.load(f)

        data_config = cfg["data_config"]
        model_config = cfg["model_config"]
        peft_lora_config = cfg["peft_lora_config"]
        self._inference_config = cfg.get("inference_config")  # {VQ_mean, VQ_std, ...}

        # Cache the inference-time data knobs (fps / num_frames / max_pixels / template)
        self._fps = data_config["fps"]
        self._num_frames = data_config["num_frames"]
        self._max_pixels = data_config["max_frame_pixels"]
        self._sample_type = data_config["sample_type"]
        self._eval_dim = data_config["eval_dim"]
        self._prompt_template_type = data_config["prompt_template_type"]

        # ---- processor + special tokens (mirror create_model_and_processor) -
        from transformers import AutoProcessor  # noqa: WPS433
        from peft import LoraConfig, get_peft_model  # noqa: WPS433

        processor = AutoProcessor.from_pretrained(
            model_config["model_name_or_path"],
            padding_side="right",
        )
        special_token_ids = None
        if model_config.get("use_special_tokens", False):
            special_tokens = ["<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>"]
            processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)

        # ---- build base reward model ---------------------------------------
        # Match VideoAlign/inference.py exactly:
        #   - attn_implementation: "flash_attention_2" if available else "sdpa"
        #     (VideoAlign uses flash_attention_2 unless `disable_flash_attn2=True`,
        #      which is False in VideoVLMRewardInference.__init__).
        #   - use_cache=False (VideoAlign sets `gradient_checkpointing=False` so
        #     `use_cache = True if False else False` = False; we just inline that).
        Qwen2VLRewardModelBT = _make_qwen2vl_reward_model_class()
        try:
            import flash_attn  # type: ignore  # noqa: F401
            attn_impl = "flash_attention_2"
        except Exception:  # noqa: BLE001
            attn_impl = "sdpa"

        model = Qwen2VLRewardModelBT.from_pretrained(
            model_config["model_name_or_path"],
            output_dim=model_config["output_dim"],
            reward_token=model_config["reward_token"],
            special_token_ids=special_token_ids,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
        )
        # `use_cache` must go through config, NOT as a from_pretrained kwarg:
        # our subclass `__init__(self, config, output_dim=..., reward_token=...,
        # special_token_ids=...)` doesn't declare `use_cache`, and some
        # transformers versions forward unknown kwargs to __init__ → TypeError.
        # Matches VideoAlign/inference.py behavior (gradient_checkpointing=False
        # → use_cache=False) without going through the kwarg path.
        model.config.use_cache = False
        if model_config.get("use_special_tokens", False):
            model.resize_token_embeddings(len(processor.tokenizer))
        model.to(dtype)

        # ---- (optionally) wrap with LoRA so checkpoint keys match ---------
        if peft_lora_config.get("lora_enable", False):
            namespan_exclude = peft_lora_config.get("lora_namespan_exclude")
            if isinstance(namespan_exclude, str):
                # the trainer ast.literal_eval's strings; keep parity
                import ast as _ast
                namespan_exclude = _ast.literal_eval(namespan_exclude)
            namespan_exclude = list(namespan_exclude or [])
            if not peft_lora_config.get("vision_lora", False) and "visual" not in namespan_exclude:
                namespan_exclude.append("visual")
            target_modules = _find_target_linear_names(
                model,
                num_lora_modules=peft_lora_config.get("num_lora_modules", -1),
                lora_namespan_exclude=namespan_exclude,
            )
            peft_config = LoraConfig(
                target_modules=target_modules,
                r=peft_lora_config["lora_r"],
                lora_alpha=peft_lora_config["lora_alpha"],
                lora_dropout=peft_lora_config["lora_dropout"],
                task_type=peft_lora_config["lora_task_type"],
                use_rslora=peft_lora_config.get("use_rslora", False),
                bias="none",
                modules_to_save=peft_lora_config.get("lora_modules_to_save"),
            )
            model = get_peft_model(model, peft_config)

        model.config.tokenizer_padding_side = processor.tokenizer.padding_side
        model.config.pad_token_id = processor.tokenizer.pad_token_id

        # ---- load trained weights -----------------------------------------
        model, _ = _load_videoalign_checkpoint(model, load_from_pretrained, load_from_pretrained_step)
        model.eval()
        model.to(self.device)

        self.model = model
        self.processor = processor

    # ---- helpers (mirror inference.py:_norm / _prepare_input) -------------

    def _norm(self, reward: Dict[str, float]) -> Dict[str, float]:
        if self._inference_config is None:
            return reward
        reward["VQ"] = (reward["VQ"] - self._inference_config["VQ_mean"]) / self._inference_config["VQ_std"]
        reward["MQ"] = (reward["MQ"] - self._inference_config["MQ_mean"]) / self._inference_config["MQ_std"]
        reward["TA"] = (reward["TA"] - self._inference_config["TA_mean"]) / self._inference_config["TA_std"]
        return reward

    def _prepare_input(self, data):
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        if isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        if isinstance(data, torch.Tensor):
            return data.to(device=self.device)
        return data

    def _build_chat(self, video_paths, prompts, fps, num_frames, max_pixels):
        if num_frames is None:
            return [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": f"file://{video_path}",
                                "max_pixels": max_pixels,
                                "fps": fps,
                                "sample_type": self._sample_type,
                            },
                            {
                                "type": "text",
                                "text": _build_prompt(prompt, self._eval_dim, self._prompt_template_type),
                            },
                        ],
                    },
                ]
                for video_path, prompt in zip(video_paths, prompts)
            ]
        return [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": f"file://{video_path}",
                            "max_pixels": max_pixels,
                            "nframes": num_frames,
                            "sample_type": self._sample_type,
                        },
                        {
                            "type": "text",
                            "text": _build_prompt(prompt, self._eval_dim, self._prompt_template_type),
                        },
                    ],
                },
            ]
            for video_path, prompt in zip(video_paths, prompts)
        ]

    def reward(
        self,
        video_paths: List[str],
        prompts: List[str],
        fps: Optional[float] = None,
        num_frames: Optional[int] = None,
        max_pixels: Optional[int] = None,
        use_norm: bool = True,
    ) -> List[Dict[str, float]]:
        assert fps is None or num_frames is None, "fps and num_frames cannot be set at the same time."

        fps = self._fps if fps is None else fps
        num_frames = self._num_frames if num_frames is None else num_frames
        max_pixels = self._max_pixels if max_pixels is None else max_pixels

        chat_data = self._build_chat(video_paths, prompts, fps, num_frames, max_pixels)
        image_inputs, video_inputs = self._process_vision_info(chat_data)

        batch = self.processor(
            text=self.processor.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        batch = self._prepare_input(batch)

        out = self.model(return_dict=True, **batch)["logits"]

        rewards: List[Dict[str, float]] = [
            {"VQ": float(row[0].item()), "MQ": float(row[1].item()), "TA": float(row[2].item())}
            for row in out
        ]
        for r in rewards:
            if use_norm:
                self._norm(r)
            r["Overall"] = r["VQ"] + r["MQ"] + r["TA"]
        return rewards


# ============================================================================
# Section 6: helpers used by Flow-Factory adapter
# ============================================================================


def _import_vision_process(videoalign_dir: str):
    """
    Import *only* `vision_process` from the VideoAlign repo. We deliberately do
    NOT import `inference` / `train_reward` / `trainer` which would pull in
    `transformers.trainer.DistributedTensorGatherer` (removed in transformers
    >= 4.42).
    """
    videoalign_dir = os.path.abspath(videoalign_dir)
    if not os.path.isdir(videoalign_dir):
        raise FileNotFoundError(
            f"videoalign_dir does not exist: {videoalign_dir}. "
            "Set it via reward extra_kwargs (e.g. videoalign_dir: /path/to/VideoAlign)."
        )
    vp_path = os.path.join(videoalign_dir, "vision_process.py")
    if not os.path.isfile(vp_path):
        raise FileNotFoundError(
            f"vision_process.py not found under {videoalign_dir}. "
            "Pass the VideoAlign repo root (the one with vision_process.py + checkpoints/)."
        )
    if videoalign_dir not in sys.path:
        sys.path.insert(0, videoalign_dir)
    import vision_process as vision_process_module  # type: ignore  # noqa: WPS433
    return vision_process_module


def _frames_to_mp4(frames: List[Image.Image], out_path: str, fps: int) -> None:
    """Write PIL frames to mp4 (same approach DanceGRPO uses)."""
    try:
        from diffusers.utils import export_to_video  # type: ignore

        export_to_video(frames, out_path, fps=fps)
        return
    except Exception:  # noqa: BLE001
        pass
    import imageio  # type: ignore
    import numpy as np

    arr = [np.array(f.convert("RGB")) for f in frames]
    imageio.mimsave(out_path, arr, fps=fps, codec="libx264", quality=8)


# ============================================================================
# Section 7: Flow-Factory adapter
# ============================================================================


class VideoAlignRewardModel(PointwiseRewardModel):
    """
    Pointwise reward model wrapping KwaiVGI/VideoReward (VideoAlign).

    Composite reward = vq_coef * VQ + mq_coef * MQ + ta_coef * TA
    (defaults 1:1:1 = SAGE-GRPO codebase default = VideoAlign's built-in `Overall`).
    """

    required_fields = ("prompt", "video")
    use_tensor_inputs = False  # We need PIL frames to dump them to mp4.

    def __init__(self, config: RewardArguments, accelerator: Accelerator):
        super().__init__(config, accelerator)

        # ---- read extra kwargs --------------------------------------------
        videoalign_dir: str = getattr(config, "videoalign_dir", None) or os.environ.get(
            "VIDEOALIGN_DIR", "/aigc/posttrain/siyuanfu/VideoAlign"
        )
        checkpoint_dir: Optional[str] = getattr(config, "checkpoint_dir", None)
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(videoalign_dir, "checkpoints")

        self.vq_coef: float = float(getattr(config, "vq_coef", 1.0))
        self.mq_coef: float = float(getattr(config, "mq_coef", 1.0))
        self.ta_coef: float = float(getattr(config, "ta_coef", 1.0))

        self.use_norm: bool = bool(getattr(config, "use_norm", True))   # DanceGRPO: True
        self.video_fps: int = int(getattr(config, "video_fps", 15))     # match Wan2.1
        self.return_metrics: bool = bool(getattr(config, "return_metrics", True))
        self.fallback_value: float = float(getattr(config, "fallback_value", -1.0))

        # ---- load self-contained inferencer -------------------------------
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(
                f"VideoAlign checkpoint_dir not found: {checkpoint_dir}. "
                f"Expected layout: {videoalign_dir}/checkpoints/{{model_config.json, checkpoint-*/}}"
            )
        vision_process_module = _import_vision_process(videoalign_dir)
        self.inferencer = _VideoAlignInferencer(
            load_from_pretrained=checkpoint_dir,
            vision_process_module=vision_process_module,
            device=str(self.device),
            dtype=self.dtype,
        )
        logger.info(
            f"[VideoAlign] loaded ckpt={checkpoint_dir} device={self.device} dtype={self.dtype} "
            f"composite (VQ,MQ,TA)=({self.vq_coef},{self.mq_coef},{self.ta_coef}) "
            f"use_norm={self.use_norm} video_fps={self.video_fps}"
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

        with tempfile.TemporaryDirectory(prefix="videoalign_") as tmpdir:
            for i, (frames, pmt) in enumerate(zip(video, prompt)):
                mp4_path = os.path.join(tmpdir, f"sample_{i:04d}.mp4")
                try:
                    _frames_to_mp4(frames, mp4_path, fps=self.video_fps)
                    abs_path = os.path.abspath(mp4_path)
                    rew = self.inferencer.reward([abs_path], [pmt], use_norm=self.use_norm)
                    vq = float(rew[0]["VQ"])
                    mq = float(rew[0]["MQ"])
                    ta = float(rew[0]["TA"])
                except Exception as e:  # noqa: BLE001
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
