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
import re
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


def _cfg_get(config, key: str):
    """Read a Qwen2-VL config attribute that may live either at the top level
    (transformers < ~4.50, which VideoAlign was authored against) or under
    `config.text_config.*` (transformers >= ~4.50, after PR #37268 standardized
    Qwen2-VL configs as nested {text_config, vision_config}).

    See https://github.com/huggingface/transformers/issues/38331.
    """
    val = getattr(config, key, None)
    if val is not None:
        return val
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None:
        val = getattr(text_cfg, key, None)
        if val is not None:
            return val
    # Final fallback: return None (mirrors original `config.<key>` for unset
    # optional fields like `pad_token_id`).
    return None


# ----------------------------------------------------------------------------
# Legacy -> new state_dict key translation (transformers Qwen2-VL restructure).
#
# transformers >= ~4.50 restructured Qwen2-VL: visual moved from top-level
# `self.visual` to `self.model.visual`, and the text submodel moved from
# `self.model` (a Qwen2Model) to `self.model.language_model` (a Qwen2VLTextModel).
# `from_pretrained()` auto-translates HF-hub legacy ckpts using this regex:
#
#   _checkpoint_conversion_mapping = {
#       "^visual": "model.visual",
#       r"^model(?!\.(language_model|visual))": "model.language_model",
#   }
#
# (https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/
#  models/qwen2_vl/modeling_qwen2_vl.py#L1222)
#
# But we DON'T go through `from_pretrained` for the VideoAlign LoRA+non_lora
# files (we load them with torch.load / safetensors.load and merge manually),
# so we have to apply the same mapping ourselves. We also handle the PEFT
# `base_model.model.<key>` prefix.
# ----------------------------------------------------------------------------


_QWEN2VL_LEGACY_TO_NEW_REGEX = [
    (re.compile(r"^visual"), "model.visual"),
    (re.compile(r"^model(?!\.(language_model|visual))"), "model.language_model"),
]

_PEFT_PREFIX = "base_model.model."


def _has_new_qwen2vl_layout(model: nn.Module) -> bool:
    """True iff we're on transformers >= ~4.50 where visual+text are nested
    under `model.model.{visual, language_model}` instead of being at the top
    level. Determines whether legacy state_dicts need translation and whether
    forward() should route through the new nested submodules.
    """
    inner = getattr(model, "model", None)
    return inner is not None and hasattr(inner, "visual") and hasattr(inner, "language_model")


def _translate_legacy_qwen2vl_keys(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Apply Qwen2VL's checkpoint_conversion_mapping manually so we can load
    VideoAlign-saved (legacy-flat) state_dicts into the new nested model.

    Handles both bare keys (`visual.patch_embed.proj.weight`) and PEFT-prefixed
    keys (`base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight`).
    Keys that don't match any pattern (e.g. `lm_head.weight`, `rm_head.weight`)
    are left untouched, matching the upstream regex semantics.
    """
    new_state_dict: Dict[str, torch.Tensor] = {}
    for key, val in state_dict.items():
        if key.startswith(_PEFT_PREFIX):
            prefix = _PEFT_PREFIX
            tail = key[len(_PEFT_PREFIX):]
        else:
            prefix = ""
            tail = key
        for pattern, repl in _QWEN2VL_LEGACY_TO_NEW_REGEX:
            new_tail, count = pattern.subn(repl, tail, count=1)
            if count:
                tail = new_tail
                break
        new_state_dict[prefix + tail] = val
    return new_state_dict


def _get_visual(model: nn.Module) -> nn.Module:
    """Resolve the visual submodule across transformers versions."""
    if hasattr(model, "visual"):
        return model.visual
    return model.model.visual


def _get_text_submodel(model: nn.Module) -> nn.Module:
    """Resolve the text submodel across transformers versions.

    Old layout: `self.model` IS the Qwen2Model (has `embed_tokens`).
    New layout: `self.model` is Qwen2VLModel; text is at `self.model.language_model`.
    """
    if hasattr(model.model, "embed_tokens"):
        return model.model
    return model.model.language_model


def _visual_dtype(visual: nn.Module) -> torch.dtype:
    """Resolve visual dtype across transformers versions (old: get_dtype(), new: .dtype)."""
    if hasattr(visual, "get_dtype"):
        return visual.get_dtype()
    if hasattr(visual, "dtype"):
        return visual.dtype
    return next(visual.parameters()).dtype


def _unwrap_visual_output(out) -> torch.Tensor:
    """Normalize the Qwen2-VL visual forward output to a plain tensor.

    Old transformers (VideoAlign's authoring env): returns torch.Tensor directly.
    Some newer transformers wrap it in `BaseModelOutputWithPooling` (an
    `ModelOutput` dataclass with `last_hidden_state` / `pooler_output`), or in
    a plain tuple. We always want the flattened patch embeddings tensor.
    """
    if torch.is_tensor(out):
        return out
    # ModelOutput / BaseModelOutputWithPooling case
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    # Some variants name it differently
    for key in ("hidden_states", "image_embeds", "video_embeds"):
        val = getattr(out, key, None)
        if torch.is_tensor(val):
            return val
    # Plain tuple (e.g. return_dict=False path)
    if isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
        return out[0]
    raise TypeError(
        f"Unexpected Qwen2-VL visual forward output type: {type(out).__name__}. "
        "Cannot extract embedding tensor."
    )


def _make_qwen2vl_reward_model_class():
    """
    Build `Qwen2VLRewardModelBT` lazily so that we don't hard-fail at module
    import time when transformers happens to be missing Qwen2-VL support.
    """
    Qwen2VLForConditionalGeneration = _load_qwen2vl_class()

    class Qwen2VLRewardModelBT(Qwen2VLForConditionalGeneration):
        """Copied 1:1 from VideoAlign/trainer.py:59-173 so saved LoRA targets line up.

        The only deviation from the verbatim trainer.py is that every
        `config.<key>` access goes through `_cfg_get` so we tolerate both the
        old flat Qwen2VLConfig (VideoAlign's authoring env) and the new nested
        one (transformers >= ~4.50).
        """

        def __init__(self, config, output_dim=4, reward_token="last", special_token_ids=None):
            super().__init__(config)
            self.output_dim = output_dim
            self.rm_head = nn.Linear(_cfg_get(config, "hidden_size"), output_dim, bias=False)
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
            **kwargs,  # Absorb extra batch fields that newer transformers' AutoProcessor
            # adds for multimodal inputs (e.g. `mm_token_type_ids`, `cache_position`).
            # VideoAlign's reward path doesn't need them — visual/text embeds are
            # already fused inside this forward via input_ids/grid_thw — so it's
            # safe to silently drop them. Matches behavior of VideoAlign's original
            # signature on older transformers where these fields weren't produced.
        ):
            output_attentions = (
                output_attentions if output_attentions is not None else self.config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # Resolve visual + text submodules across transformers versions:
            #   old: self.visual, self.model (Qwen2Model)
            #   new: self.model.visual, self.model.language_model (Qwen2VLTextModel)
            visual = _get_visual(self)
            text_model = _get_text_submodel(self)

            if inputs_embeds is None:
                inputs_embeds = text_model.embed_tokens(input_ids)
                if pixel_values is not None:
                    pixel_values = pixel_values.type(_visual_dtype(visual))
                    image_embeds = _unwrap_visual_output(
                        visual(pixel_values, grid_thw=image_grid_thw)
                    )
                    image_token_id = _cfg_get(self.config, "image_token_id")
                    image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

                if pixel_values_videos is not None:
                    pixel_values_videos = pixel_values_videos.type(_visual_dtype(visual))
                    video_embeds = _unwrap_visual_output(
                        visual(pixel_values_videos, grid_thw=video_grid_thw)
                    )
                    video_token_id = _cfg_get(self.config, "video_token_id")
                    video_mask = (input_ids == video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                    video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

                if attention_mask is not None:
                    attention_mask = attention_mask.to(inputs_embeds.device)

            outputs = text_model(
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

            pad_token_id = _cfg_get(self.config, "pad_token_id")
            if pad_token_id is None and batch_size != 1:
                raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
            if pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:
                    sequence_lengths = torch.eq(input_ids, pad_token_id).int().argmax(-1) - 1
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

    # Translate legacy-flat keys to the new nested layout iff we're on the
    # restructured transformers (>= ~4.50). On the old transformers (VideoAlign's
    # authoring env), `_translate_legacy_qwen2vl_keys` is a no-op because the
    # underlying PeftModel-wrapped model already uses the legacy key shape.
    needs_translation = _has_new_qwen2vl_layout(
        model.base_model.model if hasattr(model, "base_model") else model
    )

    if os.path.exists(full_ckpt):
        model_state_dict = torch.load(full_ckpt, map_location="cpu")
        if needs_translation:
            model_state_dict = _translate_legacy_qwen2vl_keys(model_state_dict)
        model.load_state_dict(model_state_dict)
    else:
        lora_state_dict = safetensors.torch.load_file(lora_ckpt)
        non_lora_state_dict = torch.load(non_lora_ckpt, map_location="cpu")
        lora_state_dict = _insert_adapter_name_into_state_dict(
            lora_state_dict, adapter_name="default", parameter_prefix="lora_"
        )
        if needs_translation:
            lora_state_dict = _translate_legacy_qwen2vl_keys(lora_state_dict)
            non_lora_state_dict = _translate_legacy_qwen2vl_keys(non_lora_state_dict)
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
        # Training signal lists (post-`use_norm` selection; what goes into RL):
        vq_list: List[float] = []
        mq_list: List[float] = []
        ta_list: List[float] = []
        composite_list: List[float] = []
        # Raw + z-scored lists, kept independently for wandb logging so both
        # views are visible regardless of `self.use_norm`.
        vq_raw_list: List[float] = []
        mq_raw_list: List[float] = []
        ta_raw_list: List[float] = []
        vq_norm_list: List[float] = []
        mq_norm_list: List[float] = []
        ta_norm_list: List[float] = []

        # Pull VideoAlign's z-score constants once (None ckpts skip norm).
        infer_cfg = getattr(self.inferencer, "_inference_config", None)

        with tempfile.TemporaryDirectory(prefix="videoalign_") as tmpdir:
            for i, (frames, pmt) in enumerate(zip(video, prompt)):
                mp4_path = os.path.join(tmpdir, f"sample_{i:04d}.mp4")
                try:
                    _frames_to_mp4(frames, mp4_path, fps=self.video_fps)
                    abs_path = os.path.abspath(mp4_path)
                    # Always request RAW from the inferencer; we apply z-score
                    # ourselves so we can log both scales at once.
                    rew = self.inferencer.reward([abs_path], [pmt], use_norm=False)
                    vq_raw = float(rew[0]["VQ"])
                    mq_raw = float(rew[0]["MQ"])
                    ta_raw = float(rew[0]["TA"])
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        f"[VideoAlign] reward computation failed for sample {i}: {e}; "
                        f"falling back to {self.fallback_value}."
                    )
                    vq_raw = mq_raw = ta_raw = self.fallback_value

                # Apply VideoAlign's fixed z-score (same constants inference.py
                # uses); falls back to raw if config is missing.
                if infer_cfg is not None:
                    vq_norm = (vq_raw - infer_cfg["VQ_mean"]) / infer_cfg["VQ_std"]
                    mq_norm = (mq_raw - infer_cfg["MQ_mean"]) / infer_cfg["MQ_std"]
                    ta_norm = (ta_raw - infer_cfg["TA_mean"]) / infer_cfg["TA_std"]
                else:
                    vq_norm, mq_norm, ta_norm = vq_raw, mq_raw, ta_raw

                # Pick the variant used as the RL training signal.
                if self.use_norm:
                    vq, mq, ta = vq_norm, mq_norm, ta_norm
                else:
                    vq, mq, ta = vq_raw, mq_raw, ta_raw

                vq_list.append(vq)
                mq_list.append(mq)
                ta_list.append(ta)
                composite_list.append(
                    self.vq_coef * vq + self.mq_coef * mq + self.ta_coef * ta
                )
                vq_raw_list.append(vq_raw)
                mq_raw_list.append(mq_raw)
                ta_raw_list.append(ta_raw)
                vq_norm_list.append(vq_norm)
                mq_norm_list.append(mq_norm)
                ta_norm_list.append(ta_norm)

        rewards = torch.tensor(composite_list, dtype=torch.float32, device=device)

        # Flow-Factory does not consume RewardModelOutput.extra_info, and it
        # also does not register trackers via accelerator.init_trackers — it
        # uses a stand-alone wandb.init() (see flow_factory/logger/wandb.py).
        # We push per-dim VQ/MQ/TA stats to the active wandb run using a
        # dedicated step axis (`reward/_call`) registered via
        # wandb.define_metric, so this never collides with the trainer's
        # explicit step on `train/*` and `eval/*` panels.
        self._maybe_log_dim_metrics(
            vq_list=vq_list,
            mq_list=mq_list,
            ta_list=ta_list,
            composite_list=composite_list,
            vq_raw_list=vq_raw_list,
            mq_raw_list=mq_raw_list,
            ta_raw_list=ta_raw_list,
            vq_norm_list=vq_norm_list,
            mq_norm_list=mq_norm_list,
            ta_norm_list=ta_norm_list,
        )

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

    def _maybe_log_dim_metrics(
        self,
        vq_list: List[float],
        mq_list: List[float],
        ta_list: List[float],
        composite_list: List[float],
        vq_raw_list: Optional[List[float]] = None,
        mq_raw_list: Optional[List[float]] = None,
        ta_raw_list: Optional[List[float]] = None,
        vq_norm_list: Optional[List[float]] = None,
        mq_norm_list: Optional[List[float]] = None,
        ta_norm_list: Optional[List[float]] = None,
    ) -> None:
        """Push per-call mean stats to the active wandb run.

        Logs three views of the reward (mean only -- see ``_mean`` for why no
        std), all under the ``reward/*`` namespace with an INDEPENDENT step
        axis (``reward/_call``):

        * ``reward/{VQ,MQ,TA}_mean`` -- the TRAINING signal (this is what
          GRPO actually optimizes; equals raw or z-scored depending on
          ``self.use_norm``).
        * ``reward/{VQ,MQ,TA}_raw_mean`` -- always the RAW VideoAlign reward
          (model logits, no z-score). Directly comparable to TaRoS Table 4
          and other published VideoAlign numbers.
        * ``reward/{VQ,MQ,TA}_norm_mean`` -- always the z-SCORED VideoAlign
          reward, using the fixed ``inference_config`` constants from
          ``model_config.json``.

        Per-dim std is intentionally NOT logged here because DanceGRPO's
        ``batch_size=1`` makes every call see a single sample, forcing std
        to be 0 (math, not a bug). Real batch/group-wise reward std is
        already logged by the trainer as ``train/reward_videoalign_std`` /
        ``train/reward_videoalign_group_std_*`` on the composite reward.

        Uses ``wandb.define_metric`` on first call so this never collides
        with the trainer's explicit step on ``train/*`` and ``eval/*`` panels.
        Without it, naive ``wandb.log(payload)`` here would auto-increment
        wandb's internal step counter past the trainer's explicit step and
        cause wandb to silently drop all subsequent ``train/*`` logs.

        Silent no-op when:
          * ``self.accelerator`` is missing,
          * not on the main process,
          * ``wandb`` is not installed,
          * no active wandb run (``wandb.run is None``),
          * any other error is raised (we never want logging to block training).
        """
        accel = getattr(self, "accelerator", None)
        if accel is None:
            return
        if not getattr(accel, "is_main_process", True):
            return
        if not vq_list:
            return
        try:
            import wandb  # local import keeps wandb fully optional
        except ImportError:
            return
        if wandb.run is None:
            return
        try:
            # Lazy one-time registration of the independent step axis so that
            # `reward/*` panels do NOT advance wandb's main `_step` counter
            # (which would block `train/*` logs that use explicit step).
            if not getattr(self, "_reward_metric_registered", False):
                wandb.define_metric("reward/_call")
                wandb.define_metric("reward/*", step_metric="reward/_call")
                self._reward_metric_registered = True
                self._reward_call_counter = 0

            self._reward_call_counter += 1

            def _mean(xs: List[float]) -> float:
                # Only log mean: with DanceGRPO's batch_size=1, every call sees
                # a single sample, so a per-call std is mathematically 0 and
                # carries no information. True batch/group reward std is logged
                # by the trainer as `train/reward_videoalign_std` and
                # `train/reward_videoalign_group_std_*`.
                return float(torch.tensor(xs, dtype=torch.float32).mean().item())

            vq_mean = _mean(vq_list)
            mq_mean = _mean(mq_list)
            ta_mean = _mean(ta_list)
            comp_mean = _mean(composite_list)

            payload = {
                "reward/_call": self._reward_call_counter,
                # Training signal (= raw or norm depending on use_norm config)
                "reward/VQ_mean": vq_mean,
                "reward/MQ_mean": mq_mean,
                "reward/TA_mean": ta_mean,
                "reward/composite_local_mean": comp_mean,
            }

            # Raw (TaRoS-comparable) view, always logged when available
            if vq_raw_list is not None:
                vq_raw_mean = _mean(vq_raw_list)
                mq_raw_mean = _mean(mq_raw_list)
                ta_raw_mean = _mean(ta_raw_list)
                payload.update({
                    "reward/VQ_raw_mean": vq_raw_mean,
                    "reward/MQ_raw_mean": mq_raw_mean,
                    "reward/TA_raw_mean": ta_raw_mean,
                    "reward/composite_raw_mean": (
                        self.vq_coef * vq_raw_mean
                        + self.mq_coef * mq_raw_mean
                        + self.ta_coef * ta_raw_mean
                    ),
                })

            # Z-scored view, always logged when available
            if vq_norm_list is not None:
                vq_n_mean = _mean(vq_norm_list)
                mq_n_mean = _mean(mq_norm_list)
                ta_n_mean = _mean(ta_norm_list)
                payload.update({
                    "reward/VQ_norm_mean": vq_n_mean,
                    "reward/MQ_norm_mean": mq_n_mean,
                    "reward/TA_norm_mean": ta_n_mean,
                    "reward/composite_norm_mean": (
                        self.vq_coef * vq_n_mean
                        + self.mq_coef * mq_n_mean
                        + self.ta_coef * ta_n_mean
                    ),
                })

            # commit=False -> do NOT advance wandb's main _step counter.
            # The data still ships, indexed by `reward/_call` because of the
            # define_metric registration above.
            wandb.log(payload, commit=False)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[VideoAlign] dim-metric log skipped: {e}")
