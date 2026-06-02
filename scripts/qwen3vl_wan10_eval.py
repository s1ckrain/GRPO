#!/usr/bin/env python3
"""Generate 10 Wan2.1 videos, score them with Qwen3-VL, and make a human sheet.

This script is a small end-to-end smoke evaluation for the Qwen3-VL reward
model on Wan2.1 outputs. It deliberately keeps VQ/MQ/TA as separate Qwen judge
calls through the already-running reward server.

Outputs under ``--output-dir``:
    videos/                  Wan generated mp4 files
    videos_meta.csv          metadata from VideoAlign/wan_eval/generate.py
    qwen_reward_scores.csv   one row per video with raw VQ/MQ/TA/Overall
    human_pointwise.csv      blank human scoring sheet for side-by-side review
    qwen_responses.jsonl     raw Qwen responses/errors if --return-responses

Usage:
    # terminal 1: start Qwen reward server first
    export MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
    export PROMPT_DIR=/aigc/posttrain/siyuanfu/prompts
    bash /aigc/posttrain/siyuanfu/GRPO/scripts/start_qwen3vl_reward_server.sh

    # terminal 2: generate + score 10 videos
    cd /aigc/posttrain/siyuanfu/GRPO
    python scripts/qwen3vl_wan10_eval.py --gpu 0
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger("qwen3vl_wan10_eval")
pd = None
requests = None
tqdm = None

REQUIRED_METRIC_KEYS = ("VQ", "MQ", "TA", "composite")
REQUIRED_SCORE_COLUMNS = ("reward_VQ", "reward_MQ", "reward_TA", "reward_Overall")


def default_posttrain_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_pandas():
    global pd
    if pd is not None:
        return pd
    try:
        import pandas as _pd  # type: ignore
    except ImportError as e:
        raise ImportError("qwen3vl_wan10_eval.py requires pandas.") from e
    pd = _pd
    return pd


def ensure_requests():
    global requests
    if requests is not None:
        return requests
    try:
        import requests as _requests  # type: ignore
    except ImportError as e:
        raise ImportError("qwen3vl_wan10_eval.py requires requests.") from e
    requests = _requests
    return requests


def ensure_tqdm():
    global tqdm
    if tqdm is not None:
        return tqdm
    try:
        from tqdm import tqdm as _tqdm  # type: ignore
    except ImportError:
        def _tqdm(iterable, *args, **kwargs):
            return iterable
    tqdm = _tqdm
    return tqdm


def parse_args() -> argparse.Namespace:
    root = default_posttrain_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--videoalign-dir",
        default=os.environ.get("VIDEOALIGN_DIR", "/aigc/posttrain/siyuanfu/VideoAlign"),
        help="VideoAlign repo root; uses wan_eval/generate.py and wan_eval/prompts.txt.",
    )
    parser.add_argument(
        "--prompts-file",
        default=None,
        help="Prompt file for generation. Defaults to {videoalign_dir}/wan_eval/prompts.txt.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(root / "GRPO/qwen3vl-wan10-1fps"),
        help="Output directory for generated videos and score CSVs.",
    )
    parser.add_argument(
        "--model-name",
        default="/aigc/posttrain/siyuanfu/models/Wan2.1",
        help="Wan2.1 diffusers model path/id passed to VideoAlign/wan_eval/generate.py.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index for Wan generation.")
    parser.add_argument("--num-videos", type=int, default=10)
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Videos per prompt. Default 1 means 10 prompts -> 10 videos.",
    )
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--preset-key", default=None, choices=["1.3B", "14B"])
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--skip-generate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip Wan generation and score an existing videos_meta.csv.",
    )

    parser.add_argument("--server-url", default="http://127.0.0.1:18080")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--retry-attempts", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--fallback-value", type=float, default=0.0)
    parser.add_argument(
        "--return-responses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store Qwen raw responses/errors to qwen_responses.jsonl.",
    )
    parser.add_argument(
        "--no-resume-score",
        action="store_true",
        help="Ignore existing qwen_reward_scores.csv and score all rows again.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def run_wan_generation(args: argparse.Namespace) -> Path:
    videoalign_dir = Path(args.videoalign_dir).resolve()
    generate_py = videoalign_dir / "wan_eval" / "generate.py"
    if not generate_py.exists():
        raise FileNotFoundError(f"generate.py not found: {generate_py}")

    prompts_file = Path(args.prompts_file).resolve() if args.prompts_file else videoalign_dir / "wan_eval" / "prompts.txt"
    if not prompts_file.exists():
        raise FileNotFoundError(f"prompts file not found: {prompts_file}")

    out_dir = Path(args.output_dir).resolve()
    limit_prompts = max(1, (args.num_videos + args.num_seeds - 1) // args.num_seeds)

    cmd = [
        sys.executable,
        str(generate_py),
        "--prompts_file",
        str(prompts_file),
        "--output_dir",
        str(out_dir),
        "--model_name",
        args.model_name,
        "--num_seeds",
        str(args.num_seeds),
        "--seed_base",
        str(args.seed_base),
        "--gpu",
        str(args.gpu),
        "--limit",
        str(limit_prompts),
        "--dtype",
        args.dtype,
    ]
    optional_args = {
        "--preset_key": args.preset_key,
        "--num_inference_steps": args.num_inference_steps,
        "--guidance_scale": args.guidance_scale,
        "--height": args.height,
        "--width": args.width,
        "--num_frames": args.num_frames,
        "--fps": args.fps,
    }
    for flag, value in optional_args.items():
        if value is not None:
            cmd.extend([flag, str(value)])
    if args.negative_prompt:
        cmd.extend(["--negative_prompt", args.negative_prompt])

    logger.info("running Wan generation: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    meta_csv = out_dir / "videos_meta.csv"
    if not meta_csv.exists():
        raise FileNotFoundError(f"Wan generation did not produce {meta_csv}")
    return meta_csv


def video_file_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class QwenServerClient:
    def __init__(self, args: argparse.Namespace):
        req = ensure_requests()
        self.url = args.server_url.rstrip("/")
        self.timeout = float(args.timeout)
        self.retries = int(args.retry_attempts)
        self.retry_sleep = float(args.retry_sleep)
        self.session = req.Session()
        self.health_check()

    def health_check(self) -> None:
        r = self.session.get(f"{self.url}/health", timeout=min(self.timeout, 10.0))
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "ok":
            raise RuntimeError(f"Qwen reward server unhealthy: {data}")
        logger.info("connected to Qwen reward server: %s", data)

    def score_one(
        self,
        video_path: Path,
        prompt: str,
        args: argparse.Namespace,
    ) -> Dict[str, Any]:
        payload = {
            "prompts": [prompt],
            "videos": [video_file_to_b64(video_path)],
            "vq_coef": 1.0,
            "mq_coef": 1.0,
            "ta_coef": 1.0,
            "score_scale": "raw",
            "fallback_value": args.fallback_value,
            "return_responses": args.return_responses,
        }
        last_err: Optional[BaseException] = None
        for attempt in range(self.retries):
            try:
                r = self.session.post(
                    f"{self.url}/compute",
                    json=payload,
                    timeout=self.timeout,
                )
                r.raise_for_status()
                data = r.json()
                if data.get("error"):
                    raise RuntimeError(str(data["error"]))
                metrics = data.get("metrics")
                if not isinstance(metrics, list) or len(metrics) != 1:
                    raise RuntimeError(f"invalid metrics from server: {data}")
                metric = metrics[0]
                missing_keys = [key for key in REQUIRED_METRIC_KEYS if key not in metric]
                if missing_keys:
                    raise RuntimeError(
                        f"Qwen reward server response missing metric keys {missing_keys}: {metric}"
                    )
                return metric
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt + 1 >= self.retries:
                    break
                sleep_s = self.retry_sleep * (2**attempt)
                logger.warning(
                    "Qwen score failed on attempt %s/%s for %s: %s; retrying in %.1fs",
                    attempt + 1,
                    self.retries,
                    video_path,
                    e,
                    sleep_s,
                )
                time.sleep(sleep_s)
        raise RuntimeError(f"Qwen score failed for {video_path}: {last_err}")


def load_existing_scores(path: Path):
    pd_mod = ensure_pandas()
    if not path.exists():
        return pd_mod.DataFrame()
    return pd_mod.read_csv(path)


def split_complete_existing_scores(existing):
    if existing.empty:
        return set(), []
    if "video_id" not in existing.columns:
        logger.warning("existing score CSV has no video_id column; rescoring all rows")
        return set(), []

    missing_columns = [col for col in REQUIRED_SCORE_COLUMNS if col not in existing.columns]
    if missing_columns:
        logger.warning(
            "existing score CSV missing required columns %s; rescoring all rows",
            missing_columns,
        )
        return set(), []

    complete_mask = existing[list(REQUIRED_SCORE_COLUMNS)].notna().all(axis=1)
    complete = existing[complete_mask].copy()
    incomplete_count = int((~complete_mask).sum())
    if incomplete_count:
        logger.warning(
            "existing score CSV has %s incomplete row(s); rescoring those rows",
            incomplete_count,
        )
    return set(complete["video_id"].astype(str)), complete.to_dict("records")


def metric_to_row(meta_row: Mapping[str, Any], metric: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "video_id": meta_row["video_id"],
        "prompt_id": int(meta_row["prompt_id"]),
        "seed_offset": int(meta_row["seed_offset"]),
        "seed": int(meta_row["seed"]),
        "prompt": meta_row["prompt"],
        "video_path": meta_row["video_path"],
        "reward_VQ": float(metric["VQ"]),
        "reward_MQ": float(metric["MQ"]),
        "reward_TA": float(metric["TA"]),
        "reward_Overall": float(metric["composite"]),
        "score_scale": metric.get("score_scale", "raw"),
        "errors": json.dumps(metric.get("errors", {}), ensure_ascii=False),
    }


def write_human_template(scores_csv: Path) -> Path:
    pd_mod = ensure_pandas()
    df = pd_mod.read_csv(scores_csv)
    human = df[
        [
            "video_id",
            "prompt_id",
            "seed",
            "prompt",
            "video_path",
            "reward_VQ",
            "reward_MQ",
            "reward_TA",
            "reward_Overall",
        ]
    ].copy()
    for col in ("human_VQ", "human_MQ", "human_TA", "human_notes"):
        human[col] = ""
    out_path = scores_csv.parent / "human_pointwise.csv"
    human.to_csv(out_path, index=False)
    return out_path


def score_videos(args: argparse.Namespace, meta_csv: Path) -> Path:
    pd_mod = ensure_pandas()
    progress = ensure_tqdm()
    out_dir = Path(args.output_dir).resolve()
    scores_csv = out_dir / "qwen_reward_scores.csv"
    responses_path = out_dir / "qwen_responses.jsonl"

    meta = pd_mod.read_csv(meta_csv).head(args.num_videos).copy()
    existing = load_existing_scores(scores_csv)
    done_ids = set()
    rows: List[Dict[str, Any]] = []
    if not args.no_resume_score and not existing.empty:
        done_ids, rows = split_complete_existing_scores(existing)
        logger.info("resuming Qwen scores: %s existing rows", len(done_ids))

    client = QwenServerClient(args)
    for _, row in progress(meta.iterrows(), total=len(meta), desc="Qwen scoring"):
        video_id = str(row["video_id"])
        if video_id in done_ids:
            continue
        video_path = (out_dir / str(row["video_path"])).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"generated video not found: {video_path}")
        metric = client.score_one(video_path, str(row["prompt"]), args)
        rows.append(metric_to_row(row, metric))
        pd_mod.DataFrame(rows).to_csv(scores_csv, index=False)

        if args.return_responses or "errors" in metric:
            with open(responses_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "video_id": video_id,
                            "video_path": str(video_path),
                            "prompt": str(row["prompt"]),
                            "metric": metric,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    pd_mod.DataFrame(rows).to_csv(scores_csv, index=False)
    return scores_csv


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    ensure_pandas()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_csv = out_dir / "videos_meta.csv"

    if args.skip_generate:
        if not meta_csv.exists():
            raise FileNotFoundError(f"--skip-generate set but {meta_csv} does not exist")
    else:
        meta_csv = run_wan_generation(args)

    scores_csv = score_videos(args, meta_csv)
    human_csv = write_human_template(scores_csv)
    print(f"[done] generated videos: {out_dir / 'videos'}")
    print(f"[done] metadata:         {meta_csv}")
    print(f"[done] Qwen raw scores:  {scores_csv}")
    print(f"[done] human sheet:      {human_csv}")
    print("[done] reward columns are raw Qwen scores: reward_VQ, reward_MQ, reward_TA.")


if __name__ == "__main__":
    main()
