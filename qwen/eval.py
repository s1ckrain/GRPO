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

ALL_DIMENSIONS = ("VQ", "MQ", "TA")


def parse_dimensions(raw: str) -> List[str]:
    parts = [p.strip().upper() for p in str(raw).split(",") if p.strip()]
    if not parts:
        raise ValueError("--dimensions must list at least one of VQ,MQ,TA")
    seen: List[str] = []
    for p in parts:
        if p not in ALL_DIMENSIONS:
            raise ValueError(f"invalid dimension {p!r}; choose from {ALL_DIMENSIONS}")
        if p not in seen:
            seen.append(p)
    return seen


def metric_keys_for(dims: List[str]) -> tuple:
    return tuple(dims) + ("composite",)


def score_columns_for(dims: List[str]) -> tuple:
    return tuple(f"reward_{d}" for d in dims) + ("reward_Overall",)


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
        default=str(root / "GRPO/ov-16frames"),
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
    parser.add_argument(
        "--dimensions",
        default="MQ",
        help="Comma-separated subset of VQ,MQ,TA to ask Qwen to score, e.g. 'MQ'. "
        "Unselected dimensions are left blank in the score/human CSVs.",
    )
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
    parser.add_argument(
        "--human-ground-truth",
        default=str(Path(__file__).resolve().parent / "human_ground_truth.csv"),
        help="Fixed human scores keyed by prompt text. Auto-filled into human_pointwise.csv "
        "so deterministic Wan videos never need re-scoring. Set to '' to disable.",
    )
    parser.add_argument(
        "--strict-ground-truth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if a scored video's prompt is missing from the ground truth, instead of "
        "silently leaving human_* blank. Guards against prompts/videos no longer being fixed.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    args.dim_list = parse_dimensions(args.dimensions)
    return args


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
            "dimensions": args.dim_list,
        }
        required_keys = metric_keys_for(args.dim_list)
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
                missing_keys = [key for key in required_keys if key not in metric]
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


def split_complete_existing_scores(existing, required_columns):
    if existing.empty:
        return set(), []
    if "video_id" not in existing.columns:
        logger.warning("existing score CSV has no video_id column; rescoring all rows")
        return set(), []

    missing_columns = [col for col in required_columns if col not in existing.columns]
    if missing_columns:
        logger.warning(
            "existing score CSV missing required columns %s; rescoring all rows",
            missing_columns,
        )
        return set(), []

    complete_mask = existing[list(required_columns)].notna().all(axis=1)
    complete = existing[complete_mask].copy()
    incomplete_count = int((~complete_mask).sum())
    if incomplete_count:
        logger.warning(
            "existing score CSV has %s incomplete row(s); rescoring those rows",
            incomplete_count,
        )
    return set(complete["video_id"].astype(str)), complete.to_dict("records")


def metric_to_row(
    meta_row: Mapping[str, Any],
    metric: Mapping[str, Any],
    dims: List[str],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "video_id": meta_row["video_id"],
        "prompt_id": int(meta_row["prompt_id"]),
        "seed_offset": int(meta_row["seed_offset"]),
        "seed": int(meta_row["seed"]),
        "prompt": meta_row["prompt"],
        "video_path": meta_row["video_path"],
    }
    selected = set(dims)
    for dim in ALL_DIMENSIONS:
        row[f"reward_{dim}"] = float(metric[dim]) if dim in selected else None
    row["reward_Overall"] = float(metric["composite"])
    row["scored_dimensions"] = ",".join(dims)
    row["score_scale"] = metric.get("score_scale", "raw")
    row["errors"] = json.dumps(metric.get("errors", {}), ensure_ascii=False)
    return row


HUMAN_COLUMNS = ("human_VQ", "human_MQ", "human_TA", "human_notes")


def _normalize_prompt(value: Any) -> str:
    return str(value).strip()


def load_ground_truth(path: Optional[Path]):
    pd_mod = ensure_pandas()
    if path is None:
        return None
    if not path.exists():
        logger.warning("human ground truth not found at %s; human sheet will be blank", path)
        return None
    gt = pd_mod.read_csv(path)
    if "prompt" not in gt.columns:
        raise ValueError(f"ground truth {path} must have a 'prompt' column to key on")
    gt = gt.copy()
    gt["_prompt_key"] = gt["prompt"].map(_normalize_prompt)
    dup = int(gt["_prompt_key"].duplicated().sum())
    if dup:
        logger.warning("ground truth has %s duplicate prompt key(s); keeping first", dup)
        gt = gt.drop_duplicates("_prompt_key", keep="first")
    return gt.set_index("_prompt_key")


def write_human_template(scores_csv: Path, args: argparse.Namespace) -> Path:
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
    for col in HUMAN_COLUMNS:
        human[col] = ""

    gt_path = (
        Path(args.human_ground_truth).resolve()
        if getattr(args, "human_ground_truth", "")
        else None
    )
    gt = load_ground_truth(gt_path)
    out_path = scores_csv.parent / "human_pointwise.csv"

    if gt is None:
        human.to_csv(out_path, index=False)
        return out_path

    # Align on the prompt TEXT (the instruction), not video_id: video_id is only a
    # positional label, so keying on the prompt makes any drift in the fixed
    # prompt/video set visible instead of silently mapping the wrong human score.
    gt_human_cols = [c for c in HUMAN_COLUMNS if c in gt.columns]
    keys = human["prompt"].map(_normalize_prompt)
    matched = 0
    unmatched: List[str] = []
    for idx, key in keys.items():
        if key in gt.index:
            matched += 1
            for col in gt_human_cols:
                val = gt.at[key, col]
                human.at[idx, col] = "" if pd_mod.isna(val) else val
        else:
            unmatched.append(str(human.at[idx, "video_id"]))

    run_keys = set(keys.tolist())
    gt_only = [k for k in gt.index if k not in run_keys]

    total = len(human)
    logger.info("ground truth matched %s/%s scored videos by prompt", matched, total)
    if unmatched:
        logger.warning(
            "%s scored video(s) have NO prompt match in ground truth (human_* left blank): %s",
            len(unmatched),
            unmatched,
        )
    if gt_only:
        logger.warning(
            "%s ground-truth prompt(s) were NOT produced in this run", len(gt_only)
        )
    if unmatched and getattr(args, "strict_ground_truth", False):
        raise RuntimeError(
            f"--strict-ground-truth: {len(unmatched)} scored video(s) had no matching "
            f"prompt in {gt_path}. The prompt/video set is no longer fixed as expected. "
            f"Offending video_ids: {unmatched}. Wrote scores to {scores_csv} but refused "
            f"to write a possibly-misaligned human sheet."
        )

    human.to_csv(out_path, index=False)
    return out_path


def score_videos(args: argparse.Namespace, meta_csv: Path) -> Path:
    pd_mod = ensure_pandas()
    progress = ensure_tqdm()
    out_dir = Path(args.output_dir).resolve()
    scores_csv = out_dir / "qwen_reward_scores.csv"
    responses_path = out_dir / "qwen_responses.jsonl"

    meta = pd_mod.read_csv(meta_csv).head(args.num_videos).copy()
    required_columns = score_columns_for(args.dim_list)
    existing = load_existing_scores(scores_csv)
    done_ids = set()
    rows: List[Dict[str, Any]] = []
    if not args.no_resume_score and not existing.empty:
        done_ids, rows = split_complete_existing_scores(existing, required_columns)
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
        rows.append(metric_to_row(row, metric, args.dim_list))
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

    logger.info("scoring dimensions: %s", ",".join(args.dim_list))
    scores_csv = score_videos(args, meta_csv)
    human_csv = write_human_template(scores_csv, args)
    print(f"[done] generated videos: {out_dir / 'videos'}")
    print(f"[done] metadata:         {meta_csv}")
    print(f"[done] Qwen raw scores:  {scores_csv}")
    print(f"[done] human sheet:      {human_csv}")
    print(f"[done] scored dimensions: {','.join(args.dim_list)} (unselected reward_* left blank)")
    print("[done] reward columns are raw Qwen scores: reward_VQ, reward_MQ, reward_TA.")


if __name__ == "__main__":
    main()
