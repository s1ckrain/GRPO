#!/usr/bin/env python3
"""Generate seed pairs of Wan2.1 videos and let Qwen3-VL pick the better one (pairwise MQ).

For each prompt we render two videos with different seeds (seed_offset 0 -> A,
seed_offset 1 -> B), then ask the Qwen reward server's ``/compare`` endpoint to
choose the higher motion-quality clip using the pairwise prompt
(``GRPO/prompts/pairwise_mq.txt``).  Human winners are filled in later.

Outputs under ``--output-dir``:
    videos/                    Wan generated mp4 files (p{i}_s0.mp4, p{i}_s1.mp4)
    videos_meta.csv            metadata from VideoAlign/wan_eval/generate.py
    qwen_pairwise_scores.csv   one row per pair: A/B ids + qwen_winner (+ consistency)
    human_pairwise.csv         blank human sheet (fill human_winner with 'A'/'B')
    qwen_pairwise_responses.jsonl  raw Qwen responses if --return-responses

Usage:
    # terminal 1: start the Qwen reward server first (loads the model once)
    bash GRPO/qwen/server.sh

    # terminal 2: generate 10 pairs and score them
    python GRPO/qwen/pairwise_eval.py --gpu 0
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
from typing import Any, Dict, List, Optional

logger = logging.getLogger("pairwise_eval")
pd = None
requests = None
tqdm = None


def default_posttrain_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_pandas():
    global pd
    if pd is not None:
        return pd
    try:
        import pandas as _pd  # type: ignore
    except ImportError as e:
        raise ImportError("pairwise_eval.py requires pandas.") from e
    pd = _pd
    return pd


def ensure_requests():
    global requests
    if requests is not None:
        return requests
    try:
        import requests as _requests  # type: ignore
    except ImportError as e:
        raise ImportError("pairwise_eval.py requires requests.") from e
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--videoalign-dir",
        default=os.environ.get("VIDEOALIGN_DIR", "/aigc/posttrain/siyuanfu/VideoAlign"),
        help="VideoAlign repo root; uses wan_eval/generate.py and wan_eval/prompts.txt.",
    )
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(root / "GRPO/qwen3vl-pairwise-mq"),
        help="Output directory for generated videos and pairwise CSVs.",
    )
    parser.add_argument(
        "--model-name",
        default="/aigc/posttrain/siyuanfu/models/Wan2.1",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index for Wan generation.")
    parser.add_argument("--num-pairs", type=int, default=10, help="Number of prompts/pairs.")
    parser.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help="A uses seed_base, B uses seed_base+1 (one pair per prompt).",
    )
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
        help="Skip Wan generation and pair an existing videos_meta.csv.",
    )

    parser.add_argument("--server-url", default="http://127.0.0.1:18080")
    parser.add_argument(
        "--pairwise-prompt",
        default=str(root / "GRPO/prompts/pairwise_mq.txt"),
        help="Pairwise judge prompt file sent verbatim to the server as the instruction.",
    )
    parser.add_argument(
        "--include-prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append the video's text-to-video prompt after the instruction. "
        "Off for MQ (motion quality is judged prompt-agnostically).",
    )
    parser.add_argument(
        "--both-orders",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also query the swapped (B,A) order and record a position-bias "
        "consistency flag. Doubles Qwen calls.",
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--retry-attempts", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument(
        "--return-responses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store Qwen raw responses to qwen_pairwise_responses.jsonl.",
    )
    parser.add_argument(
        "--no-resume-score",
        action="store_true",
        help="Ignore existing qwen_pairwise_scores.csv and rescore all pairs.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def run_wan_generation(args: argparse.Namespace) -> Path:
    videoalign_dir = Path(args.videoalign_dir).resolve()
    generate_py = videoalign_dir / "wan_eval" / "generate.py"
    if not generate_py.exists():
        raise FileNotFoundError(f"generate.py not found: {generate_py}")

    prompts_file = (
        Path(args.prompts_file).resolve()
        if args.prompts_file
        else videoalign_dir / "wan_eval" / "prompts.txt"
    )
    if not prompts_file.exists():
        raise FileNotFoundError(f"prompts file not found: {prompts_file}")

    out_dir = Path(args.output_dir).resolve()
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
        "2",  # exactly two seeds per prompt -> one A/B pair
        "--seed_base",
        str(args.seed_base),
        "--gpu",
        str(args.gpu),
        "--limit",
        str(args.num_pairs),
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


def build_pairs(meta, out_dir: Path, num_pairs: int) -> List[Dict[str, Any]]:
    """Pair seed_offset 0 (A) with seed_offset 1 (B) for each prompt_id."""
    pairs: List[Dict[str, Any]] = []
    for prompt_id, group in meta.groupby("prompt_id"):
        by_offset = {int(r["seed_offset"]): r for _, r in group.iterrows()}
        if 0 not in by_offset or 1 not in by_offset:
            logger.warning(
                "prompt_id=%s does not have both seed_offset 0 and 1; skipping", prompt_id
            )
            continue
        a, b = by_offset[0], by_offset[1]
        pairs.append(
            {
                "prompt_id": int(prompt_id),
                "prompt": str(a["prompt"]),
                "video_a_id": str(a["video_id"]),
                "video_b_id": str(b["video_id"]),
                "video_a_path": str(a["video_path"]),
                "video_b_path": str(b["video_path"]),
                "seed_a": int(a["seed"]),
                "seed_b": int(b["seed"]),
            }
        )
        if len(pairs) >= num_pairs:
            break
    return pairs


class PairwiseClient:
    def __init__(self, args: argparse.Namespace):
        req = ensure_requests()
        self.url = args.server_url.rstrip("/")
        self.timeout = float(args.timeout)
        self.retries = int(args.retry_attempts)
        self.retry_sleep = float(args.retry_sleep)
        self.return_responses = bool(args.return_responses)
        self.session = req.Session()
        self.health_check()

    def health_check(self) -> None:
        r = self.session.get(f"{self.url}/health", timeout=min(self.timeout, 10.0))
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "ok":
            raise RuntimeError(f"Qwen reward server unhealthy: {data}")
        logger.info("connected to Qwen reward server: %s", data)

    def compare(
        self,
        instruction: str,
        video_a_b64: str,
        video_b_b64: str,
        prompt: Optional[str],
    ) -> Dict[str, Any]:
        payload = {
            "instruction": instruction,
            "videos_a": [video_a_b64],
            "videos_b": [video_b_b64],
            "return_responses": self.return_responses,
        }
        if prompt is not None:
            payload["prompts"] = [prompt]

        last_err: Optional[BaseException] = None
        for attempt in range(self.retries):
            try:
                r = self.session.post(f"{self.url}/compare", json=payload, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                if data.get("error"):
                    raise RuntimeError(str(data["error"]))
                results = data.get("results")
                if not isinstance(results, list) or len(results) != 1:
                    raise RuntimeError(f"invalid results from server: {data}")
                return results[0]
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt + 1 >= self.retries:
                    break
                sleep_s = self.retry_sleep * (2**attempt)
                logger.warning(
                    "compare failed on attempt %s/%s: %s; retrying in %.1fs",
                    attempt + 1,
                    self.retries,
                    e,
                    sleep_s,
                )
                time.sleep(sleep_s)
        raise RuntimeError(f"compare failed: {last_err}")


def _winner_to_video_id(winner: Optional[str], pair: Dict[str, Any]) -> Optional[str]:
    if winner == "A":
        return pair["video_a_id"]
    if winner == "B":
        return pair["video_b_id"]
    return None


def score_pairs(args: argparse.Namespace, meta_csv: Path) -> Path:
    pd_mod = ensure_pandas()
    progress = ensure_tqdm()
    out_dir = Path(args.output_dir).resolve()
    scores_csv = out_dir / "qwen_pairwise_scores.csv"
    responses_path = out_dir / "qwen_pairwise_responses.jsonl"

    instruction = Path(args.pairwise_prompt).read_text(encoding="utf-8").strip()
    if not instruction:
        raise ValueError(f"pairwise prompt file is empty: {args.pairwise_prompt}")

    meta = pd_mod.read_csv(meta_csv)
    pairs = build_pairs(meta, out_dir, args.num_pairs)
    if not pairs:
        raise RuntimeError("no valid A/B pairs found in videos_meta.csv")

    done_ids = set()
    rows: List[Dict[str, Any]] = []
    if not args.no_resume_score and scores_csv.exists():
        prev = pd_mod.read_csv(scores_csv)
        if "qwen_winner" in prev.columns and "prompt_id" in prev.columns:
            complete = prev[prev["qwen_winner"].isin(["A", "B"])]
            done_ids = set(complete["prompt_id"].astype(int))
            rows = complete.to_dict("records")
            logger.info("resuming pairwise scores: %s existing pair(s)", len(done_ids))

    client = PairwiseClient(args)
    for pair in progress(pairs, total=len(pairs), desc="Qwen pairwise"):
        if pair["prompt_id"] in done_ids:
            continue
        path_a = (out_dir / pair["video_a_path"]).resolve()
        path_b = (out_dir / pair["video_b_path"]).resolve()
        for p in (path_a, path_b):
            if not p.exists():
                raise FileNotFoundError(f"generated video not found: {p}")
        a_b64 = video_file_to_b64(path_a)
        b_b64 = video_file_to_b64(path_b)
        prompt_text = pair["prompt"] if args.include_prompt else None

        res = client.compare(instruction, a_b64, b_b64, prompt_text)
        winner = res.get("winner")
        row = dict(pair)
        row["qwen_winner"] = winner
        row["qwen_winner_video_id"] = _winner_to_video_id(winner, pair)
        row["error"] = res.get("error", "")

        if args.both_orders:
            res_swapped = client.compare(instruction, b_b64, a_b64, prompt_text)
            swapped = res_swapped.get("winner")
            # In the swapped call, position A held the original B clip.
            mapped = {"A": "B", "B": "A"}.get(swapped) if swapped else None
            row["qwen_winner_swapped"] = mapped
            row["consistent"] = bool(winner and mapped and winner == mapped)
            if res_swapped.get("error"):
                row["error"] = f'{row["error"]}; swapped: {res_swapped["error"]}'.strip("; ")

        rows.append(row)
        pd_mod.DataFrame(rows).to_csv(scores_csv, index=False)

        if args.return_responses:
            with open(responses_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"pair": pair, "result": res}, ensure_ascii=False) + "\n")

    pd_mod.DataFrame(rows).to_csv(scores_csv, index=False)
    return scores_csv


def write_human_template(scores_csv: Path) -> Path:
    pd_mod = ensure_pandas()
    df = pd_mod.read_csv(scores_csv)
    keep = [
        "prompt_id",
        "prompt",
        "video_a_id",
        "video_b_id",
        "video_a_path",
        "video_b_path",
        "qwen_winner",
    ]
    keep = [c for c in keep if c in df.columns]
    human = df[keep].copy()
    human["human_winner"] = ""
    human["human_notes"] = ""
    out_path = scores_csv.parent / "human_pairwise.csv"
    human.to_csv(out_path, index=False)
    return out_path


def print_summary(scores_csv: Path, both_orders: bool) -> None:
    pd_mod = ensure_pandas()
    df = pd_mod.read_csv(scores_csv)
    total = len(df)
    a_wins = int((df["qwen_winner"] == "A").sum())
    b_wins = int((df["qwen_winner"] == "B").sum())
    failed = total - a_wins - b_wins
    print("")
    print(f"[summary] pairs scored: {total}")
    print(f"[summary] Qwen winner A: {a_wins}  |  B: {b_wins}  |  unparsed/failed: {failed}")
    if both_orders and "consistent" in df.columns:
        consistent = int(df["consistent"].sum())
        print(
            f"[summary] order-consistency (A,B vs B,A agree): "
            f"{consistent}/{total} = {consistent / total * 100:.1f}%"
            if total
            else "[summary] order-consistency: n/a"
        )


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

    scores_csv = score_pairs(args, meta_csv)
    human_csv = write_human_template(scores_csv)
    print(f"[done] generated videos: {out_dir / 'videos'}")
    print(f"[done] metadata:         {meta_csv}")
    print(f"[done] Qwen pairwise:    {scores_csv}")
    print(f"[done] human sheet:      {human_csv}  (fill human_winner with 'A' or 'B')")
    print_summary(scores_csv, args.both_orders)


if __name__ == "__main__":
    main()
