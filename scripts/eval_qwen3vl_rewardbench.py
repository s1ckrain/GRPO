#!/usr/bin/env python3
"""Evaluate the Qwen3-VL video reward model on VideoGen-RewardBench.

This is the Qwen3-VL counterpart of ``VideoAlign/eval_videogen_rewardbench.py``.
It uses the same pair -> single -> pair conversion and the same accuracy
calculation, but obtains VQ/MQ/TA scores from the Qwen3-VL reward logic in
``scripts/qwen3vl_reward_server.py``.

Recommended flow:

  1. Start the Qwen3-VL reward server on one GPU:

       export MODEL_PATH="/path/to/Qwen3-VL-72B"
       export PROMPT_DIR="/path/to/prompts"
       ./scripts/start_qwen3vl_reward_server.sh

  2. Evaluate RewardBench through the server:

       python scripts/eval_qwen3vl_rewardbench.py \
         --backend server \
         --server-url http://127.0.0.1:18080 \
         --videoalign-dir /aigc/posttrain/siyuanfu/VideoAlign

For quick one-process debugging you can use ``--backend direct --model-path``;
that loads Qwen3-VL in this process and reuses the exact same scoring class as
the server.
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

logger = logging.getLogger("eval_qwen3vl_rewardbench")
pd = None
requests = None
tqdm = None

REWARD_COLUMNS = ("reward_VQ", "reward_MQ", "reward_TA", "reward_Overall")
LABEL_MAP = {"A": 1, "B": -1, "same": 0}


def default_posttrain_root() -> Path:
    # GRPO/scripts/eval_qwen3vl_rewardbench.py -> posttrain/
    return Path(__file__).resolve().parents[2]


def default_videoalign_dir() -> Path:
    return Path(os.environ.get("VIDEOALIGN_DIR", "/aigc/posttrain/siyuanfu/VideoAlign"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eval Qwen3-VL reward on VideoGen-RewardBench."
    )
    parser.add_argument(
        "--backend",
        choices=["server", "direct"],
        default="server",
        help="server: query an already-running reward server; direct: load Qwen3-VL here.",
    )
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:18080",
        help="Qwen3-VL reward server URL for --backend server.",
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("QWEN3VL_MODEL_PATH", ""),
        help="Qwen3-VL model path for --backend direct.",
    )
    parser.add_argument(
        "--prompt-dir",
        default=str(default_posttrain_root() / "prompts"),
        help="Directory containing aesthetic_quality.txt, motion_quality_noprompt.txt, "
        "and instruction_following.txt. Used by --backend direct; for --backend server "
        "the prompt dir is controlled by the server startup.",
    )
    parser.add_argument(
        "--videoalign-dir",
        default=str(default_videoalign_dir()),
        help="VideoAlign repo root. Defaults to /aigc/posttrain/siyuanfu/VideoAlign "
        "or VIDEOALIGN_DIR if set.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="RewardBench data directory. Defaults to {videoalign_dir}/datasets.",
    )
    parser.add_argument("--anno-path", default=None)
    parser.add_argument("--out-dir", default="qwen3vl-videogen-rewardbench-output")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing out_single.csv rows with finite scores.",
    )

    # Reward composition. Defaults match the training reward wrapper.
    parser.add_argument("--vq-coef", type=float, default=1.0)
    parser.add_argument("--mq-coef", type=float, default=1.0)
    parser.add_argument("--ta-coef", type=float, default=1.0)
    parser.add_argument("--score-scale", choices=["raw", "unit"], default="raw")
    parser.add_argument("--fallback-value", type=float, default=0.0)
    parser.add_argument(
        "--return-responses",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Store raw Qwen responses in out_single.responses.jsonl.",
    )

    # Server request controls.
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--retry-attempts", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)

    # Direct backend controls. These mirror qwen3vl_reward_server.py.
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
    )
    parser.add_argument(
        "--device-map-mode",
        default="auto",
        choices=["auto", "single", "none", "balanced", "balanced_low_0", "sequential"],
        help="Direct backend device map mode. auto is safest for large/meta-loaded models.",
    )
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--attn-implementation",
        default="auto",
        help="Direct backend attention backend: auto, flash_attention_2, sdpa, or none.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--video-fps",
        type=float,
        default=1.0,
        help="Qwen video frame sampling fps for --backend direct. "
        "For --backend server, set this when starting the server.",
    )
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
    return parser.parse_args()


def ensure_pandas():
    global pd
    if pd is not None:
        return pd
    try:
        import pandas as _pd  # type: ignore
    except ImportError as e:
        raise ImportError(
            "eval_qwen3vl_rewardbench.py requires pandas, matching "
            "VideoAlign/eval_videogen_rewardbench.py. Install pandas in the "
            "evaluation environment."
        ) from e
    pd = _pd
    return pd


def ensure_requests():
    global requests
    if requests is not None:
        return requests
    try:
        import requests as _requests  # type: ignore
    except ImportError as e:
        raise ImportError(
            "--backend server requires requests. Install requests in the "
            "evaluation environment, or use --backend direct."
        ) from e
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


def convert_pair_to_single(df_pair_anno: pd.DataFrame) -> pd.DataFrame:
    required = [
        "path_A",
        "path_B",
        "prompt",
        "fps_A",
        "num_frames_A",
        "fps_B",
        "num_frames_B",
    ]
    missing = [col for col in required if col not in df_pair_anno.columns]
    if missing:
        raise ValueError(f"annotation CSV missing required columns: {missing}")

    a_model = "A_model" if "A_model" in df_pair_anno.columns else None
    b_model = "B_model" if "B_model" in df_pair_anno.columns else None

    df_a = df_pair_anno[["path_A", "prompt", "fps_A", "num_frames_A"]].copy()
    df_a.columns = ["path", "prompt", "fps", "num_frames"]
    df_a["model"] = df_pair_anno[a_model] if a_model else "A"

    df_b = df_pair_anno[["path_B", "prompt", "fps_B", "num_frames_B"]].copy()
    df_b.columns = ["path", "prompt", "fps", "num_frames"]
    df_b["model"] = df_pair_anno[b_model] if b_model else "B"

    df_single = pd.concat([df_a, df_b], axis=0)
    df_single = df_single.drop_duplicates(subset=["path"])
    df_single = df_single.sort_values(by=["path"]).reset_index(drop=True)
    return df_single[["path", "model", "prompt", "fps", "num_frames"]]


def convert_single_to_pair(
    df_pair_anno: pd.DataFrame,
    df_single_pred: pd.DataFrame,
) -> pd.DataFrame:
    score_dict: Dict[str, Dict[str, float]] = {}
    for _, row in df_single_pred.iterrows():
        score_dict[row["path"]] = {
            key: float(row[key])
            for key in REWARD_COLUMNS
            if key in df_single_pred.columns
        }

    out = df_pair_anno.copy()
    for key in REWARD_COLUMNS:
        out[f"{key}_A"] = 0.0
        out[f"{key}_B"] = 0.0

    for idx, row in out.iterrows():
        for key in REWARD_COLUMNS:
            out.at[idx, f"{key}_A"] = score_dict[row["path_A"]][key]
            out.at[idx, f"{key}_B"] = score_dict[row["path_B"]][key]
    return out


def suff_stats(h: Iterable[int], m: Iterable[float], epsilon: float) -> Tuple[int, int, int, int, int]:
    c = d = th = tm = thm = 0
    for hi, mi in zip(h, m):
        if hi == 0 and abs(mi) <= epsilon:
            thm += 1
        elif hi == 0:
            th += 1
        elif abs(mi) <= epsilon:
            tm += 1
        elif hi * mi > 0:
            c += 1
        else:
            d += 1
    return c, d, th, tm, thm


def calc_acc(c: int, d: int, th: int, tm: int, thm: int) -> float:
    denom = c + d + th + tm + thm
    return 0.0 if denom == 0 else (c + thm) / denom


def calc_accuracy_with_ties(h: Iterable[int], m: Iterable[float]) -> float:
    h_list = list(h)
    m_list = list(m)
    c, d, th, tm, thm = suff_stats(h_list, m_list, -1)
    sorted_pairs = sorted(zip(h_list, m_list), key=lambda x: abs(x[1]))
    acc_star = float("-inf")
    epsilon_curr = -1.0
    stat = {"c": c, "d": d, "th": th, "tm": tm, "thm": thm}

    for hi, mi in sorted_pairs:
        if hi == 0 and abs(mi) < epsilon_curr:
            stat["thm"] -= 1
        elif hi == 0:
            stat["th"] -= 1
        elif abs(mi) < epsilon_curr:
            stat["tm"] -= 1
        elif hi * mi > 0:
            stat["c"] -= 1
        else:
            stat["d"] -= 1

        epsilon_curr = abs(mi)

        if hi == 0 and abs(mi) <= epsilon_curr:
            stat["thm"] += 1
        elif hi == 0:
            stat["th"] += 1
        elif abs(mi) <= epsilon_curr:
            stat["tm"] += 1
        elif hi * mi > 0:
            stat["c"] += 1
        else:
            stat["d"] += 1

        acc_star = max(acc_star, calc_acc(**stat))
    return 0.0 if acc_star == float("-inf") else float(acc_star)


def calc_accuracy_without_ties(h: Iterable[int], m: Iterable[float]) -> float:
    c, d, _, tm, _ = suff_stats(h, m, -1)
    denom = c + d + tm
    return 0.0 if denom == 0 else c / denom


def video_file_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class ServerScorer:
    def __init__(self, args: argparse.Namespace):
        req = ensure_requests()
        self.url = args.server_url.rstrip("/")
        self.timeout = float(args.timeout)
        self.retries = int(args.retry_attempts)
        self.retry_sleep = float(args.retry_sleep)
        self.session = req.Session()
        self._check_health()

    def _check_health(self) -> None:
        r = self.session.get(f"{self.url}/health", timeout=min(self.timeout, 10.0))
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "ok":
            raise RuntimeError(f"Qwen3-VL reward server is unhealthy: {data}")
        logger.info("connected to Qwen3-VL reward server: %s", data)

    def score_batch(
        self,
        video_paths: List[Path],
        prompts: List[str],
        args: argparse.Namespace,
    ) -> List[Dict[str, Any]]:
        payload = {
            "prompts": prompts,
            "videos": [video_file_to_b64(path) for path in video_paths],
            "vq_coef": args.vq_coef,
            "mq_coef": args.mq_coef,
            "ta_coef": args.ta_coef,
            "score_scale": args.score_scale,
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
                    raise RuntimeError(data["error"])
                metrics = data.get("metrics")
                if not isinstance(metrics, list) or len(metrics) != len(video_paths):
                    raise RuntimeError(f"invalid metrics returned by server: {data}")
                return metrics
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt + 1 >= self.retries:
                    break
                sleep_s = self.retry_sleep * (2**attempt)
                logger.warning(
                    "server scoring failed on attempt %s/%s: %s; retrying in %.1fs",
                    attempt + 1,
                    self.retries,
                    e,
                    sleep_s,
                )
                import time

                time.sleep(sleep_s)
        raise RuntimeError(f"server scoring failed after {self.retries} attempts: {last_err}")


class DirectScorer:
    def __init__(self, args: argparse.Namespace):
        if not args.model_path:
            raise ValueError("--model-path is required for --backend direct")

        scripts_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(scripts_dir))
        from qwen3vl_reward_server import LRUCache, Qwen3VLJudge  # noqa: E402

        self.judge = Qwen3VLJudge(args)
        self.cache = LRUCache(args.cache_size)

    def score_batch(
        self,
        video_paths: List[Path],
        prompts: List[str],
        args: argparse.Namespace,
    ) -> List[Dict[str, Any]]:
        metrics: List[Dict[str, Any]] = []
        for path, prompt in zip(video_paths, prompts):
            key = f"{path.resolve()}::{prompt}::{args.score_scale}::{args.vq_coef},{args.mq_coef},{args.ta_coef}"
            cached = self.cache.get(key)
            if cached is not None:
                metrics.append(dict(cached))
                continue
            metric = self.judge.score_video(
                str(path.resolve()),
                prompt,
                vq_coef=args.vq_coef,
                mq_coef=args.mq_coef,
                ta_coef=args.ta_coef,
                score_scale=args.score_scale,
                fallback_value=args.fallback_value,
                return_responses=args.return_responses,
            )
            if "errors" not in metric:
                self.cache.put(key, dict(metric))
            metrics.append(metric)
        return metrics


def build_scorer(args: argparse.Namespace):
    if args.backend == "server":
        return ServerScorer(args)
    return DirectScorer(args)


def resolve_video_path(data_dir: Path, rel_path: str) -> Path:
    path = Path(str(rel_path))
    if path.is_absolute():
        return path
    return (data_dir / path).resolve()


def load_or_init_single_predictions(
    df_pair_anno: pd.DataFrame,
    out_single_path: Path,
    resume: bool,
) -> pd.DataFrame:
    if resume and out_single_path.exists():
        df = pd.read_csv(out_single_path)
        required = {"path", "prompt", *REWARD_COLUMNS}
        if required.issubset(df.columns):
            return df
        logger.warning("existing %s has incompatible columns; rebuilding", out_single_path)

    df = convert_pair_to_single(df_pair_anno)
    for col in REWARD_COLUMNS:
        df[col] = float("nan")
    return df


def write_response_log(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def calculate_and_save_accuracy(df_pair_pred: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    attrs = [attr for attr in ("VQ", "MQ", "TA", "Overall") if attr in df_pair_pred.columns]
    results: Dict[str, Any] = {}
    for attr in attrs:
        pred_col = f"reward_{attr}"
        df_pair_pred[pred_col] = (
            df_pair_pred[f"reward_{attr}_A"] - df_pair_pred[f"reward_{attr}_B"]
        )
        human = df_pair_pred[attr].map(LABEL_MAP)
        if human.isna().any():
            bad_values = df_pair_pred.loc[human.isna(), attr]
            if bad_values.isna().all():
                logger.warning(
                    "skipping %s accuracy because the human-label column is empty/NaN",
                    attr,
                )
                continue
            bad = sorted(set(bad_values.astype(str)))
            logger.warning(
                "skipping %s accuracy because it contains unsupported labels: %s",
                attr,
                bad,
            )
            continue

        results[f"{attr} Accuracy"] = {
            "with_ties": calc_accuracy_with_ties(human.astype(int), df_pair_pred[pred_col]),
            "without_ties": calc_accuracy_without_ties(human.astype(int), df_pair_pred[pred_col]),
        }
        print(
            f"{attr} Accuracy: "
            f"With ties: {results[f'{attr} Accuracy']['with_ties']}, "
            f"Without ties: {results[f'{attr} Accuracy']['without_ties']}"
        )

    with open(out_dir / "accuracy.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    return results


def main() -> None:
    args = parse_args()
    ensure_pandas()
    progress = ensure_tqdm()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    videoalign_dir = Path(args.videoalign_dir).resolve()
    data_dir = (
        Path(args.data_dir).resolve()
        if args.data_dir
        else videoalign_dir / "datasets"
    )
    anno_path = (
        Path(args.anno_path).resolve()
        if args.anno_path
        else data_dir / "videogen-rewardbench.csv"
    )
    if not anno_path.exists():
        raise FileNotFoundError(
            f"annotation CSV not found: {anno_path}. "
            "Pass --anno-path to the RewardBench CSV."
        )

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_single_path = out_dir / "out_single.csv"
    out_pair_path = out_dir / "out_pair.csv"
    response_log_path = out_dir / "out_single.responses.jsonl"

    df_pair_anno = pd.read_csv(anno_path)
    if args.limit is not None:
        # Limit pairs first, then de-duplicate single videos from those pairs.
        df_pair_anno = df_pair_anno.head(args.limit).reset_index(drop=True)

    df_single_pred = load_or_init_single_predictions(
        df_pair_anno,
        out_single_path,
        resume=args.resume,
    )
    scorer = build_scorer(args)

    metadata = {
        "backend": args.backend,
        "server_url": args.server_url if args.backend == "server" else None,
        "model_path": args.model_path if args.backend == "direct" else None,
        "prompt_dir": args.prompt_dir,
        "videoalign_dir": str(videoalign_dir),
        "data_dir": str(data_dir),
        "anno_path": str(anno_path),
        "vq_coef": args.vq_coef,
        "mq_coef": args.mq_coef,
        "ta_coef": args.ta_coef,
        "score_scale": args.score_scale,
        "fallback_value": args.fallback_value,
    }
    with open(out_dir / "eval_config.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    pending_paths: List[Path] = []
    pending_prompts: List[str] = []
    pending_indices: List[int] = []

    pbar = progress(df_single_pred.iterrows(), total=len(df_single_pred), desc="Scoring")
    for idx, row in pbar:
        existing = row.get("reward_Overall")
        if args.resume and pd.notna(existing):
            continue

        video_path = resolve_video_path(data_dir, str(row["path"]))
        if not video_path.exists():
            logger.warning("missing video %s; using fallback", video_path)
            metric = {
                "VQ": args.fallback_value,
                "MQ": args.fallback_value,
                "TA": args.fallback_value,
                "composite": (
                    args.vq_coef * args.fallback_value
                    + args.mq_coef * args.fallback_value
                    + args.ta_coef * args.fallback_value
                ),
                "errors": {"missing_video": str(video_path)},
            }
            store_metric(df_single_pred, idx, metric)
            continue

        pending_paths.append(video_path)
        pending_prompts.append(str(row["prompt"]))
        pending_indices.append(idx)

        flush = len(pending_paths) >= max(1, args.batch_size)
        if not flush:
            continue
        flush_batch(
            scorer,
            args,
            df_single_pred,
            pending_paths,
            pending_prompts,
            pending_indices,
            out_single_path,
            response_log_path,
        )

    if pending_paths:
        flush_batch(
            scorer,
            args,
            df_single_pred,
            pending_paths,
            pending_prompts,
            pending_indices,
            out_single_path,
            response_log_path,
        )

    df_single_pred.to_csv(out_single_path, index=False)
    df_pair_pred = convert_single_to_pair(df_pair_anno, df_single_pred)
    df_pair_pred.to_csv(out_pair_path, index=False)
    calculate_and_save_accuracy(df_pair_pred, out_dir)
    print(f"[eval] single predictions: {out_single_path}")
    print(f"[eval] pair predictions:   {out_pair_path}")
    print(f"[eval] accuracy:           {out_dir / 'accuracy.json'}")


def store_metric(df: pd.DataFrame, idx: int, metric: Mapping[str, Any]) -> None:
    df.at[idx, "reward_VQ"] = float(metric.get("VQ", 0.0))
    df.at[idx, "reward_MQ"] = float(metric.get("MQ", 0.0))
    df.at[idx, "reward_TA"] = float(metric.get("TA", 0.0))
    df.at[idx, "reward_Overall"] = float(metric.get("composite", 0.0))


def flush_batch(
    scorer,
    args: argparse.Namespace,
    df_single_pred: pd.DataFrame,
    pending_paths: List[Path],
    pending_prompts: List[str],
    pending_indices: List[int],
    out_single_path: Path,
    response_log_path: Path,
) -> None:
    try:
        metrics = scorer.score_batch(pending_paths, pending_prompts, args)
    except Exception as e:  # noqa: BLE001
        logger.warning("batch scoring failed; using fallback for %s videos: %s", len(pending_paths), e)
        fallback_composite = (
            args.vq_coef * args.fallback_value
            + args.mq_coef * args.fallback_value
            + args.ta_coef * args.fallback_value
        )
        metrics = [
            {
                "VQ": args.fallback_value,
                "MQ": args.fallback_value,
                "TA": args.fallback_value,
                "composite": fallback_composite,
                "errors": {"batch_error": str(e)},
            }
            for _ in pending_paths
        ]

    response_rows: List[Dict[str, Any]] = []
    for path, prompt, idx, metric in zip(pending_paths, pending_prompts, pending_indices, metrics):
        store_metric(df_single_pred, idx, metric)
        if args.return_responses or "errors" in metric:
            response_rows.append(
                {
                    "path": str(path),
                    "prompt": prompt,
                    "metric": metric,
                }
            )

    write_response_log(response_log_path, response_rows)
    df_single_pred.to_csv(out_single_path, index=False)
    pending_paths.clear()
    pending_prompts.clear()
    pending_indices.clear()


if __name__ == "__main__":
    main()
