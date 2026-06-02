#!/usr/bin/env python3
"""Compute per-dimension agreement accuracy between human and Qwen3-VL scores.

This is the standalone scoring step that runs *after* a human has filled in
``human_pointwise.csv`` produced by ``qwen3vl_wan10_eval.py``.  For each
dimension (VQ / MQ / TA) a video counts as accurate when the human score and the
Qwen reward agree within ``--threshold`` (default 2, inclusive); the per-dimension
accuracy is ``n_accurate / n_evaluated`` over the rows where the human cell is
filled in.

Inputs (both default to live under ``--output-dir``):
    qwen_reward_scores.csv   reward_VQ / reward_MQ / reward_TA / reward_Overall
    human_pointwise.csv      human_VQ / human_MQ / human_TA

Outputs under ``--output-dir`` (override with the explicit flags):
    accuracy_detail.csv      one row per video x evaluated dimension with diff/flag
    (a summary table is also printed to stdout)

Usage:
    # only look at MQ
    python qwen3vl_accuracy.py --output-dir GRPO/qwen3vl-2fps --dimensions MQ

    # auto-detect whichever dimensions the human actually filled in
    python qwen3vl_accuracy.py --output-dir GRPO/qwen3vl-2fps
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

ALL_DIMENSIONS = ("VQ", "MQ", "TA")


def ensure_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as e:
        raise ImportError("qwen3vl_accuracy.py requires pandas.") from e
    return pd


def parse_dimensions(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory holding qwen_reward_scores.csv and human_pointwise.csv.",
    )
    parser.add_argument(
        "--scores-csv",
        default=None,
        help="Qwen score CSV. Defaults to {output_dir}/qwen_reward_scores.csv.",
    )
    parser.add_argument(
        "--human-csv",
        default=None,
        help="Filled human sheet. Defaults to {output_dir}/human_pointwise.csv.",
    )
    parser.add_argument(
        "--detail-csv",
        default=None,
        help="Output per-row detail CSV. Defaults to {output_dir}/accuracy_detail.csv.",
    )
    parser.add_argument(
        "--dimensions",
        default=None,
        help="Comma-separated subset of VQ,MQ,TA. "
        "If omitted, auto-detects whichever dimensions the human filled in.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="A video is accurate when |human - qwen| <= threshold (inclusive). Default 2.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace):
    out_dir = Path(args.output_dir).resolve() if args.output_dir else None
    scores_csv = Path(args.scores_csv).resolve() if args.scores_csv else None
    human_csv = Path(args.human_csv).resolve() if args.human_csv else None
    detail_csv = Path(args.detail_csv).resolve() if args.detail_csv else None

    if scores_csv is None:
        if out_dir is None:
            raise ValueError("provide --scores-csv or --output-dir")
        scores_csv = out_dir / "qwen_reward_scores.csv"
    if human_csv is None:
        if out_dir is None:
            raise ValueError("provide --human-csv or --output-dir")
        human_csv = out_dir / "human_pointwise.csv"
    if detail_csv is None:
        base = out_dir if out_dir is not None else scores_csv.parent
        detail_csv = base / "accuracy_detail.csv"

    if not scores_csv.exists():
        raise FileNotFoundError(f"Qwen score CSV not found: {scores_csv}")
    if not human_csv.exists():
        raise FileNotFoundError(f"human sheet not found: {human_csv}")
    return scores_csv, human_csv, detail_csv


def detect_dimensions(pd, human_df, requested: Optional[List[str]]) -> List[str]:
    if requested is not None:
        missing = [d for d in requested if f"human_{d}" not in human_df.columns]
        if missing:
            raise ValueError(
                f"human sheet has no columns for dimensions {missing} "
                f"(expected human_{missing[0]} ...)"
            )
        return requested
    detected: List[str] = []
    for dim in ALL_DIMENSIONS:
        col = f"human_{dim}"
        if col not in human_df.columns:
            continue
        values = pd.to_numeric(human_df[col], errors="coerce")
        if values.notna().any():
            detected.append(dim)
    if not detected:
        raise ValueError(
            "no human scores found in any of human_VQ/human_MQ/human_TA; "
            "fill in the sheet or pass --dimensions explicitly"
        )
    return detected


def main() -> None:
    args = parse_args()
    pd = ensure_pandas()
    requested = parse_dimensions(args.dimensions)
    scores_csv, human_csv, detail_csv = resolve_paths(args)

    scores_df = pd.read_csv(scores_csv)
    human_df = pd.read_csv(human_csv)
    for df, name in ((scores_df, scores_csv), (human_df, human_csv)):
        if "video_id" not in df.columns:
            raise ValueError(f"{name} has no video_id column to align on")

    dims = detect_dimensions(pd, human_df, requested)

    score_cols = ["video_id"] + [f"reward_{d}" for d in dims]
    missing_score_cols = [c for c in score_cols if c not in scores_df.columns]
    if missing_score_cols:
        raise ValueError(f"{scores_csv} missing columns {missing_score_cols}")

    prompt_cols = [c for c in ("prompt_id", "prompt", "video_path") if c in human_df.columns]
    human_cols = ["video_id"] + prompt_cols + [f"human_{d}" for d in dims]
    merged = human_df[human_cols].merge(
        scores_df[score_cols], on="video_id", how="inner", validate="one_to_one"
    )
    if merged.empty:
        raise ValueError("no overlapping video_id between the human sheet and Qwen scores")

    detail_rows = []
    summary_rows = []
    for dim in dims:
        human_vals = pd.to_numeric(merged[f"human_{dim}"], errors="coerce")
        qwen_vals = pd.to_numeric(merged[f"reward_{dim}"], errors="coerce")
        evaluable = human_vals.notna() & qwen_vals.notna()
        n_eval = int(evaluable.sum())
        n_accurate = 0
        for idx in merged.index[evaluable]:
            h = float(human_vals[idx])
            q = float(qwen_vals[idx])
            diff = abs(h - q)
            accurate = diff <= args.threshold
            n_accurate += int(accurate)
            row = {"video_id": merged.at[idx, "video_id"]}
            for c in prompt_cols:
                row[c] = merged.at[idx, c]
            row["dimension"] = dim
            row["human"] = h
            row["qwen"] = q
            row["abs_diff"] = diff
            row["accurate"] = bool(accurate)
            detail_rows.append(row)
        accuracy = (n_accurate / n_eval) if n_eval else float("nan")
        summary_rows.append(
            {
                "dimension": dim,
                "n_evaluated": n_eval,
                "n_accurate": n_accurate,
                "accuracy": accuracy,
            }
        )

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(detail_csv, index=False)
    summary_df = pd.DataFrame(summary_rows)

    print(f"threshold (|human - qwen| <=): {args.threshold}")
    print(f"aligned videos: {len(merged)}")
    print("")
    print("per-dimension accuracy:")
    for r in summary_rows:
        acc = r["accuracy"]
        acc_str = "n/a" if acc != acc else f"{acc * 100:.1f}%"  # acc != acc detects NaN
        print(
            f"  {r['dimension']:<3} accuracy={acc_str:>7}  "
            f"({r['n_accurate']}/{r['n_evaluated']})"
        )
    print("")
    print(f"[done] per-row detail: {detail_csv}")

    # Make the summary easy to grep/pipe.
    print("")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)
