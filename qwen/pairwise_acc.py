#!/usr/bin/env python3
"""Compute Qwen-vs-human agreement for pairwise MQ judgments.

Reads ``human_pairwise.csv`` (which already carries both ``qwen_winner`` and the
hand-labelled ``human_winner``) and reports how often Qwen agrees with the human.

human_winner values:
    A / B   the human says that clip is clearly better
    T       a tie ("差不多") -- EITHER A or B is acceptable, so any valid Qwen
            pick (A or B) counts as correct; only a missing/unparsed Qwen answer
            counts as wrong here.

Metrics printed:
    overall accuracy   ties count as correct when Qwen produced a valid A/B
    decisive accuracy  restricted to human A/B pairs (ties excluded)

If ``qwen_pairwise_scores.csv`` is present it is used as the authoritative source
of ``qwen_winner`` (joined on prompt_id) and the A/B video ids are cross-checked
to surface any drift in the fixed pair set.

Usage:
    # defaults to GRPO/qwen/human_pairwise.csv next to this script
    python GRPO/qwen/pairwise_acc.py

    # or point at a run directory
    python GRPO/qwen/pairwise_acc.py --output-dir GRPO/qwen3vl-pairwise-mq
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


def ensure_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as e:
        raise ImportError("pairwise_acc.py requires pandas.") from e
    return pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", default=None, help="Run directory holding the CSVs.")
    parser.add_argument(
        "--human-csv",
        default=None,
        help="Human sheet with qwen_winner + human_winner. "
        "Defaults to {output_dir}/human_pairwise.csv, else ./human_pairwise.csv next to this script.",
    )
    parser.add_argument(
        "--scores-csv",
        default=None,
        help="Optional Qwen pairwise scores for authoritative qwen_winner. "
        "Defaults to {output_dir}/qwen_pairwise_scores.csv if it exists.",
    )
    parser.add_argument("--detail-csv", default=None, help="Defaults to <human_csv dir>/pairwise_accuracy_detail.csv.")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace):
    out_dir = Path(args.output_dir).resolve() if args.output_dir else None
    human_csv = Path(args.human_csv).resolve() if args.human_csv else None
    scores_csv = Path(args.scores_csv).resolve() if args.scores_csv else None
    detail_csv = Path(args.detail_csv).resolve() if args.detail_csv else None

    if human_csv is None:
        human_csv = (
            out_dir / "human_pairwise.csv"
            if out_dir is not None
            else Path(__file__).resolve().parent / "human_pairwise.csv"
        )
    if not human_csv.exists():
        raise FileNotFoundError(f"human pairwise sheet not found: {human_csv}")

    if scores_csv is None:
        candidate = (out_dir or human_csv.parent) / "qwen_pairwise_scores.csv"
        scores_csv = candidate if candidate.exists() else None

    if detail_csv is None:
        detail_csv = human_csv.parent / "pairwise_accuracy_detail.csv"
    return human_csv, scores_csv, detail_csv


def _norm(value, allowed) -> Optional[str]:
    pd = ensure_pandas()
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().upper()
    return text if text in allowed else None


def main() -> None:
    args = parse_args()
    pd = ensure_pandas()
    human_csv, scores_csv, detail_csv = resolve_paths(args)

    human = pd.read_csv(human_csv)
    if "human_winner" not in human.columns or "prompt_id" not in human.columns:
        raise ValueError(f"{human_csv} must have 'prompt_id' and 'human_winner' columns")

    drift = []
    if scores_csv is not None:
        scores = pd.read_csv(scores_csv)
        if "qwen_winner" not in scores.columns or "prompt_id" not in scores.columns:
            raise ValueError(f"{scores_csv} must have 'prompt_id' and 'qwen_winner' columns")
        cols = ["prompt_id", "qwen_winner"] + [
            c for c in ("video_a_id", "video_b_id") if c in scores.columns
        ]
        merged = human.merge(scores[cols], on="prompt_id", how="inner", suffixes=("", "_qwen"))
        qwen_col = "qwen_winner_qwen" if "qwen_winner_qwen" in merged.columns else "qwen_winner"
        for col in ("video_a_id", "video_b_id"):
            qc = f"{col}_qwen"
            if col in merged.columns and qc in merged.columns:
                mism = merged[merged[col].astype(str) != merged[qc].astype(str)]
                for _, r in mism.iterrows():
                    drift.append((int(r["prompt_id"]), col, r[col], r[qc]))
    else:
        if "qwen_winner" not in human.columns:
            raise ValueError(
                f"{human_csv} has no 'qwen_winner' column and no qwen_pairwise_scores.csv was found"
            )
        merged = human
        qwen_col = "qwen_winner"

    detail_rows = []
    n_eval = 0
    n_correct = 0
    n_tie = 0
    n_decisive = 0
    n_decisive_correct = 0
    n_qwen_unparsed = 0

    for _, r in merged.iterrows():
        human_w = _norm(r["human_winner"], {"A", "B", "T"})
        if human_w is None:
            continue  # unlabelled row
        qwen_w = _norm(r[qwen_col], {"A", "B"})
        n_eval += 1
        if qwen_w is None:
            n_qwen_unparsed += 1

        if human_w == "T":
            n_tie += 1
            correct = qwen_w is not None  # either A or B is acceptable
        else:
            n_decisive += 1
            correct = (qwen_w is not None) and (qwen_w == human_w)
            n_decisive_correct += int(correct)
        n_correct += int(correct)

        row = {"prompt_id": int(r["prompt_id"])}
        if "prompt" in merged.columns:
            row["prompt"] = r["prompt"]
        row["qwen_winner"] = qwen_w
        row["human_winner"] = human_w
        row["is_tie"] = human_w == "T"
        row["correct"] = bool(correct)
        detail_rows.append(row)

    pd.DataFrame(detail_rows).to_csv(detail_csv, index=False)

    overall = (n_correct / n_eval) if n_eval else float("nan")
    decisive = (n_decisive_correct / n_decisive) if n_decisive else float("nan")

    def pct(x: float) -> str:
        return "n/a" if x != x else f"{x * 100:.1f}%"

    print(f"human sheet: {human_csv}")
    if scores_csv is not None:
        print(f"qwen scores: {scores_csv}")
    print("")
    print(f"labelled pairs evaluated: {n_eval}  (ties: {n_tie}, decisive A/B: {n_decisive})")
    print(f"overall accuracy (ties = any valid A/B is correct): {pct(overall)} ({n_correct}/{n_eval})")
    print(f"decisive accuracy (ties excluded):                  {pct(decisive)} ({n_decisive_correct}/{n_decisive})")
    if n_qwen_unparsed:
        print(f"[note] {n_qwen_unparsed} pair(s) had no parseable Qwen winner (counted as wrong)")
    if drift:
        print("")
        print(f"[WARNING] {len(drift)} A/B id mismatch(es) vs qwen scores (pair set drifted):")
        for prompt_id, col, hv, qv in drift:
            print(f"  prompt_id={prompt_id} {col}: human={hv} qwen={qv}")
    print("")
    print(f"[done] per-pair detail: {detail_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)
