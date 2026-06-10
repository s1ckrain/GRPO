#!/usr/bin/env python3
"""Compute Qwen-vs-human agreement for pairwise MQ judgments.

Runs *after* a human fills ``human_winner`` ('A' or 'B') in ``human_pairwise.csv``
produced by ``pairwise_eval.py``.  Agreement = fraction of pairs where the Qwen
winner matches the human winner, over pairs the human actually labelled.

Alignment is keyed on ``prompt_id`` and additionally checks that ``video_a_id`` /
``video_b_id`` match between the two files, so any drift in the fixed pair set is
surfaced instead of silently comparing the wrong videos.

Inputs (default under ``--output-dir``):
    qwen_pairwise_scores.csv   prompt_id, video_a_id, video_b_id, qwen_winner
    human_pairwise.csv         prompt_id, ..., human_winner

Output:
    pairwise_accuracy_detail.csv   per-pair qwen/human winner + agree flag
    (a summary is also printed to stdout)

Usage:
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=None, help="Directory holding the two CSVs.")
    parser.add_argument("--scores-csv", default=None, help="Defaults to {output_dir}/qwen_pairwise_scores.csv.")
    parser.add_argument("--human-csv", default=None, help="Defaults to {output_dir}/human_pairwise.csv.")
    parser.add_argument("--detail-csv", default=None, help="Defaults to {output_dir}/pairwise_accuracy_detail.csv.")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace):
    out_dir = Path(args.output_dir).resolve() if args.output_dir else None
    scores_csv = Path(args.scores_csv).resolve() if args.scores_csv else None
    human_csv = Path(args.human_csv).resolve() if args.human_csv else None
    detail_csv = Path(args.detail_csv).resolve() if args.detail_csv else None

    if scores_csv is None:
        if out_dir is None:
            raise ValueError("provide --scores-csv or --output-dir")
        scores_csv = out_dir / "qwen_pairwise_scores.csv"
    if human_csv is None:
        if out_dir is None:
            raise ValueError("provide --human-csv or --output-dir")
        human_csv = out_dir / "human_pairwise.csv"
    if detail_csv is None:
        base = out_dir if out_dir is not None else scores_csv.parent
        detail_csv = base / "pairwise_accuracy_detail.csv"

    if not scores_csv.exists():
        raise FileNotFoundError(f"Qwen pairwise CSV not found: {scores_csv}")
    if not human_csv.exists():
        raise FileNotFoundError(f"human pairwise sheet not found: {human_csv}")
    return scores_csv, human_csv, detail_csv


def _norm_winner(value) -> Optional[str]:
    pd = ensure_pandas()
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().upper()
    if text in {"A", "B"}:
        return text
    return None


def main() -> None:
    args = parse_args()
    pd = ensure_pandas()
    scores_csv, human_csv, detail_csv = resolve_paths(args)

    scores = pd.read_csv(scores_csv)
    human = pd.read_csv(human_csv)
    for df, name, col in (
        (scores, scores_csv, "qwen_winner"),
        (human, human_csv, "human_winner"),
    ):
        for required in ("prompt_id", col):
            if required not in df.columns:
                raise ValueError(f"{name} missing required column '{required}'")

    score_cols = ["prompt_id", "qwen_winner"]
    for c in ("video_a_id", "video_b_id"):
        if c in scores.columns:
            score_cols.append(c)
    human_cols = ["prompt_id", "human_winner"]
    for c in ("video_a_id", "video_b_id", "prompt"):
        if c in human.columns:
            human_cols.append(c)

    merged = human[human_cols].merge(
        scores[score_cols], on="prompt_id", how="inner", suffixes=("_human", "_qwen")
    )
    if merged.empty:
        raise ValueError("no overlapping prompt_id between the human sheet and Qwen scores")

    # Detect pair-set drift: the same prompt_id must reference the same A/B videos.
    drift = []
    for col in ("video_a_id", "video_b_id"):
        hc, qc = f"{col}_human", f"{col}_qwen"
        if hc in merged.columns and qc in merged.columns:
            mism = merged[merged[hc].astype(str) != merged[qc].astype(str)]
            for _, r in mism.iterrows():
                drift.append((int(r["prompt_id"]), col, r[hc], r[qc]))

    detail_rows = []
    n_eval = 0
    n_agree = 0
    for _, r in merged.iterrows():
        qw = _norm_winner(r["qwen_winner"])
        hw = _norm_winner(r["human_winner"])
        if hw is None:
            continue  # human has not labelled this pair yet
        n_eval += 1
        agree = (qw is not None) and (qw == hw)
        n_agree += int(agree)
        row = {"prompt_id": int(r["prompt_id"])}
        if "prompt" in merged.columns:
            row["prompt"] = r["prompt"]
        row["qwen_winner"] = qw
        row["human_winner"] = hw
        row["agree"] = bool(agree)
        detail_rows.append(row)

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(detail_csv, index=False)

    accuracy = (n_agree / n_eval) if n_eval else float("nan")
    print(f"aligned pairs: {len(merged)}")
    print(f"human-labelled pairs evaluated: {n_eval}")
    acc_str = "n/a" if accuracy != accuracy else f"{accuracy * 100:.1f}%"
    print(f"pairwise agreement (qwen_winner == human_winner): {acc_str} ({n_agree}/{n_eval})")
    if drift:
        print("")
        print(f"[WARNING] {len(drift)} A/B id mismatch(es) between the two files (pair set drifted):")
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
