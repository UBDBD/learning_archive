#!/usr/bin/env python3
"""Reproduce the final selected DACON submission.

This script intentionally reproduces the final leaderboard-selected artifact,
not every exploratory public probe.  The final file is built from the previous
public-best submission by replacing only S3 with a mean-locked move toward the
stage2 S3 prediction.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
EPS = 1e-6

DEFAULT_BASE = "submissions/final/base_public_best_before_final.csv"
DEFAULT_STAGE2 = "submissions/final/stage2_source.csv"
DEFAULT_REFERENCE = "submissions/final/final_selected.csv"
DEFAULT_REPORT = "reports/final_reproduction_report.json"
FINAL_PUBLIC_SCORE = 0.5726881984
FINAL_PUBLIC_RANK = 67


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the final selected DACON submission.csv"
    )
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--report-out", default=DEFAULT_REPORT)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--stage2-source", default=DEFAULT_STAGE2)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--target", choices=TARGETS, default="S3")
    parser.add_argument("--weight", type=float, default=0.70)
    return parser.parse_args()


def clip_prob(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)


def logit_prob(values: np.ndarray) -> np.ndarray:
    values = clip_prob(values)
    return np.log(values / (1.0 - values))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def mean_lock_logit(values: np.ndarray, target_mean: float) -> Tuple[np.ndarray, float]:
    """Shift logits so the transformed probabilities keep a target mean."""

    values = clip_prob(values)
    target_mean = float(target_mean)
    if not EPS <= target_mean <= 1.0 - EPS:
        raise ValueError(f"target_mean must be inside ({EPS}, {1.0 - EPS})")

    logits = logit_prob(values)
    lo, hi = -50.0, 50.0
    for _ in range(120):
        mid = (lo + hi) / 2.0
        mid_mean = float(sigmoid(logits + mid).mean())
        if mid_mean < target_mean:
            lo = mid
        else:
            hi = mid

    shift = (lo + hi) / 2.0
    locked = clip_prob(sigmoid(logits + shift))
    return locked, shift



def read_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required submission file: {path}")
    return pd.read_csv(path)


def validate_layout(frame: pd.DataFrame, sample: pd.DataFrame, name: str) -> None:
    if list(frame.columns) != list(sample.columns):
        raise ValueError(f"{name}: columns do not match sample submission")
    if len(frame) != len(sample):
        raise ValueError(f"{name}: row count does not match sample submission")

    id_cols = [col for col in sample.columns if col not in TARGETS]
    if not frame[id_cols].equals(sample[id_cols]):
        raise ValueError(f"{name}: subject/date row order does not match sample")

    probs = frame[TARGETS].to_numpy(dtype=float)
    if not np.isfinite(probs).all():
        raise ValueError(f"{name}: non-finite probability found")
    if probs.min() < 0.0 or probs.max() > 1.0:
        raise ValueError(f"{name}: probability outside [0, 1]")


def distance_summary(candidate: pd.DataFrame, base: pd.DataFrame, target: str) -> Dict[str, Any]:
    target_diff = np.abs(
        candidate[target].to_numpy(dtype=float) - base[target].to_numpy(dtype=float)
    )
    all_diff = np.abs(
        candidate[TARGETS].to_numpy(dtype=float) - base[TARGETS].to_numpy(dtype=float)
    )
    return {
        f"{target.lower()}_mean_abs": float(target_diff.mean()),
        f"{target.lower()}_max_abs": float(target_diff.max()),
        f"{target.lower()}_pct_abs_gt_0p01": float((target_diff > 0.01).mean()),
        "changed_targets": [
            col
            for col in TARGETS
            if float(
                np.abs(
                    candidate[col].to_numpy(dtype=float)
                    - base[col].to_numpy(dtype=float)
                ).max()
            )
            > 1e-12
        ],
        "mean_abs_all": float(all_diff.mean()),
        "max_abs_all": float(all_diff.max()),
        "pct_abs_diff_gt_0p01_all": float((all_diff > 0.01).mean()),
        "pct_abs_diff_gt_0p03_all": float((all_diff > 0.03).mean()),
    }


def reference_summary(
    output_path: Path, candidate: pd.DataFrame, reference_path: Path, sample: pd.DataFrame
) -> Dict[str, Any]:
    if not reference_path.exists():
        return {"reference_path": str(reference_path), "exists": False}

    reference = read_submission(reference_path)
    validate_layout(reference, sample, "reference")
    value_diff = np.abs(
        candidate[TARGETS].to_numpy(dtype=float)
        - reference[TARGETS].to_numpy(dtype=float)
    )
    return {
        "reference_path": str(reference_path),
        "exists": True,
        "byte_equal": output_path.read_bytes() == reference_path.read_bytes(),
        "max_abs_target_diff": float(value_diff.max()),
        "mean_abs_target_diff": float(value_diff.mean()),
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    sample_path = data_dir / "ch2026_submission_sample.csv"
    sample = read_submission(sample_path)

    base_path = Path(args.base)
    stage2_path = Path(args.stage2_source)
    reference_path = Path(args.reference)
    output_path = Path(args.out)
    report_path = Path(args.report_out)

    base = read_submission(base_path)
    stage2 = read_submission(stage2_path)
    validate_layout(base, sample, "base")
    validate_layout(stage2, sample, "stage2_source")

    target = args.target
    weight = float(args.weight)
    base_target = clip_prob(base[target].to_numpy(dtype=float))
    stage2_target = clip_prob(stage2[target].to_numpy(dtype=float))

    raw_target = clip_prob(base_target + weight * (stage2_target - base_target))
    locked_target, shift = mean_lock_logit(raw_target, float(base_target.mean()))

    candidate = base.copy()
    candidate[target] = locked_target
    validate_layout(candidate, sample, "candidate")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(output_path, index=False)
    report: Dict[str, Any] = {
        "purpose": "reproduce_final_selected_submission",
        "final_public_score": FINAL_PUBLIC_SCORE,
        "final_public_rank": FINAL_PUBLIC_RANK,
        "data_dir": str(data_dir),
        "sample": str(sample_path),
        "base": str(base_path),
        "stage2_source": str(stage2_path),
        "output": str(output_path),
        "target": target,
        "weight": weight,
        "formula": (
            f"{target}_raw = clip(base_{target} + weight * "
            f"(stage2_{target} - base_{target})); "
            f"{target}_final = sigmoid(logit({target}_raw) + intercept)"
        ),
        "mean_lock": {
            "base_mean": float(base_target.mean()),
            "stage2_source_mean": float(stage2_target.mean()),
            "raw_toward_stage2_mean": float(raw_target.mean()),
            "locked_mean": float(locked_target.mean()),
            "logit_intercept_shift": float(shift),
        },
        "target_means": {
            col: float(candidate[col].to_numpy(dtype=float).mean()) for col in TARGETS
        },
        "distance_vs_base": distance_summary(candidate, base, target),
        "probability_range": {
            "min": float(candidate[TARGETS].to_numpy(dtype=float).min()),
            "max": float(candidate[TARGETS].to_numpy(dtype=float).max()),
        },
    }
    report["reference_check"] = reference_summary(
        output_path, candidate, reference_path, sample
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {output_path}")
    print(f"Wrote {report_path}")
    if report["reference_check"].get("exists"):
        print(f"reference_byte_equal: {report['reference_check']['byte_equal']}")
        print(
            "reference_max_abs_target_diff: "
            f"{report['reference_check']['max_abs_target_diff']:.12g}"
        )


if __name__ == "__main__":
    main()
