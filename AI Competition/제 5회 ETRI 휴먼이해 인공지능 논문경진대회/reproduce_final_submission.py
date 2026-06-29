#!/usr/bin/env python3
"""Reproduce the final selected DACON submission.

The cleaned package keeps one local final artifact:
submissions/final/final_submission.csv.  By default this script validates that
artifact against the official sample submission layout, copies it to the
requested output path, and writes a compact reproduction report.

If the historical base/stage2 files are restored, ``--mode mean-lock`` can
rebuild the final S3 adjustment that produced the selected artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
EPS = 1e-6

DEFAULT_FINAL_ARTIFACT = "submissions/final/final_submission.csv"
DEFAULT_BASE = "submissions/final/base_public_best_before_final.csv"
DEFAULT_STAGE2 = "submissions/final/stage2_source.csv"
DEFAULT_REPORT = "reports/final_reproduction_report.json"

FINAL_PUBLIC_SCORE = 0.5726881984
FINAL_PUBLIC_RANK = 67
FINAL_PRIVATE_SCORE = 0.61533
FINAL_PRIVATE_RANK = 324


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the final selected DACON submission.csv"
    )
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--report-out", default=DEFAULT_REPORT)
    parser.add_argument(
        "--mode",
        choices=["artifact", "mean-lock"],
        default="artifact",
        help=(
            "artifact validates/copies the retained final CSV. mean-lock rebuilds "
            "the historical S3 adjustment from base and stage2 files."
        ),
    )
    parser.add_argument("--final-artifact", default=DEFAULT_FINAL_ARTIFACT)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--stage2-source", default=DEFAULT_STAGE2)
    parser.add_argument("--reference", default=DEFAULT_FINAL_ARTIFACT)
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


def frame_summary(frame: pd.DataFrame) -> Dict[str, Any]:
    probs = frame[TARGETS].to_numpy(dtype=float)
    return {
        "shape": list(frame.shape),
        "target_means": {
            col: float(frame[col].to_numpy(dtype=float).mean()) for col in TARGETS
        },
        "probability_range": {
            "min": float(probs.min()),
            "max": float(probs.max()),
        },
    }


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


def byte_equal(path_a: Path, path_b: Path) -> bool:
    return path_a.read_bytes() == path_b.read_bytes()


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
        "byte_equal": byte_equal(output_path, reference_path),
        "max_abs_target_diff": float(value_diff.max()),
        "mean_abs_target_diff": float(value_diff.mean()),
    }


def base_report(args: argparse.Namespace, sample_path: Path, output_path: Path) -> Dict[str, Any]:
    return {
        "purpose": "reproduce_final_submission",
        "mode": args.mode,
        "final_public_score": FINAL_PUBLIC_SCORE,
        "final_public_rank": FINAL_PUBLIC_RANK,
        "final_private_score": FINAL_PRIVATE_SCORE,
        "final_private_rank": FINAL_PRIVATE_RANK,
        "data_dir": str(Path(args.data_dir)),
        "sample": str(sample_path),
        "output": str(output_path),
    }


def reproduce_from_artifact(
    args: argparse.Namespace, sample: pd.DataFrame, sample_path: Path
) -> Dict[str, Any]:
    artifact_path = Path(args.final_artifact)
    output_path = Path(args.out)
    report_path = Path(args.report_out)

    artifact = read_submission(artifact_path)
    validate_layout(artifact, sample, "final_artifact")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.resolve() != artifact_path.resolve():
        shutil.copyfile(artifact_path, output_path)

    report = base_report(args, sample_path, output_path)
    report.update(
        {
            "final_artifact": str(artifact_path),
            "artifact_byte_equal": byte_equal(output_path, artifact_path),
            **frame_summary(artifact),
        }
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def reproduce_from_mean_lock(
    args: argparse.Namespace, sample: pd.DataFrame, sample_path: Path
) -> Dict[str, Any]:
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

    report = base_report(args, sample_path, output_path)
    report.update(
        {
            "base": str(base_path),
            "stage2_source": str(stage2_path),
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
            **frame_summary(candidate),
            "distance_vs_base": distance_summary(candidate, base, target),
        }
    )
    report["reference_check"] = reference_summary(
        output_path, candidate, reference_path, sample
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    sample_path = data_dir / "ch2026_submission_sample.csv"
    sample = read_submission(sample_path)

    if args.mode == "artifact":
        report = reproduce_from_artifact(args, sample, sample_path)
        print(f"Wrote {args.out}")
        print(f"Wrote {args.report_out}")
        print(f"artifact_byte_equal: {report['artifact_byte_equal']}")
        return

    report = reproduce_from_mean_lock(args, sample, sample_path)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.report_out}")
    if report["reference_check"].get("exists"):
        print(f"reference_byte_equal: {report['reference_check']['byte_equal']}")
        print(
            "reference_max_abs_target_diff: "
            f"{report['reference_check']['max_abs_target_diff']:.12g}"
        )


if __name__ == "__main__":
    main()
