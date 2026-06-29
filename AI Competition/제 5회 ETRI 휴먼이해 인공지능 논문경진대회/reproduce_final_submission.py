#!/usr/bin/env python3
"""Validate and reproduce the final selected DACON ETRI submission.

The cleaned project keeps only the result/reproducibility path.  It validates
the retained final CSV against the official sample submission layout, copies it
to the requested output path, and writes a compact report.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict

import pandas as pd


TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

DEFAULT_FINAL_ARTIFACT = "submissions/final/final_submission.csv"
DEFAULT_REPORT = "reports/final_reproduction_report.json"

FINAL_PUBLIC_SCORE = 0.5726881984
FINAL_PUBLIC_RANK = 67
FINAL_PRIVATE_SCORE = 0.61533
FINAL_PRIVATE_RANK = 324


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and reproduce the final selected DACON submission.csv"
    )
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--report-out", default=DEFAULT_REPORT)
    parser.add_argument("--final-artifact", default=DEFAULT_FINAL_ARTIFACT)
    return parser.parse_args()


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

    probs = frame[TARGETS]
    if not probs.notna().all().all():
        raise ValueError(f"{name}: missing probability found")
    if (probs < 0.0).any().any() or (probs > 1.0).any().any():
        raise ValueError(f"{name}: probability outside [0, 1]")


def frame_summary(frame: pd.DataFrame) -> Dict[str, Any]:
    probs = frame[TARGETS]
    return {
        "shape": list(frame.shape),
        "target_means": {col: float(frame[col].mean()) for col in TARGETS},
        "probability_range": {
            "min": float(probs.min().min()),
            "max": float(probs.max().max()),
        },
    }


def byte_equal(path_a: Path, path_b: Path) -> bool:
    return path_a.read_bytes() == path_b.read_bytes()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    sample_path = data_dir / "ch2026_submission_sample.csv"
    artifact_path = Path(args.final_artifact)
    output_path = Path(args.out)
    report_path = Path(args.report_out)

    sample = read_submission(sample_path)
    artifact = read_submission(artifact_path)
    validate_layout(artifact, sample, "final_artifact")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.resolve() != artifact_path.resolve():
        shutil.copyfile(artifact_path, output_path)

    report: Dict[str, Any] = {
        "purpose": "reproduce_final_submission",
        "final_public_score": FINAL_PUBLIC_SCORE,
        "final_public_rank": FINAL_PUBLIC_RANK,
        "final_private_score": FINAL_PRIVATE_SCORE,
        "final_private_rank": FINAL_PRIVATE_RANK,
        "data_dir": str(data_dir),
        "sample": str(sample_path),
        "final_artifact": str(artifact_path),
        "output": str(output_path),
        "artifact_byte_equal": byte_equal(output_path, artifact_path),
        **frame_summary(artifact),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {output_path}")
    print(f"Wrote {report_path}")
    print(f"artifact_byte_equal: {report['artifact_byte_equal']}")


if __name__ == "__main__":
    main()
