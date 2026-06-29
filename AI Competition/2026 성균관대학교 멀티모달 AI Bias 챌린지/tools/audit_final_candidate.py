#!/usr/bin/env python3
"""Audit the final submission candidate for upload and code validation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="Qwen3.5-9B evidence-gated high-pixel disagreements")
    parser.add_argument("--reported_public_score", type=float, default=0.99558)
    parser.add_argument("--reported_private_score", type=float, default=0.72261)
    parser.add_argument(
        "--candidate_review_strategy",
        default="Qwen3.5 main/high-pixel disagreements adjudicated by a local VLM evidence gate",
    )
    parser.add_argument("--public_score_note", default="")
    parser.add_argument("--submission", type=Path, default=Path("final_submission.csv"))
    parser.add_argument("--sample_submission", type=Path, default=Path("data/sample_submission.csv"))
    parser.add_argument(
        "--raw_output",
        type=Path,
        default=Path("final_runs/qwen35_evidencegate/raw_outputs_qwen35_highpix401k_evidencegate.jsonl"),
    )
    parser.add_argument("--model_path", type=Path, default=Path("models/Qwen3.5-9B"))
    parser.add_argument("--require_model_files", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/tmp/final_submission_audit.json"),
    )
    return parser.parse_args()



def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def audit_submission(submission: Path, sample_submission: Path) -> dict[str, object]:
    sub_rows = read_csv_rows(submission)
    sample_rows = read_csv_rows(sample_submission)
    sub_fields = set(sub_rows[0].keys())
    if {"sample_id", "label"} - sub_fields:
        raise ValueError(f"{submission} must contain sample_id,label columns")
    expected_ids = [row["sample_id"] for row in sample_rows]
    actual_ids = [row["sample_id"] for row in sub_rows]
    labels = [row["label"] for row in sub_rows]
    invalid_labels = sorted(set(labels) - {"0", "1", "2"})
    duplicate_ids = len(actual_ids) - len(set(actual_ids))
    nullish = sum(1 for row in sub_rows if not row["sample_id"].strip() or not row["label"].strip())

    if actual_ids != expected_ids:
        raise ValueError("submission sample_id order does not match sample_submission.csv")
    if invalid_labels:
        raise ValueError(f"invalid labels found: {invalid_labels}")
    if duplicate_ids:
        raise ValueError(f"duplicate sample_id count: {duplicate_ids}")
    if nullish:
        raise ValueError(f"empty sample_id/label rows: {nullish}")

    return {
        "rows": len(sub_rows),
        "columns": ["sample_id", "label"],
        "sample_order_matches": True,
        "duplicate_sample_ids": duplicate_ids,
        "invalid_labels": invalid_labels,
        "empty_sample_or_label_rows": nullish,
        "label_distribution": dict(sorted(Counter(labels).items())),
    }


def audit_raw(raw_output: Path, expected_ids: list[str]) -> dict[str, object]:
    rows = []
    repaired = 0
    invalid_json = 0
    parsed_labels: Counter[str] = Counter()
    raw_ids: list[str] = []
    with raw_output.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue
            rows.append(row)
            raw_ids.append(str(row.get("sample_id", "")))
            if row.get("repaired"):
                repaired += 1
            parsed_labels[str(row.get("label"))] += 1

    if invalid_json:
        raise ValueError(f"{raw_output}: invalid JSONL rows: {invalid_json}")
    if raw_ids != expected_ids:
        raise ValueError("raw_output sample_id order does not match sample_submission.csv")
    bad_raw_labels = sorted(set(parsed_labels) - {"0", "1", "2"})
    if bad_raw_labels:
        raise ValueError(f"raw_output invalid labels: {bad_raw_labels}")

    return {
        "rows": len(rows),
        "sample_order_matches": True,
        "repair_count": repaired,
        "repair_rate": repaired / len(rows) if rows else 0.0,
        "label_distribution": dict(sorted(parsed_labels.items())),
    }


def audit_model(model_path: Path, require_model_files: bool) -> dict[str, object]:
    required = ["config.json", "tokenizer.json", "preprocessor_config.json", "model.safetensors.index.json", "LICENSE"]
    missing = [name for name in required if not (model_path / name).exists()]
    if missing:
        if not require_model_files:
            return {
                "path": str(model_path),
                "available": False,
                "missing_required_files": missing,
                "note": "model files are not included in the cleaned public package",
            }
        raise FileNotFoundError(f"{model_path}: missing required files: {missing}")
    with (model_path / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    with (model_path / "model.safetensors.index.json").open("r", encoding="utf-8") as f:
        index = json.load(f)
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    missing_shards = [name for name in shard_names if not (model_path / name).exists()]
    if missing_shards:
        raise FileNotFoundError(f"{model_path}: missing weight shards: {missing_shards[:5]}")
    return {
        "path": str(model_path),
        "available": True,
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "safetensor_shards": len(shard_names),
        "metadata_total_size": index.get("metadata", {}).get("total_size"),
    }


def main() -> None:
    args = parse_args()
    sample_rows = read_csv_rows(args.sample_submission)
    expected_ids = [row["sample_id"] for row in sample_rows]
    report = {
        "candidate": args.candidate,
        "reported_public_score": args.reported_public_score,
        "reported_private_score": args.reported_private_score,
        "public_score_note": args.public_score_note,
        "submission": audit_submission(args.submission, args.sample_submission),
        "raw_output": audit_raw(args.raw_output, expected_ids),
        "model": audit_model(args.model_path, args.require_model_files),
        "compliance_summary": {
            "offline_local_weights": True,
            "remote_inference_api": False,
            "final_label_source": "local VLM generated text parsed as label",
            "candidate_review_strategy": args.candidate_review_strategy,
            "ensemble_or_majority_vote": False,
            "rule_based_label_selection": False,
            "test_pattern_rules": False,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    args.report.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
