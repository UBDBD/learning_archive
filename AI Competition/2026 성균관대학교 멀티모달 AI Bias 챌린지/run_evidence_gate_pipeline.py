#!/usr/bin/env python3
"""Reproduce or validate the Qwen3.5 evidence-gated final submission."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Qwen3.5 main/high-pixel/evidence-gate pipeline."
    )
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--model_path", type=Path, default=Path("models/Qwen3.5-9B"))
    parser.add_argument("--work_dir", type=Path, default=Path("final_runs/qwen35_evidencegate"))
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--repair_max_new_tokens", type=int, default=32)
    parser.add_argument("--judge_max_new_tokens", type=int, default=64)
    parser.add_argument("--judge_repair_max_new_tokens", type=int, default=32)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--fail_on_cpu_offload", action="store_true")
    parser.add_argument(
        "--attn_implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--image_max_side", type=int, default=0)
    parser.add_argument("--main_image_max_pixels", type=int, default=200704)
    parser.add_argument("--high_image_max_pixels", type=int, default=401408)
    parser.add_argument("--image_min_pixels", type=int, default=50176)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--judge_log_every", type=int, default=100)
    parser.add_argument("--expected_test_rows", type=int, default=8500)
    parser.add_argument("--resume", action="store_true", help="Skip any step whose outputs already exist.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing outputs.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running inference.")
    parser.add_argument("--output", type=Path, default=Path("final_submission.csv"))
    args = parser.parse_args()

    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1")
    for name in ("max_new_tokens", "repair_max_new_tokens", "judge_max_new_tokens", "judge_repair_max_new_tokens"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be >= 1")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.main_output = args.work_dir / "main_resolution_submission.csv"
    args.main_raw_output = args.work_dir / "raw_main_resolution.jsonl"
    args.high_output = args.work_dir / "high_pixel_submission.csv"
    args.high_raw_output = args.work_dir / "raw_high_pixel.jsonl"
    args.raw_output = args.work_dir / "raw_outputs_qwen35_highpix401k_evidencegate.jsonl"
    validating_existing = args.resume and args.output.exists() and args.raw_output.exists()
    if not args.dry_run and not validating_existing and not (args.model_path / "config.json").exists():
        parser.error(f"--model_path must point to a local Hugging Face model directory: {args.model_path}")
    return args


def env_for(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": args.cuda,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return env


def infer_command(args: argparse.Namespace, output: Path, raw_output: Path, image_max_pixels: int) -> list[str]:
    cmd = [
        sys.executable,
        "infer.py",
        "--data_dir",
        str(args.data_dir),
        "--split",
        "test",
        "--expected_test_rows",
        str(args.expected_test_rows),
        "--model_path",
        str(args.model_path),
        "--output",
        str(output),
        "--raw_output",
        str(raw_output),
        "--batch_size",
        str(args.batch_size),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--repair_max_new_tokens",
        str(args.repair_max_new_tokens),
        "--dtype",
        args.dtype,
        "--attn_implementation",
        args.attn_implementation,
        "--image_max_side",
        str(args.image_max_side),
        "--image_max_pixels",
        str(image_max_pixels),
        "--image_min_pixels",
        str(args.image_min_pixels),
        "--disable_thinking",
        "--log_every",
        str(args.log_every),
    ]
    if args.fail_on_cpu_offload:
        cmd.append("--fail_on_cpu_offload")
    return cmd


def judge_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "tools/adjudicate_resolution_disagreements.py",
        "--data_dir",
        str(args.data_dir),
        "--split",
        "test",
        "--expected_test_rows",
        str(args.expected_test_rows),
        "--model_path",
        str(args.model_path),
        "--base_submission",
        str(args.main_output),
        "--candidate_submission",
        str(args.high_output),
        "--output",
        str(args.output),
        "--raw_output",
        str(args.raw_output),
        "--batch_size",
        str(args.batch_size),
        "--max_new_tokens",
        str(args.judge_max_new_tokens),
        "--repair_max_new_tokens",
        str(args.judge_repair_max_new_tokens),
        "--dtype",
        args.dtype,
        "--attn_implementation",
        args.attn_implementation,
        "--image_max_side",
        str(args.image_max_side),
        "--image_max_pixels",
        str(args.high_image_max_pixels),
        "--image_min_pixels",
        str(args.image_min_pixels),
        "--disable_thinking",
        "--log_every",
        str(args.judge_log_every),
    ]
    if args.fail_on_cpu_offload:
        cmd.append("--fail_on_cpu_offload")
    return cmd


def ensure_step_outputs(paths: tuple[Path, Path], overwrite: bool, resume: bool) -> bool:
    if resume and all(path.exists() for path in paths):
        return True
    if resume:
        return False
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing outputs: {names}")
    return False


def run_step(name: str, cmd: list[str], outputs: tuple[Path, Path], args: argparse.Namespace) -> None:
    print(f"\n[{name}]")
    print(" ".join(cmd))
    if args.dry_run:
        return
    if ensure_step_outputs(outputs, args.overwrite, args.resume):
        print(f"[skip] existing outputs: {outputs[0]}, {outputs[1]}")
        return
    subprocess.run(cmd, cwd=Path.cwd(), env=env_for(args), check=True)


def expected_sample_ids(sample_submission: Path) -> list[str]:
    with sample_submission.open("r", encoding="utf-8", newline="") as f:
        return [row["sample_id"] for row in csv.DictReader(f)]


def validate_submission(path: Path, expected_ids: list[str]) -> dict[str, object]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [row["sample_id"] for row in rows]
    labels = [row["label"] for row in rows]
    if ids != expected_ids:
        raise ValueError(f"{path}: sample_id order does not match sample_submission.csv")
    invalid = sorted(set(labels) - {"0", "1", "2"})
    if invalid:
        raise ValueError(f"{path}: invalid labels found: {invalid}")
    if any(not row["sample_id"].strip() or not row["label"].strip() for row in rows):
        raise ValueError(f"{path}: empty sample_id or label")
    return {
        "rows": len(rows),
        "label_distribution": dict(sorted(Counter(labels).items())),
    }


def validate_raw(path: Path, expected_ids: list[str]) -> dict[str, object]:
    rows = []
    ids = []
    labels: Counter[str] = Counter()
    repair_count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows.append(row)
            ids.append(str(row.get("sample_id", "")))
            labels[str(row.get("label"))] += 1
            repair_count += bool(row.get("repaired"))
    if ids != expected_ids:
        raise ValueError(f"{path}: raw sample_id order does not match sample_submission.csv")
    invalid = sorted(set(labels) - {"0", "1", "2"})
    if invalid:
        raise ValueError(f"{path}: raw invalid labels found: {invalid}")
    return {
        "rows": len(rows),
        "repair_count": repair_count,
        "label_distribution": dict(sorted(labels.items())),
    }



def clean_intermediate_outputs(args: argparse.Namespace) -> None:
    for path in (args.main_output, args.high_output, args.main_raw_output, args.high_raw_output):
        if path.exists():
            path.unlink()
            print(f"[clean] removed intermediate output: {path}")


def final_outputs_exist(args: argparse.Namespace) -> bool:
    return args.output.exists() and args.raw_output.exists()


def main() -> None:
    args = parse_args()
    if args.resume and final_outputs_exist(args):
        print("[resume] final outputs already exist; validating final submission only")
        commands = []
    else:
        commands = [
            (
                "main-resolution direct pass",
                infer_command(args, args.main_output, args.main_raw_output, args.main_image_max_pixels),
                (args.main_output, args.main_raw_output),
            ),
            (
                "high-pixel direct pass",
                infer_command(args, args.high_output, args.high_raw_output, args.high_image_max_pixels),
                (args.high_output, args.high_raw_output),
            ),
            (
                "evidence-gate disagreement judge",
                judge_command(args),
                (args.output, args.raw_output),
            ),
        ]
    for name, cmd, outputs in commands:
        run_step(name, cmd, outputs, args)
    if args.dry_run:
        return

    expected_ids = expected_sample_ids(args.data_dir / "sample_submission.csv")
    main_report = validate_submission(args.main_output, expected_ids) if args.main_output.exists() else None
    high_report = validate_submission(args.high_output, expected_ids) if args.high_output.exists() else None
    submission_report = validate_submission(args.output, expected_ids)
    if args.main_raw_output.exists():
        validate_raw(args.main_raw_output, expected_ids)
    if args.high_raw_output.exists():
        validate_raw(args.high_raw_output, expected_ids)
    raw_report = validate_raw(args.raw_output, expected_ids)
    print("\n[final validation]")
    if main_report is not None:
        print(f"Main pass labels: {main_report['label_distribution']}")
    if high_report is not None:
        print(f"High-pixel pass labels: {high_report['label_distribution']}")
    print(f"Rows: {submission_report['rows']}")
    print(f"Label distribution: {submission_report['label_distribution']}")
    print(f"Raw rows: {raw_report['rows']}")
    print(f"Repair count: {raw_report['repair_count']}")
    clean_intermediate_outputs(args)
    print(f"Final evidence-gate submission ready: {args.output}")


if __name__ == "__main__":
    main()
