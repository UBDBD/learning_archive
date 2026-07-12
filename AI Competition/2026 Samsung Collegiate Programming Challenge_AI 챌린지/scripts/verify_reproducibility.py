from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from make_submission import build_payload, validate_payload, write_submission_csv
from run_dev import load_baseline_scorer, load_json, load_jsonl, run_harness


FINAL_ARCHIVE = ROOT / "artifacts" / "submissions" / "submission_2026-07-12_final_one_chance.csv"
WORKING_SUBMISSION = ROOT / "submission.csv"
EXPECTED_DEV_OVERALL = 0.9446


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_bytes(tasks: list[dict[str, object]]) -> bytes:
    payload = build_payload(tasks)
    validate_payload(payload, {str(task["id"]) for task in tasks})
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "submission.csv"
        write_submission_csv(payload, output)
        return output.read_bytes()


def check_submission_bytes(expected: bytes, path: Path) -> None:
    actual = path.read_bytes()
    if actual != expected:
        raise AssertionError(f"generated submission differs from {path.relative_to(ROOT)}")


def score_dev() -> dict[str, object]:
    tasks = load_jsonl(ROOT / "data" / "dev_tasks.jsonl")
    answers = load_json(ROOT / "data" / "dev_answers.json")
    payload = run_harness(tasks)
    scorer = load_baseline_scorer()
    return scorer["score_dev_submission"](payload, answers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify deterministic SCPC submission reproduction")
    parser.add_argument("--skip-dev", action="store_true", help="Skip the local dev-score regression check")
    args = parser.parse_args()

    tasks = load_jsonl(ROOT / "data" / "screening_tasks.jsonl")
    first = generate_bytes(tasks)
    second = generate_bytes(tasks)
    if first != second:
        raise AssertionError("two independent submission generations differ")

    check_submission_bytes(first, WORKING_SUBMISSION)
    check_submission_bytes(first, FINAL_ARCHIVE)
    print(f"submission_sha256={sha256(WORKING_SUBMISSION)}")
    print("screening_tasks=700")
    print("deterministic_generation=ok")
    print("working_submission_matches_archive=ok")

    if not args.skip_dev:
        report = score_dev()
        if report.get("overall") != EXPECTED_DEV_OVERALL:
            raise AssertionError(
                f"dev score changed: expected {EXPECTED_DEV_OVERALL}, got {report.get('overall')}"
            )
        print("dev_overall=0.9446")
        print("dev_regression=ok")


if __name__ == "__main__":
    main()
