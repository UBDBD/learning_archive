from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from harness import FIXED_SLM_ID, SUBMISSION_SCHEMA, VALID_CONTROLS, VALID_SCOPE_MODES, FinalHarness
from run_dev import load_jsonl, participant_task_view


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def answer_one(harness: Any, task: dict[str, Any], session: dict[str, Any]) -> dict[str, Any]:
    answer = harness.answer_task(participant_task_view(task), session)
    if not isinstance(answer, dict):
        raise TypeError(f"answer_task returned non-object for {task.get('id')}")
    return answer


def build_payload(tasks: list[dict[str, Any]], *, enable_explicit_redaction_override: bool = True) -> dict[str, Any]:
    ordered = sorted(tasks, key=lambda t: (str(t.get("session_id", "")), int(t.get("turn_index", 0)), str(t.get("id", ""))))
    harness = FinalHarness(enable_explicit_redaction_override=enable_explicit_redaction_override)
    harness.prepare([])
    sessions: dict[str, dict[str, Any]] = {}
    answers: dict[str, dict[str, Any]] = {}
    for task in ordered:
        sid = str(task.get("session_id", ""))
        answers[str(task["id"])] = answer_one(harness, task, sessions.setdefault(sid, {}))
    return {
        "schema": SUBMISSION_SCHEMA,
        "meta": {
            "harness_name": "strategy_harness_v1",
            "uses_external_api": False,
            "fixed_slm_policy": "local_fixed_slm_only",
            "model_id": FIXED_SLM_ID,
            "temperature": 0.0,
            "seed": 42,
        },
        "answers": answers,
    }


def validate_payload(payload: dict[str, Any], expected_ids: set[str]) -> None:
    if payload.get("schema") != SUBMISSION_SCHEMA:
        raise ValueError("invalid schema")
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("missing meta")
    if meta.get("uses_external_api") is not False:
        raise ValueError("meta.uses_external_api must be false")
    if meta.get("fixed_slm_policy") != "local_fixed_slm_only":
        raise ValueError("invalid fixed_slm_policy")
    if meta.get("model_id") != FIXED_SLM_ID:
        raise ValueError("invalid model_id")
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("answers must be an object")
    missing = expected_ids - set(answers)
    extra = set(answers) - expected_ids
    if missing:
        raise ValueError(f"missing answers: {sorted(missing)[:5]} total={len(missing)}")
    if extra:
        raise ValueError(f"extra answers: {sorted(extra)[:5]} total={len(extra)}")
    for task_id, answer in answers.items():
        for field in ["focal_id", "target", "control", "content_scope", "policy", "plan_events"]:
            if field not in answer:
                raise ValueError(f"{task_id} missing {field}")
        if answer["control"] not in VALID_CONTROLS:
            raise ValueError(f"{task_id} invalid control")
        scope = answer["content_scope"]
        if not isinstance(scope, dict) or scope.get("mode") not in VALID_SCOPE_MODES:
            raise ValueError(f"{task_id} invalid scope")
        for field in ["allowed_fields", "excluded_fields"]:
            if not isinstance(scope.get(field), list):
                raise ValueError(f"{task_id} scope.{field} must be list")
        if not isinstance(scope.get("requires_user_confirmation"), bool):
            raise ValueError(f"{task_id} invalid confirmation flag")
        policy = answer["policy"]
        if not isinstance(policy, dict):
            raise ValueError(f"{task_id} invalid policy")
        if not isinstance(policy.get("risk_flags"), list) or not isinstance(policy.get("violations"), list):
            raise ValueError(f"{task_id} invalid policy arrays")
        if not isinstance(policy.get("requires_confirmation"), bool):
            raise ValueError(f"{task_id} invalid policy confirmation")
        events = answer["plan_events"]
        if not isinstance(events, list) or len(events) > 18:
            raise ValueError(f"{task_id} invalid plan_events")
        for event in events:
            if not isinstance(event, dict) or not {"verb", "target", "args"} <= set(event):
                raise ValueError(f"{task_id} invalid plan event")
            if not isinstance(event["args"], dict):
                raise ValueError(f"{task_id} event args must be object")


def write_submission_csv(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["submission"])
        writer.writerow([json.dumps(payload, ensure_ascii=False, separators=(",", ":"))])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an SCPC submission CSV")
    parser.add_argument(
        "--profile",
        choices=("conservative", "aggressive_redaction"),
        default="aggressive_redaction",
        help="Use only validated rules or also enable the explicit-redaction override",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "submission.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_jsonl(DATA_DIR / "screening_tasks.jsonl")
    payload = build_payload(
        tasks,
        enable_explicit_redaction_override=args.profile == "aggressive_redaction",
    )
    validate_payload(payload, {str(task["id"]) for task in tasks})
    out_path = args.output if args.output.is_absolute() else ROOT / args.output
    write_submission_csv(payload, out_path)
    print(f"wrote: {out_path}")
    print(f"answers: {len(payload['answers'])}")
    print(f"profile: {args.profile}")
    print(json.dumps(payload["meta"], ensure_ascii=False))


if __name__ == "__main__":
    main()
