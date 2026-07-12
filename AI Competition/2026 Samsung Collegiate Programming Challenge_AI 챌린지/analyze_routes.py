from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

ROUTE_TYPES = (
    "route_candidate_snapshot",
    "dispatch_authority_check",
    "share_boundary_update",
    "route_binding_order",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_submission(path: Path) -> dict[str, Any]:
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one data row")
    return json.loads(rows[0]["submission"])


def record_map(task: dict[str, Any]) -> dict[str, Any]:
    return {
        str(record.get("type")): record.get("value")
        for record in task.get("device_state", {}).get("records", [])
    }


def route_key(task: dict[str, Any]) -> tuple[str, ...]:
    records = record_map(task)
    return tuple(str(records.get(record_type, "-")) for record_type in ROUTE_TYPES)


def final_directive(prompt: str) -> str:
    if "단," in prompt:
        return prompt.rsplit("단,", 1)[-1].strip()
    return "(no explicit final directive)"


def summarize_dev() -> None:
    tasks = load_jsonl(DATA_DIR / "dev_tasks.jsonl")
    answers = json.loads((DATA_DIR / "dev_answers.json").read_text(encoding="utf-8"))["answers"]
    groups: dict[tuple[str, ...], Counter[tuple[str, str, str]]] = defaultdict(Counter)
    directives: dict[str, Counter[tuple[str, str, str]]] = defaultdict(Counter)

    for task in tasks:
        answer = answers[str(task["id"])]
        outcome = (answer["target"], answer["control"], answer["content_scope"]["mode"])
        groups[route_key(task)][outcome] += 1
        directives[final_directive(str(task.get("prompt", "")))][outcome] += 1

    print("DEV ROUTE COMBINATIONS")
    for key, outcomes in sorted(groups.items(), key=lambda item: (-sum(item[1].values()), item[0])):
        if key == ("-", "-", "-", "-"):
            continue
        print(sum(outcomes.values()), dict(zip(ROUTE_TYPES, key)), dict(outcomes))

    print("\nDEV FINAL DIRECTIVES")
    for directive, outcomes in sorted(directives.items(), key=lambda item: (-sum(item[1].values()), item[0])):
        print(sum(outcomes.values()), directive, dict(outcomes))


def summarize_screening(submission_path: Path) -> None:
    tasks = load_jsonl(DATA_DIR / "screening_tasks.jsonl")
    answers = load_submission(submission_path)["answers"]
    route_groups: dict[tuple[str, ...], Counter[tuple[str, str, str]]] = defaultdict(Counter)
    directive_groups: dict[str, Counter[tuple[str, str, str]]] = defaultdict(Counter)
    route_directives: dict[tuple[tuple[str, ...], str], Counter[tuple[str, str, str]]] = defaultdict(Counter)

    for task in tasks:
        answer = answers[str(task["id"])]
        outcome = (answer["target"], answer["control"], answer["content_scope"]["mode"])
        key = route_key(task)
        directive = final_directive(str(task.get("prompt", "")))
        route_groups[key][outcome] += 1
        directive_groups[directive][outcome] += 1
        route_directives[(key, directive)][outcome] += 1

    print(f"\nSCREENING ROUTE COMBINATIONS ({submission_path.name})")
    for key, outcomes in sorted(route_groups.items(), key=lambda item: (-sum(item[1].values()), item[0])):
        if key == ("-", "-", "-", "-"):
            continue
        print(sum(outcomes.values()), dict(zip(ROUTE_TYPES, key)), dict(outcomes))

    print(f"\nSCREENING FINAL DIRECTIVES ({submission_path.name})")
    for directive, outcomes in sorted(directive_groups.items(), key=lambda item: (-sum(item[1].values()), item[0])):
        print(sum(outcomes.values()), directive, dict(outcomes))

    print(f"\nSCREENING ROUTE + DIRECTIVE ({submission_path.name})")
    for (key, directive), outcomes in sorted(
        route_directives.items(), key=lambda item: (-sum(item[1].values()), item[0][0], item[0][1])
    ):
        if key == ("-", "-", "-", "-"):
            continue
        print(sum(outcomes.values()), dict(zip(ROUTE_TYPES, key)), directive, dict(outcomes))


def main() -> None:
    summarize_dev()
    summarize_screening(ROOT / "artifacts" / "submissions" / "submission_public_0_7546.csv")


if __name__ == "__main__":
    main()
