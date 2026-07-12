from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harness import FinalHarness


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def participant_task_view(task: dict[str, Any]) -> dict[str, Any]:
    view = json.loads(json.dumps(task, ensure_ascii=False))
    for key in list(view):
        if (
            key in {"expected_events", "answer"}
            or key.startswith("expected_")
            or key.endswith("_brief")
            or key.endswith("_notes")
            or key.endswith("_rubric")
            or key.endswith("_keywords")
            or key.endswith("_tags")
        ):
            view.pop(key, None)
    return view


def answer_one(harness: Any, task: dict[str, Any], session: dict[str, Any]) -> dict[str, Any]:
    return harness.answer_task(task, session)


def run_harness(tasks: list[dict[str, Any]], harness_cls: type = FinalHarness, *, harness_name: str = "strategy_harness") -> dict[str, Any]:
    ordered = sorted(tasks, key=lambda t: (str(t.get("session_id", "")), int(t.get("turn_index", 0)), str(t.get("id", ""))))
    harness = harness_cls()
    prepare = getattr(harness, "prepare", None)
    if callable(prepare):
        prepare([])

    sessions: dict[str, dict[str, Any]] = {}
    answers: dict[str, dict[str, Any]] = {}
    for task in ordered:
        sid = str(task.get("session_id", ""))
        session = sessions.setdefault(sid, {})
        answers[str(task["id"])] = answer_one(harness, participant_task_view(task), session)

    return {
        "schema": "scpc.final.answer.v1",
        "meta": {
            "harness_name": harness_name,
            "uses_external_api": False,
            "fixed_slm_policy": "local_fixed_slm_only",
            "model_id": "scpc-final-fixed-slm-local-facade",
            "temperature": 0.0,
            "seed": 42,
        },
        "answers": answers,
    }


def load_baseline_scorer() -> dict[str, Any]:
    nb = load_json(DATA_DIR / "SCPC2026_Final_baseline.ipynb")
    ns: dict[str, Any] = {"FinalHarness": FinalHarness}
    for idx in [1, 3, 9, 11]:
        src = "from __future__ import annotations\n" + "".join(nb["cells"][idx]["source"])
        exec(src, ns)
    return ns


def write_failure_report(payload: dict[str, Any], reference: dict[str, Any], tasks: list[dict[str, Any]], scorer_ns: dict[str, Any]) -> list[dict[str, Any]]:
    by_task = {str(t["id"]): t for t in tasks}
    rows: list[dict[str, Any]] = []
    text = scorer_ns["_text"]
    scope_score = scorer_ns["_scope_score"]
    policy_score = scorer_ns["_policy_score"]
    plan_score = scorer_ns["_plan_score"]
    weights = scorer_ns["WEIGHTS"]

    for task_id, ref in reference["answers"].items():
        pred = payload["answers"][task_id]
        focal = 1.0 if text(pred.get("focal_id")) == text(ref.get("focal_id")) else 0.0
        target = focal * (1.0 if text(pred.get("target")) == text(ref.get("target")) else 0.0)
        control = focal * (1.0 if text(pred.get("control")) == text(ref.get("control")) else 0.0)
        dependent = target * control
        axes = {
            "focal": focal,
            "target": target,
            "control": control,
            "content_scope": dependent * scope_score(pred.get("content_scope"), ref.get("content_scope")),
            "policy": dependent * policy_score(pred.get("policy"), ref.get("policy")),
            "plan": dependent * plan_score(pred.get("plan_events"), ref.get("expected_events")),
            "semantic_response": 0.0,
            "counterfactual": 0.0,
        }
        score = sum(axes[k] * weights[k] for k in weights)
        task = by_task[task_id]
        records = [r.get("type") for r in task.get("device_state", {}).get("records", [])]
        rows.append(
            {
                "task_id": task_id,
                "score": round(score, 4),
                "axes": {k: round(v, 4) for k, v in axes.items()},
                "pred": pred,
                "ref": ref,
                "record_types": records,
                "prompt": task.get("prompt", ""),
                "history": task.get("visible_history", []),
            }
        )

    rows.sort(key=lambda r: r["score"])
    return rows


def main() -> None:
    dev_tasks = load_jsonl(DATA_DIR / "dev_tasks.jsonl")
    dev_answers = load_json(DATA_DIR / "dev_answers.json")
    payload = run_harness(dev_tasks)

    scorer_ns = load_baseline_scorer()
    report = scorer_ns["score_dev_submission"](payload, dev_answers)
    failures = write_failure_report(payload, dev_answers, dev_tasks, scorer_ns)

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "dev_score.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "dev_failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "dev_payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("wrote reports/dev_score.json")
    print("wrote reports/dev_failures.json")


if __name__ == "__main__":
    main()
