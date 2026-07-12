from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_submission(path: Path) -> dict[str, Any]:
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f))
    return json.loads(row["submission"])


def normalize(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    text = re.sub(r"final_(?:dev|screening)_[a-z0-9_]+", "TASK_ID", text)
    text = re.sub(r"(?:obj|rec|mem_profile)_[a-z0-9]+", "OPAQUE_ID", text)
    text = re.sub(r"WM-\d+", "WM_REF", text)
    text = re.sub(r"marker_[a-z]+", "MARKER", text)
    text = re.sub(r"\d+", "NUM", text)
    return text.lower()


def task_text(task: dict[str, Any], focal_id: str, *, records_only: bool = False) -> str:
    focal = next(
        (obj for obj in task.get("device_state", {}).get("objects", []) if str(obj.get("id")) == focal_id),
        {},
    )
    records = task.get("device_state", {}).get("records", [])
    if records_only:
        value = {str(r.get("type")): r.get("value") for r in records}
    else:
        value = {
            "prompt": task.get("prompt"),
            "history": task.get("visible_history"),
            "records": records,
            "focal": {"type": focal.get("type"), "attrs": focal.get("attrs")},
        }
    return normalize(value)


def build_model(analyzer: str, ngram_range: tuple[int, int], min_df: int = 1) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [("text", TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=min_df), 0)],
                    remainder="drop",
                ),
            ),
            ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced", C=4.0)),
        ]
    )


def main() -> None:
    dev_tasks = load_jsonl(DATA / "dev_tasks.jsonl")
    dev_answers = json.loads((DATA / "dev_answers.json").read_text(encoding="utf-8"))["answers"]
    screening_tasks = load_jsonl(DATA / "screening_tasks.jsonl")
    current = load_submission(ROOT / "artifacts" / "submissions" / "submission_public_0_8459.csv")["answers"]

    dev_rows = [
        [task_text(task, str(dev_answers[str(task["id"])]["focal_id"]))]
        for task in dev_tasks
    ]
    labels = [str(dev_answers[str(task["id"])]["control"]) for task in dev_tasks]
    screen_rows = [[task_text(task, str(current[str(task["id"])]["focal_id"]))] for task in screening_tasks]

    specs = [
        ("word", (1, 2), 1),
        ("char_wb", (3, 5), 1),
        ("char_wb", (4, 7), 2),
    ]
    predictions: list[list[str]] = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for spec in specs:
        model = build_model(*spec)
        cv_pred = cross_val_predict(model, dev_rows, labels, cv=cv)
        accuracy = sum(a == b for a, b in zip(cv_pred, labels)) / len(labels)
        print("CV", spec, round(accuracy, 4), Counter(zip(labels, cv_pred)))
        model.fit(dev_rows, labels)
        predictions.append(list(model.predict(screen_rows)))

    task_by_id = {str(task["id"]): task for task in screening_tasks}
    disagreements: list[tuple[str, str, str, str]] = []
    for index, task in enumerate(screening_tasks):
        task_id = str(task["id"])
        if "단," in str(task.get("prompt", "")):
            continue
        votes = [pred[index] for pred in predictions]
        if len(set(votes)) != 1:
            continue
        predicted = votes[0]
        existing = str(current[task_id]["control"])
        if predicted != existing:
            disagreements.append((task_id, existing, predicted, str(task.get("prompt", ""))))

    print("\nUNANIMOUS NO-SUFFIX DISAGREEMENTS", len(disagreements))
    print("TRANSITIONS", Counter((old, new) for _, old, new, _ in disagreements))
    for row in disagreements:
        task = task_by_id[row[0]]
        records = {str(r.get("type")): r.get("value") for r in task.get("device_state", {}).get("records", [])}
        print(" | ".join(row), "|", json.dumps(records, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
