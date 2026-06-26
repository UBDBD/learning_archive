# Project Structure

Public GitHub package for the DACON/ETRI lifelog competition project.

Competition data, final submission CSV files, and public-probe reports are not
included in this repository because they contain competition-provided data,
derived predictions, or leaderboard operation history.

## Root

- `README.md`: Korean project overview and quick start.
- `solution.py`: training/inference script. Requires the original DACON data under `data/`.
- `reproduce_final_submission.py`: local final-submission reproduction helper. Requires private local submission artifacts that are intentionally not committed.
- `requirements.txt`: Python dependencies.
- `PROJECT_STRUCTURE.md`: this file.

## Excluded Local Data

The following paths are ignored by `.gitignore` and should stay local only:

- `data/`: DACON-provided train/test metadata and lifelog parquet files.
- `submission.csv`: root upload file.
- `submissions/`: final and exploratory submission CSV files.
- `reports/`: reproduction reports and public-probe queue records.
- `models/`, `artifacts/`, `cache/`: generated model artifacts.

To run the code, download the official data from DACON and place files in the
same `data/` layout described in `README.md`.

## Docs

- `docs/competition_overview.md`: competition goal, evaluation, data structure, and submission rules.
- `docs/model_and_reproducibility.md`: final artifact reproduction and model/code explanation.
- `docs/private_score_postmortem.md`: private-score overfit analysis.

## Run Baseline/Solution Pipeline

```bash
python3 solution.py --data-dir data --out submission.csv
```

Exact final artifact reproduction additionally requires the local files under
`submissions/final/`, which are not committed to GitHub.
