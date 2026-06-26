#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproducible baseline-to-strong solution for the 2026 lifelog challenge."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except Exception:  # LightGBM is optional; the sklearn ensemble remains fully runnable.
    LGBMClassifier = None


TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
KEYS = ["subject_id", "date"]
WINDOWS = ["dawn", "morning", "afternoon", "evening"]
EPS = 1e-6
TIME_LABEL_PARAMS = {
    "Q1": {"k": 40, "power": 1.0, "smoothing": 8.0},
    "Q2": {"k": 40, "power": 1.5, "smoothing": 8.0},
    "Q3": {"k": 2, "power": 0.0, "smoothing": 2.0},
    "S1": {"k": 40, "power": 0.5, "smoothing": 8.0},
    "S2": {"k": 40, "power": 0.5, "smoothing": 4.0},
    "S3": {"k": 40, "power": 0.0, "smoothing": 4.0},
    "S4": {"k": 40, "power": 0.5, "smoothing": 4.0},
}
TIME_COMPONENT_RECIPES = {
    "Q1": ["default", "prev_next", "linear", "abs_k40", "run_same"],
    "Q2": ["default", "abs_k8", "prev_next", "run_same"],
    "Q3": ["default", "prev_next", "linear", "abs_k8", "run_same"],
    "S1": ["default", "abs_k80", "nearest", "subject", "run_same"],
    "S2": ["default", "abs_k40", "abs_k80", "prev"],
    "S3": ["default", "subject", "run_same"],
    "S4": ["default", "abs_k40", "abs_k80", "next", "run_same"],
}
TIME_COMPONENT_MODEL_FEATURE_TARGETS = {"Q1", "Q2", "Q3", "S4"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--aggressive-out", default="submissions/generated/submission_aggressive.csv")
    parser.add_argument("--low-time-out", default="submissions/generated/submission_low_time.csv")
    parser.add_argument("--ensemble-out", default="submissions/generated/submission_ensemble.csv")
    parser.add_argument("--notime-out", default="submissions/generated/submission_notime.csv")
    parser.add_argument("--shrunk-out", default="submissions/generated/submission_shrunk.csv")
    parser.add_argument("--stack-out", default="submissions/generated/submission_stack.csv")
    parser.add_argument("--stack-safe-out", default="submissions/generated/submission_stack_safe.csv")
    parser.add_argument("--joint-out", default="submissions/generated/submission_joint.csv")
    parser.add_argument("--stack-joint-out", default="submissions/generated/submission_stack_joint.csv")
    parser.add_argument("--stage2-out", default="submissions/generated/submission_stage2.csv")
    parser.add_argument("--stage2-joint-out", default="submissions/generated/submission_stage2_joint.csv")
    parser.add_argument("--report-out", default="reports/cv_report.json")
    parser.add_argument("--importance-out", default="reports/feature_importance.csv")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--feature-route", choices=["full", "norelroll", "stable"], default="full")
    parser.add_argument("--disable-subject-rolling", action="store_true")
    parser.add_argument("--high-card-top-n", type=int, default=None)
    parser.add_argument("--disable-raw-id-features", action="store_true")
    parser.add_argument("--stable-high-card-summary-only", action="store_true")
    parser.add_argument("--skip-meta-models", action="store_true")
    parser.add_argument("--public-blend-base", default=None)
    parser.add_argument("--public-blend-alt", default=None)
    parser.add_argument("--public-blend-weight", type=float, default=None)
    parser.add_argument("--public-blend-report-out", default="reports/public_blend_report.json")
    parser.add_argument("--prediction-reference", default=None)
    parser.add_argument("--prediction-distance-out", default="reports/prediction_distance.csv")
    parser.add_argument("--time-blend-cap", type=float, default=0.55)
    parser.add_argument("--low-time-blend-cap", type=float, default=0.25)
    parser.add_argument("--block-gap-days", type=int, default=5)
    parser.add_argument("--enable-sleep-v2", action="store_true")
    parser.add_argument("--target-feature-mode", choices=["all", "compact"], default="all")
    parser.add_argument("--max-target-features", type=int, default=900)
    parser.add_argument("--skip-group-cv", action="store_true")
    parser.add_argument("--skip-block-cv", action="store_true")
    return parser.parse_args()


def clean_token(value: object, max_len: int = 48) -> str:
    raw = str(value)
    token = re.sub(r"[^0-9a-zA-Z]+", "_", raw.lower()).strip("_")
    digest = hashlib.md5(raw.encode("utf-8", errors="ignore")).hexdigest()[:8]
    if not token:
        token = f"val_{digest}"
    if len(token) > max_len:
        token = f"{token[:max_len]}_{digest}"
    return token


def clip_prob(values: np.ndarray) -> np.ndarray:
    return np.clip(values.astype(float), EPS, 1.0 - EPS)


def logit_prob(values: np.ndarray) -> np.ndarray:
    p = clip_prob(values)
    return np.log(p / (1.0 - p))


def time_window(hour: pd.Series) -> pd.Series:
    bins = [-1, 5, 11, 17, 23]
    return pd.cut(hour, bins=bins, labels=WINDOWS).astype(str)


def ensure_datetime(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    for col in cols:
        df[col] = pd.to_datetime(df[col])
    return df


def load_tables(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "ch2026_metrics_train.csv")
    sample = pd.read_csv(data_dir / "ch2026_submission_sample.csv")
    train = ensure_datetime(train, ["sleep_date", "lifelog_date"])
    sample = ensure_datetime(sample, ["sleep_date", "lifelog_date"])
    sample_for_output = sample.copy()

    train["is_train"] = 1
    sample["is_train"] = 0
    train["row_id"] = np.arange(len(train))
    sample["row_id"] = np.arange(len(sample))

    combined = pd.concat([train, sample], ignore_index=True, sort=False)
    combined["date"] = combined["lifelog_date"].dt.normalize()
    combined["sleep_date"] = pd.to_datetime(combined["sleep_date"])
    combined["lifelog_date"] = pd.to_datetime(combined["lifelog_date"])
    return train, sample_for_output, combined


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    min_date = out["lifelog_date"].min()
    out["lifelog_dow"] = out["lifelog_date"].dt.dayofweek
    out["lifelog_day"] = out["lifelog_date"].dt.day
    out["lifelog_month"] = out["lifelog_date"].dt.month
    out["lifelog_is_weekend"] = (out["lifelog_dow"] >= 5).astype(int)
    out["sleep_dow"] = out["sleep_date"].dt.dayofweek
    out["elapsed_day"] = (out["lifelog_date"] - min_date).dt.days
    out["subject_num"] = out["subject_id"].str.extract(r"(\d+)").astype(float)
    out = out.sort_values(["subject_id", "lifelog_date", "is_train", "row_id"]).copy()
    out["subject_seq"] = out.groupby("subject_id").cumcount()
    out["subject_elapsed_day"] = (
        out["lifelog_date"] - out.groupby("subject_id")["lifelog_date"].transform("min")
    ).dt.days
    out = out.sort_values(["is_train", "row_id"], ascending=[False, True]).copy()
    return out.reset_index(drop=True)


def read_sensor(data_dir: Path, filename: str, keys: pd.DataFrame) -> pd.DataFrame:
    path = data_dir / "ch2025_data_items" / filename
    if not path.exists():
        warnings.warn(f"Missing sensor file: {path}")
        return pd.DataFrame(columns=KEYS + ["timestamp", "hour"])

    try:
        df = pd.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError(
            "Reading parquet files requires pyarrow. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.normalize()
    df["hour"] = df["timestamp"].dt.hour
    keep = keys.drop_duplicates().copy()
    df = df.merge(keep, on=KEYS, how="inner")
    return df


def read_sensor_episode(data_dir: Path, filename: str, base: pd.DataFrame) -> pd.DataFrame:
    path = data_dir / "ch2025_data_items" / filename
    if not path.exists():
        warnings.warn(f"Missing sensor file: {path}")
        return pd.DataFrame(columns=KEYS + ["timestamp", "hour", "episode_phase"])

    try:
        df = pd.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError(
            "Reading parquet files requires pyarrow. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    targets = base[["subject_id", "date"]].drop_duplicates().copy()
    pre_map = targets.rename(columns={"date": "target_date"})
    pre_map["source_date"] = pre_map["target_date"]
    pre_map["episode_part"] = "pre"
    post_map = targets.rename(columns={"date": "target_date"})
    post_map["source_date"] = post_map["target_date"] + pd.Timedelta(days=1)
    post_map["episode_part"] = "post"
    mapping = pd.concat([pre_map, post_map], ignore_index=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["source_date"] = df["timestamp"].dt.normalize()
    df["hour"] = df["timestamp"].dt.hour
    df = df.merge(mapping, on=["subject_id", "source_date"], how="inner")
    if df.empty:
        return pd.DataFrame(columns=KEYS + ["timestamp", "hour", "episode_phase"])

    pre_mask = (df["episode_part"] == "pre") & (df["hour"] >= 18)
    post_mask = (df["episode_part"] == "post") & (df["hour"] < 12)
    df = df[pre_mask | post_mask].copy()
    if df.empty:
        return pd.DataFrame(columns=KEYS + ["timestamp", "hour", "episode_phase"])

    df["episode_phase"] = np.where(
        df["episode_part"] == "pre",
        "pre",
        np.where(df["hour"] < 6, "night", "wake"),
    )
    df["date"] = df["target_date"]
    return df.drop(columns=["target_date", "source_date", "episode_part"])


def with_available(out: pd.DataFrame, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    counts = df.groupby(KEYS).size().rename(f"{prefix}_rows").reset_index()
    out = counts.merge(out, on=KEYS, how="left")
    out[f"{prefix}_available"] = 1
    return out


def numeric_episode_features(
    df: pd.DataFrame,
    value_cols: Sequence[str],
    prefix: str,
) -> pd.DataFrame:
    if df.empty or "episode_phase" not in df.columns:
        return pd.DataFrame(columns=KEYS)

    use_cols = [col for col in value_cols if col in df.columns]
    if not use_cols:
        return pd.DataFrame(columns=KEYS)

    data = df[KEYS + ["episode_phase"] + use_cols].copy()
    for col in use_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    pieces = []
    grouped = data.groupby(KEYS, dropna=False)
    overall = grouped[use_cols].agg(["count", "mean", "std", "min", "max", "median", "sum"])
    overall.columns = [f"{prefix}_episode_all_{col}_{stat}" for col, stat in overall.columns]
    pieces.append(overall)

    phase = data.groupby(KEYS + ["episode_phase"], dropna=False)[use_cols].agg(
        ["count", "mean", "std", "max", "sum"]
    )
    phase = phase.unstack("episode_phase")
    phase.columns = [
        f"{prefix}_episode_{clean_token(phase_name)}_{col}_{stat}"
        for col, stat, phase_name in phase.columns
    ]
    pieces.append(phase)
    out = pd.concat(pieces, axis=1).reset_index()
    return with_available(out, df, f"{prefix}_episode")


def sequence_condition_features(
    df: pd.DataFrame,
    condition: pd.Series,
    prefix: str,
    name: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=KEYS)

    tmp = df[KEYS + ["timestamp", "hour"]].copy()
    tmp["condition"] = condition.astype(bool).to_numpy()
    tmp = tmp.sort_values(KEYS + ["timestamp"])

    rows = []
    for key, group in tmp.groupby(KEYS, sort=False):
        arr = group["condition"].to_numpy(dtype=bool)
        hours = group["hour"].to_numpy(dtype=float)
        if len(arr) == 0:
            continue
        longest = 0
        current = 0
        for value in arr:
            current = current + 1 if value else 0
            longest = max(longest, current)
        true_hours = hours[arr]
        rows.append(
            {
                "subject_id": key[0],
                "date": key[1],
                f"{prefix}_{name}_episode_sum": float(arr.sum()),
                f"{prefix}_{name}_episode_ratio": float(arr.mean()),
                f"{prefix}_{name}_episode_longest_run": float(longest),
                f"{prefix}_{name}_episode_transitions": float(np.sum(arr[1:] != arr[:-1])) if len(arr) > 1 else 0.0,
                f"{prefix}_{name}_episode_first_hour": float(true_hours.min()) if len(true_hours) else np.nan,
                f"{prefix}_{name}_episode_last_hour": float(true_hours.max()) if len(true_hours) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def numeric_daily_features(
    df: pd.DataFrame,
    value_cols: Sequence[str],
    prefix: str,
    include_windows: bool = True,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=KEYS)

    use_cols = [col for col in value_cols if col in df.columns]
    if not use_cols:
        return pd.DataFrame(columns=KEYS)

    data = df[KEYS + ["hour"] + use_cols].copy()
    for col in use_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    pieces = []
    grouped = data.groupby(KEYS, dropna=False)
    agg = grouped[use_cols].agg(["count", "mean", "std", "min", "max", "median", "sum"])
    agg.columns = [f"{prefix}_{col}_{stat}" for col, stat in agg.columns]
    pieces.append(agg)

    for q, name in [(0.25, "q25"), (0.75, "q75")]:
        qdf = grouped[use_cols].quantile(q)
        qdf.columns = [f"{prefix}_{col}_{name}" for col in qdf.columns]
        pieces.append(qdf)

    if include_windows:
        data["window"] = time_window(data["hour"])
        windowed = data.groupby(KEYS + ["window"], dropna=False)[use_cols].agg(
            ["count", "mean", "max", "sum"]
        )
        windowed = windowed.unstack("window")
        windowed.columns = [
            f"{prefix}_{col}_{clean_token(window)}_{stat}"
            for col, stat, window in windowed.columns
        ]
        pieces.append(windowed)

    out = pd.concat(pieces, axis=1).reset_index()
    return with_available(out, df, prefix)


def bool_daily_features(
    df: pd.DataFrame,
    bool_expr: pd.Series,
    prefix: str,
    name: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=KEYS)
    tmp = df[KEYS].copy()
    tmp[name] = bool_expr.astype(float).to_numpy()
    out = tmp.groupby(KEYS)[name].agg(["sum", "mean"]).reset_index()
    out.columns = KEYS + [f"{prefix}_{name}_sum", f"{prefix}_{name}_ratio"]
    return out


def category_daily_features(
    df: pd.DataFrame,
    col: str,
    prefix: str,
    top_values: Optional[Sequence[object]] = None,
    include_proportion: bool = True,
) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return pd.DataFrame(columns=KEYS)

    tmp = df[KEYS + [col]].dropna().copy()
    if tmp.empty:
        return pd.DataFrame(columns=KEYS)

    if top_values is None:
        top_values = sorted(tmp[col].unique())

    tmp[col] = tmp[col].astype(str)
    top_values = [str(v) for v in top_values]
    tmp = tmp[tmp[col].isin(top_values)]
    if tmp.empty:
        return pd.DataFrame(columns=KEYS)

    counts = tmp.groupby(KEYS + [col]).size().unstack(col, fill_value=0)
    counts.columns = [f"{prefix}_{col}_{clean_token(v)}_count" for v in counts.columns]
    pieces = [counts]

    if include_proportion:
        denom = df.groupby(KEYS).size().replace(0, np.nan)
        props = counts.div(denom, axis=0)
        props.columns = [c.replace("_count", "_prop") for c in counts.columns]
        pieces.append(props)

    return pd.concat(pieces, axis=1).reset_index()


def top_category_value_features(
    df: pd.DataFrame,
    cat_col: str,
    prefix: str,
    top_n: int,
    value_col: Optional[str] = None,
    value_aggs: Sequence[str] = ("sum", "mean", "max"),
) -> pd.DataFrame:
    if df.empty or cat_col not in df.columns:
        return pd.DataFrame(columns=KEYS)

    tmp = df[KEYS + [cat_col] + ([value_col] if value_col else [])].dropna(subset=[cat_col]).copy()
    if tmp.empty:
        return pd.DataFrame(columns=KEYS)

    tmp[cat_col] = tmp[cat_col].astype(str)
    top = tmp[cat_col].value_counts().head(top_n).index
    tmp = tmp[tmp[cat_col].isin(top)]
    if tmp.empty:
        return pd.DataFrame(columns=KEYS)

    pieces = []
    counts = tmp.groupby(KEYS + [cat_col]).size().unstack(cat_col, fill_value=0)
    counts.columns = [f"{prefix}_{cat_col}_{clean_token(c)}_count" for c in counts.columns]
    pieces.append(counts)

    if value_col:
        tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
        for agg_name in value_aggs:
            values = tmp.groupby(KEYS + [cat_col])[value_col].agg(agg_name).unstack(cat_col)
            values.columns = [
                f"{prefix}_{cat_col}_{clean_token(c)}_{value_col}_{agg_name}"
                for c in values.columns
            ]
            pieces.append(values)

    return pd.concat(pieces, axis=1).fillna(0).reset_index()


def merge_feature_frames(base: pd.DataFrame, frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    out = base.copy()
    for frame in frames:
        if frame is None or frame.empty:
            continue
        out = out.merge(frame, on=KEYS, how="left")
    return out


def process_scalar_status(
    data_dir: Path,
    keys: pd.DataFrame,
    filename: str,
    value_col: str,
    prefix: str,
    categorical: bool = True,
) -> pd.DataFrame:
    df = read_sensor(data_dir, filename, keys)
    if df.empty:
        return pd.DataFrame(columns=KEYS)

    feats = [numeric_daily_features(df, [value_col], prefix, include_windows=True)]
    if categorical:
        feats.append(category_daily_features(df, value_col, prefix))
    return merge_feature_frames(df[KEYS].drop_duplicates(), feats)


def process_light(data_dir: Path, keys: pd.DataFrame, filename: str, value_col: str, prefix: str) -> pd.DataFrame:
    df = read_sensor(data_dir, filename, keys)
    if df.empty:
        return pd.DataFrame(columns=KEYS)

    value = pd.to_numeric(df[value_col], errors="coerce")
    feats = [
        numeric_daily_features(df, [value_col], prefix, include_windows=True),
        bool_daily_features(df, value <= 10, prefix, "dark"),
        bool_daily_features(df, value >= 300, prefix, "bright"),
        bool_daily_features(df, (df["hour"] >= 18) & (value >= 100), prefix, "evening_bright"),
    ]
    return merge_feature_frames(df[KEYS].drop_duplicates(), feats)


def process_pedo(data_dir: Path, keys: pd.DataFrame) -> pd.DataFrame:
    df = read_sensor(data_dir, "ch2025_wPedo.parquet", keys)
    if df.empty:
        return pd.DataFrame(columns=KEYS)

    value_cols = [
        "step",
        "step_frequency",
        "running_step",
        "walking_step",
        "distance",
        "speed",
        "burned_calories",
    ]
    step = pd.to_numeric(df["step"], errors="coerce")
    speed = pd.to_numeric(df["speed"], errors="coerce")
    feats = [
        numeric_daily_features(df, value_cols, "wPedo", include_windows=True),
        bool_daily_features(df, step > 0, "wPedo", "active_minute"),
        bool_daily_features(df, step == 0, "wPedo", "zero_step"),
        bool_daily_features(df, speed > 0.1, "wPedo", "moving_minute"),
    ]
    return merge_feature_frames(df[KEYS].drop_duplicates(), feats)


def normalize_record(value: object) -> Dict[str, object]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "as_py"):
        return value.as_py()
    try:
        return dict(value)  # type: ignore[arg-type]
    except Exception:
        return {}


def explode_struct_list(df: pd.DataFrame, list_col: str) -> pd.DataFrame:
    if df.empty or list_col not in df.columns:
        return pd.DataFrame(columns=KEYS + ["hour"])

    meta_cols = KEYS + [c for c in ["timestamp", "hour", "episode_phase"] if c in df.columns]
    exploded = df[meta_cols + [list_col]].explode(list_col)
    exploded = exploded[exploded[list_col].notna()]
    if exploded.empty:
        return pd.DataFrame(columns=KEYS + ["hour"])

    records = [normalize_record(v) for v in exploded[list_col].to_list()]
    rec_df = pd.DataFrame.from_records(records)
    exploded = exploded[meta_cols].reset_index(drop=True)
    return pd.concat([exploded, rec_df.reset_index(drop=True)], axis=1)


def process_hr(data_dir: Path, keys: pd.DataFrame) -> pd.DataFrame:
    df = read_sensor(data_dir, "ch2025_wHr.parquet", keys)
    if df.empty:
        return pd.DataFrame(columns=KEYS)

    exploded = df[KEYS + ["hour", "heart_rate"]].explode("heart_rate")
    exploded = exploded[exploded["heart_rate"].notna()].copy()
    exploded["heart_rate"] = pd.to_numeric(exploded["heart_rate"], errors="coerce")
    feats = [
        numeric_daily_features(exploded, ["heart_rate"], "wHr", include_windows=True),
        bool_daily_features(exploded, (exploded["hour"] < 6), "wHr", "night_valid"),
        bool_daily_features(exploded, (exploded["hour"] >= 18), "wHr", "evening_valid"),
    ]
    return merge_feature_frames(df[KEYS].drop_duplicates(), feats)


def process_gps(data_dir: Path, keys: pd.DataFrame) -> pd.DataFrame:
    df = read_sensor(data_dir, "ch2025_mGps.parquet", keys)
    if df.empty:
        return pd.DataFrame(columns=KEYS)

    gps = explode_struct_list(df, "m_gps")
    if gps.empty:
        return pd.DataFrame(columns=KEYS)

    for col in ["altitude", "latitude", "longitude", "speed"]:
        gps[col] = pd.to_numeric(gps.get(col), errors="coerce")

    lat_cell = gps["latitude"].round(4).astype(str)
    lon_cell = gps["longitude"].round(4).astype(str)
    gps["location_cell"] = lat_cell + "_" + lon_cell
    feats = [
        numeric_daily_features(gps, ["altitude", "latitude", "longitude", "speed"], "mGps", True),
        bool_daily_features(gps, gps["speed"] <= 0.1, "mGps", "stationary"),
        bool_daily_features(gps, gps["speed"] >= 0.5, "mGps", "moving"),
    ]
    cell_counts = gps.groupby(KEYS)["location_cell"].nunique().rename("mGps_location_cell_nunique").reset_index()
    feats.append(cell_counts)
    return merge_feature_frames(df[KEYS].drop_duplicates(), feats)


def process_wifi(
    data_dir: Path,
    keys: pd.DataFrame,
    top_n: int,
    include_raw_id_features: bool = True,
) -> pd.DataFrame:
    df = read_sensor(data_dir, "ch2025_mWifi.parquet", keys)
    if df.empty:
        return pd.DataFrame(columns=KEYS)

    wifi = explode_struct_list(df, "m_wifi")
    if wifi.empty:
        return pd.DataFrame(columns=KEYS)

    wifi["rssi"] = pd.to_numeric(wifi.get("rssi"), errors="coerce")
    feats = [
        numeric_daily_features(wifi, ["rssi"], "mWifi", include_windows=True),
        bool_daily_features(wifi, wifi["rssi"] >= -60, "mWifi", "strong_signal"),
    ]
    if include_raw_id_features and top_n > 0:
        feats.append(top_category_value_features(wifi, "bssid", "mWifi", top_n=top_n, value_col="rssi", value_aggs=("max",)))
    unique = wifi.groupby(KEYS)["bssid"].nunique().rename("mWifi_bssid_nunique").reset_index()
    feats.append(unique)
    return merge_feature_frames(df[KEYS].drop_duplicates(), feats)


def process_ble(
    data_dir: Path,
    keys: pd.DataFrame,
    top_n: int,
    include_raw_id_features: bool = True,
) -> pd.DataFrame:
    df = read_sensor(data_dir, "ch2025_mBle.parquet", keys)
    if df.empty:
        return pd.DataFrame(columns=KEYS)

    ble = explode_struct_list(df, "m_ble")
    if ble.empty:
        return pd.DataFrame(columns=KEYS)

    ble["rssi"] = pd.to_numeric(ble.get("rssi"), errors="coerce")
    feats = [
        numeric_daily_features(ble, ["rssi"], "mBle", include_windows=True),
        bool_daily_features(ble, ble["rssi"] >= -60, "mBle", "strong_signal"),
    ]
    if include_raw_id_features and top_n > 0:
        feats.append(top_category_value_features(ble, "address", "mBle", top_n=top_n, value_col="rssi", value_aggs=("max",)))
    feats.append(top_category_value_features(ble, "device_class", "mBle", top_n=12, value_col="rssi", value_aggs=("mean", "max")))
    unique = ble.groupby(KEYS)["address"].nunique().rename("mBle_address_nunique").reset_index()
    feats.append(unique)
    return merge_feature_frames(df[KEYS].drop_duplicates(), feats)


def parse_ambience_item(value: object) -> Tuple[Optional[str], float]:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, dict):
        label = value.get("label") or value.get("name") or value.get("sound")
        prob = value.get("probability") or value.get("prob") or value.get("score")
    elif isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 2:
        label, prob = value[0], value[1]
    else:
        return None, np.nan
    try:
        return str(label), float(prob)
    except Exception:
        return str(label), np.nan


def entropy(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    arr = arr[arr > 0]
    total = arr.sum()
    if total <= 0:
        return 0.0
    prob = arr / total
    return float(-(prob * np.log(prob)).sum())


def process_ambience(data_dir: Path, keys: pd.DataFrame, top_n: int) -> pd.DataFrame:
    df = read_sensor(data_dir, "ch2025_mAmbience.parquet", keys)
    if df.empty:
        return pd.DataFrame(columns=KEYS)

    exploded = df[KEYS + ["hour", "m_ambience"]].explode("m_ambience")
    exploded = exploded[exploded["m_ambience"].notna()].copy()
    parsed = [parse_ambience_item(v) for v in exploded["m_ambience"].to_list()]
    ambience = exploded[KEYS + ["hour"]].reset_index(drop=True)
    ambience["label"] = [p[0] for p in parsed]
    ambience["prob"] = [p[1] for p in parsed]
    ambience = ambience.dropna(subset=["label"]).copy()

    feats = [
        numeric_daily_features(ambience, ["prob"], "mAmbience", include_windows=True),
        top_category_value_features(
            ambience, "label", "mAmbience", top_n=top_n, value_col="prob", value_aggs=("sum", "mean", "max")
        ),
    ]
    label_unique = ambience.groupby(KEYS)["label"].nunique().rename("mAmbience_label_nunique").reset_index()
    ent = ambience.groupby(KEYS)["prob"].apply(entropy).rename("mAmbience_prob_entropy").reset_index()
    feats.extend([label_unique, ent])

    idx = ambience.groupby(KEYS)["prob"].idxmax()
    top_label_rows = ambience.loc[idx, KEYS + ["label", "prob"]].copy()
    top_labels = top_label_rows["label"].value_counts().head(top_n).index
    feats.append(category_daily_features(top_label_rows, "label", "mAmbience_top", top_values=top_labels, include_proportion=False))
    top_prob = top_label_rows.rename(columns={"prob": "mAmbience_top_prob"})[KEYS + ["mAmbience_top_prob"]]
    feats.append(top_prob)
    return merge_feature_frames(df[KEYS].drop_duplicates(), feats)


def process_usage(
    data_dir: Path,
    keys: pd.DataFrame,
    top_n: int,
    include_raw_id_features: bool = True,
) -> pd.DataFrame:
    df = read_sensor(data_dir, "ch2025_mUsageStats.parquet", keys)
    if df.empty:
        return pd.DataFrame(columns=KEYS)

    usage = explode_struct_list(df, "m_usage_stats")
    if usage.empty:
        return pd.DataFrame(columns=KEYS)

    usage["total_time"] = pd.to_numeric(usage.get("total_time"), errors="coerce")
    feats = [
        numeric_daily_features(usage, ["total_time"], "mUsageStats", include_windows=True),
    ]
    if include_raw_id_features and top_n > 0:
        feats.append(
            top_category_value_features(
                usage, "app_name", "mUsageStats", top_n=top_n, value_col="total_time", value_aggs=("sum", "mean", "max")
            )
        )
    app_unique = usage.groupby(KEYS)["app_name"].nunique().rename("mUsageStats_app_nunique").reset_index()
    app_count = usage.groupby(KEYS).size().rename("mUsageStats_app_count").reset_index()
    feats.extend([app_unique, app_count])
    return merge_feature_frames(df[KEYS].drop_duplicates(), feats)


def process_episode_scalar(
    data_dir: Path,
    base: pd.DataFrame,
    filename: str,
    value_col: str,
    prefix: str,
    categorical: bool = False,
) -> pd.DataFrame:
    df = read_sensor_episode(data_dir, filename, base)
    if df.empty:
        return pd.DataFrame(columns=KEYS)

    feats = [numeric_episode_features(df, [value_col], prefix)]
    if categorical:
        feats.append(category_daily_features(df, value_col, f"{prefix}_episode"))
    value = pd.to_numeric(df[value_col], errors="coerce")

    if value_col == "m_screen_use":
        feats.append(sequence_condition_features(df, value == 0, prefix, "screen_off"))
        feats.append(sequence_condition_features(df, value == 1, prefix, "screen_on"))
    elif value_col == "m_charging":
        feats.append(sequence_condition_features(df, value == 1, prefix, "charging"))
        feats.append(sequence_condition_features(df, value == 0, prefix, "not_charging"))
    elif value_col == "m_activity":
        feats.append(sequence_condition_features(df, value == 0, prefix, "still"))
        feats.append(sequence_condition_features(df, value > 0, prefix, "active"))

    return merge_feature_frames(df[KEYS].drop_duplicates(), feats)


def process_episode_light(
    data_dir: Path,
    base: pd.DataFrame,
    filename: str,
    value_col: str,
    prefix: str,
) -> pd.DataFrame:
    df = read_sensor_episode(data_dir, filename, base)
    if df.empty:
        return pd.DataFrame(columns=KEYS)

    value = pd.to_numeric(df[value_col], errors="coerce")
    feats = [
        numeric_episode_features(df, [value_col], prefix),
        sequence_condition_features(df, value <= 10, prefix, "dark"),
        sequence_condition_features(df, value >= 300, prefix, "bright"),
    ]
    return merge_feature_frames(df[KEYS].drop_duplicates(), feats)


def process_episode_pedo(data_dir: Path, base: pd.DataFrame) -> pd.DataFrame:
    df = read_sensor_episode(data_dir, "ch2025_wPedo.parquet", base)
    if df.empty:
        return pd.DataFrame(columns=KEYS)

    value_cols = ["step", "step_frequency", "distance", "speed", "burned_calories"]
    step = pd.to_numeric(df["step"], errors="coerce")
    speed = pd.to_numeric(df["speed"], errors="coerce")
    feats = [
        numeric_episode_features(df, value_cols, "wPedo"),
        sequence_condition_features(df, step == 0, "wPedo", "zero_step"),
        sequence_condition_features(df, step > 0, "wPedo", "active_step"),
        sequence_condition_features(df, speed <= 0.1, "wPedo", "stationary"),
        sequence_condition_features(df, speed > 0.1, "wPedo", "moving"),
    ]
    return merge_feature_frames(df[KEYS].drop_duplicates(), feats)


def process_episode_hr(data_dir: Path, base: pd.DataFrame) -> pd.DataFrame:
    df = read_sensor_episode(data_dir, "ch2025_wHr.parquet", base)
    if df.empty:
        return pd.DataFrame(columns=KEYS)

    exploded = df[KEYS + ["timestamp", "hour", "episode_phase", "heart_rate"]].explode("heart_rate")
    exploded = exploded[exploded["heart_rate"].notna()].copy()
    exploded["heart_rate"] = pd.to_numeric(exploded["heart_rate"], errors="coerce")
    feats = [
        numeric_episode_features(exploded, ["heart_rate"], "wHr"),
        sequence_condition_features(
            exploded,
            exploded["heart_rate"] <= exploded.groupby("subject_id")["heart_rate"].transform("median"),
            "wHr",
            "below_subject_median",
        ),
    ]
    return merge_feature_frames(df[KEYS].drop_duplicates(), feats)


def make_sleep_grid(base: pd.DataFrame, freq_minutes: int = 5) -> pd.DataFrame:
    targets = base[["subject_id", "date"]].drop_duplicates().copy()
    n_bins = int(18 * 60 / freq_minutes)
    offsets = pd.to_timedelta(np.arange(n_bins) * freq_minutes, unit="m")
    rows = []
    for row in targets.itertuples(index=False):
        start = pd.Timestamp(row.date) + pd.Timedelta(hours=18)
        tmp = pd.DataFrame(
            {
                "subject_id": row.subject_id,
                "date": row.date,
                "bin_idx": np.arange(n_bins, dtype=int),
                "bin_start": start + offsets,
            }
        )
        rows.append(tmp)
    grid = pd.concat(rows, ignore_index=True)
    grid["episode_hour"] = grid["bin_idx"] * freq_minutes / 60.0
    grid["clock_hour"] = (18.0 + grid["episode_hour"]) % 24.0
    grid["is_core_sleep_time"] = ((grid["clock_hour"] >= 22) | (grid["clock_hour"] < 9)).astype(float)
    return grid


def sensor_bin(df: pd.DataFrame, freq_minutes: int = 5) -> pd.DataFrame:
    out = df.copy()
    out["bin_start"] = out["timestamp"].dt.floor(f"{freq_minutes}min")
    return out


def sleep_block_aggregates(data_dir: Path, base: pd.DataFrame, freq_minutes: int = 5) -> pd.DataFrame:
    grid = make_sleep_grid(base, freq_minutes=freq_minutes)
    merge_cols = KEYS + ["bin_start"]

    def merge_bins(frame: pd.DataFrame) -> None:
        nonlocal grid
        if frame is not None and not frame.empty:
            grid = grid.merge(frame, on=merge_cols, how="left")

    screen = read_sensor_episode(data_dir, "ch2025_mScreenStatus.parquet", base)
    if not screen.empty:
        screen = sensor_bin(screen, freq_minutes)
        screen["m_screen_use"] = pd.to_numeric(screen["m_screen_use"], errors="coerce")
        merge_bins(
            screen.groupby(merge_cols)["m_screen_use"]
            .agg(sleepblock_screen_on_mean="mean", sleepblock_screen_count="count")
            .reset_index()
        )

    charging = read_sensor_episode(data_dir, "ch2025_mACStatus.parquet", base)
    if not charging.empty:
        charging = sensor_bin(charging, freq_minutes)
        charging["m_charging"] = pd.to_numeric(charging["m_charging"], errors="coerce")
        merge_bins(
            charging.groupby(merge_cols)["m_charging"]
            .agg(sleepblock_charging_mean="mean", sleepblock_charging_count="count")
            .reset_index()
        )

    activity = read_sensor_episode(data_dir, "ch2025_mActivity.parquet", base)
    if not activity.empty:
        activity = sensor_bin(activity, freq_minutes)
        activity["m_activity"] = pd.to_numeric(activity["m_activity"], errors="coerce")
        activity["sleepblock_activity_still"] = (activity["m_activity"] == 0).astype(float)
        activity["sleepblock_activity_active"] = (activity["m_activity"] > 0).astype(float)
        merge_bins(
            activity.groupby(merge_cols).agg(
                sleepblock_activity_mean=("m_activity", "mean"),
                sleepblock_activity_still_ratio=("sleepblock_activity_still", "mean"),
                sleepblock_activity_active_ratio=("sleepblock_activity_active", "mean"),
                sleepblock_activity_count=("m_activity", "count"),
            ).reset_index()
        )

    pedo = read_sensor_episode(data_dir, "ch2025_wPedo.parquet", base)
    if not pedo.empty:
        pedo = sensor_bin(pedo, freq_minutes)
        for col in ["step", "speed", "distance", "burned_calories"]:
            pedo[col] = pd.to_numeric(pedo[col], errors="coerce")
        pedo["sleepblock_pedo_zero"] = (pedo["step"] == 0).astype(float)
        pedo["sleepblock_pedo_moving"] = ((pedo["step"] > 0) | (pedo["speed"] > 0.1)).astype(float)
        merge_bins(
            pedo.groupby(merge_cols).agg(
                sleepblock_step_sum=("step", "sum"),
                sleepblock_speed_mean=("speed", "mean"),
                sleepblock_distance_sum=("distance", "sum"),
                sleepblock_calories_sum=("burned_calories", "sum"),
                sleepblock_pedo_zero_ratio=("sleepblock_pedo_zero", "mean"),
                sleepblock_pedo_moving_ratio=("sleepblock_pedo_moving", "mean"),
                sleepblock_pedo_count=("step", "count"),
            ).reset_index()
        )

    for filename, value_col, prefix in [
        ("ch2025_mLight.parquet", "m_light", "mlight"),
        ("ch2025_wLight.parquet", "w_light", "wlight"),
    ]:
        light = read_sensor_episode(data_dir, filename, base)
        if light.empty:
            continue
        light = sensor_bin(light, freq_minutes)
        light[value_col] = pd.to_numeric(light[value_col], errors="coerce")
        light[f"sleepblock_{prefix}_dark"] = (light[value_col] <= 10).astype(float)
        light[f"sleepblock_{prefix}_bright"] = (light[value_col] >= 300).astype(float)
        merge_bins(
            light.groupby(merge_cols).agg(
                **{
                    f"sleepblock_{prefix}_mean": (value_col, "mean"),
                    f"sleepblock_{prefix}_max": (value_col, "max"),
                    f"sleepblock_{prefix}_dark_ratio": (f"sleepblock_{prefix}_dark", "mean"),
                    f"sleepblock_{prefix}_bright_ratio": (f"sleepblock_{prefix}_bright", "mean"),
                    f"sleepblock_{prefix}_count": (value_col, "count"),
                }
            ).reset_index()
        )

    hr = read_sensor_episode(data_dir, "ch2025_wHr.parquet", base)
    if not hr.empty:
        hr = hr[KEYS + ["timestamp", "heart_rate"]].explode("heart_rate")
        hr = hr[hr["heart_rate"].notna()].copy()
        hr["heart_rate"] = pd.to_numeric(hr["heart_rate"], errors="coerce")
        hr = sensor_bin(hr, freq_minutes)
        merge_bins(
            hr.groupby(merge_cols)["heart_rate"]
            .agg(sleepblock_hr_mean="mean", sleepblock_hr_std="std", sleepblock_hr_count="count")
            .reset_index()
        )

    usage = read_sensor_episode(data_dir, "ch2025_mUsageStats.parquet", base)
    if not usage.empty:
        usage = explode_struct_list(usage, "m_usage_stats")
        if not usage.empty:
            usage["total_time"] = pd.to_numeric(usage["total_time"], errors="coerce")
            usage = sensor_bin(usage, freq_minutes)
            merge_bins(
                usage.groupby(merge_cols).agg(
                    sleepblock_usage_total_time_sum=("total_time", "sum"),
                    sleepblock_usage_app_count=("app_name", "count"),
                ).reset_index()
            )

    return grid


def longest_true_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values.astype(bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return int(longest)


def fill_short_gaps(mask: np.ndarray, max_gap: int = 2) -> np.ndarray:
    out = mask.astype(bool).copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i]:
            i += 1
            continue
        start = i
        while i < n and not out[i]:
            i += 1
        gap = i - start
        if start > 0 and i < n and gap <= max_gap:
            out[start:i] = True
    return out


def extract_segments(mask: np.ndarray, min_len: int = 6) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, value in enumerate(mask.astype(bool)):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start >= min_len:
                segments.append((start, i - 1))
            start = None
    if start is not None and len(mask) - start >= min_len:
        segments.append((start, len(mask) - 1))
    return segments


def count_true_segments(values: np.ndarray) -> int:
    arr = values.astype(bool)
    if len(arr) == 0:
        return 0
    starts = arr & np.r_[True, ~arr[:-1]]
    return int(starts.sum())


def best_scored_interval(
    values: np.ndarray,
    min_len: int,
    max_len: int,
) -> Tuple[int, int]:
    if len(values) == 0:
        return 0, 0
    clean = np.where(np.isfinite(values), values, -0.25).astype(float)
    n = len(clean)
    min_len = max(1, min(min_len, n))
    max_len = max(min_len, min(max_len, n))
    cumsum = np.r_[0.0, np.cumsum(clean)]
    best: Tuple[int, int] = (0, min_len - 1)
    best_value = -np.inf
    for start in range(n):
        longest = min(max_len, n - start)
        if longest < min_len:
            break
        for length in range(min_len, longest + 1):
            end = start + length - 1
            mean_score = (cumsum[end + 1] - cumsum[start]) / length
            duration_bonus = min(length, int(9 * 60 / 5)) / max_len
            value = mean_score + 0.08 * duration_bonus
            if value > best_value:
                best_value = value
                best = (start, end)
    return best


def summarize_sleep_block(group: pd.DataFrame, freq_minutes: int = 5) -> Dict[str, float]:
    g = group.sort_values("bin_idx").reset_index(drop=True)
    n = len(g)

    def arr(col: str) -> np.ndarray:
        return pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float) if col in g else np.full(n, np.nan)

    screen_on = arr("sleepblock_screen_on_mean")
    step_sum = arr("sleepblock_step_sum")
    still = arr("sleepblock_activity_still_ratio")
    speed = arr("sleepblock_speed_mean")
    m_dark = arr("sleepblock_mlight_dark_ratio")
    w_dark = arr("sleepblock_wlight_dark_ratio")
    charging = arr("sleepblock_charging_mean")
    usage_apps = arr("sleepblock_usage_app_count")
    hr_mean = arr("sleepblock_hr_mean")

    subject_hr_median = np.nanmedian(hr_mean)
    low_hr = np.where(np.isfinite(hr_mean) & np.isfinite(subject_hr_median), (hr_mean <= subject_hr_median).astype(float), np.nan)
    screen_off = np.where(np.isfinite(screen_on), 1.0 - np.clip(screen_on, 0, 1), np.nan)
    no_step = np.where(np.isfinite(step_sum), (step_sum <= 0).astype(float), np.nan)
    still = np.where(np.isfinite(still), np.clip(still, 0, 1), np.nan)
    low_speed = np.where(np.isfinite(speed), (speed <= 0.1).astype(float), np.nan)
    dark = np.nanmean(np.vstack([m_dark, w_dark]), axis=0)
    no_usage = np.where(np.isfinite(usage_apps), (usage_apps <= 0).astype(float), np.nan)
    charging = np.where(np.isfinite(charging), np.clip(charging, 0, 1), np.nan)
    time_prior = g["is_core_sleep_time"].to_numpy(dtype=float)

    component_values = np.vstack([screen_off, no_step, still, low_speed, dark, low_hr, no_usage, charging, time_prior])
    weights = np.asarray([1.5, 1.1, 1.0, 0.5, 0.9, 0.5, 0.5, 0.25, 0.35], dtype=float)
    valid = np.isfinite(component_values)
    weighted_sum = np.nansum(component_values * weights[:, None], axis=0)
    weight_sum = np.sum(valid * weights[:, None], axis=0)
    score = np.divide(weighted_sum, weight_sum, out=np.full(n, np.nan), where=weight_sum > 0)
    score = np.where(np.isfinite(score), score, 0.0)
    score_smooth = pd.Series(score).rolling(3, center=True, min_periods=1).mean().to_numpy()
    coverage = weight_sum / weights.sum()

    threshold = max(0.52, min(0.72, float(np.nanquantile(score_smooth, 0.60))))
    candidate = (score_smooth >= threshold) & (coverage >= 0.20)
    candidate = fill_short_gaps(candidate, max_gap=2)
    segments = extract_segments(candidate, min_len=6)

    if not segments:
        ranked = np.argsort(score_smooth)[::-1]
        top = np.zeros(n, dtype=bool)
        top[ranked[: max(6, min(24, n // 6))]] = True
        top = fill_short_gaps(top, max_gap=2)
        segments = extract_segments(top, min_len=3)

    if segments:
        def segment_value(seg: Tuple[int, int]) -> float:
            s, e = seg
            idx = np.arange(s, e + 1)
            duration = len(idx) * freq_minutes
            center_hour = float(np.nanmean(g.loc[idx, "clock_hour"]))
            core_bonus = float(np.nanmean(g.loc[idx, "is_core_sleep_time"]))
            score_bonus = float(np.nanmean(score_smooth[idx]))
            late_penalty = 0.15 if 9 <= center_hour < 18 else 0.0
            return duration / 60.0 + 2.0 * score_bonus + core_bonus - late_penalty

        best_start, best_end = max(segments, key=segment_value)
    else:
        best_start, best_end = 0, n - 1

    block_idx = np.arange(best_start, best_end + 1)
    pre_idx = np.arange(max(0, best_start - 12), best_start)
    inner_idx = block_idx[(block_idx >= best_start + 6) & (block_idx <= best_end - 6)]
    if len(inner_idx) == 0:
        inner_idx = block_idx

    disturbance = np.nanmean(
        np.vstack(
            [
                np.where(np.isfinite(screen_on), screen_on > 0.1, np.nan),
                np.where(np.isfinite(step_sum), step_sum > 0, np.nan),
                np.where(np.isfinite(still), still < 0.8, np.nan),
                np.where(np.isfinite(dark), dark < 0.5, np.nan),
                np.where(np.isfinite(usage_apps), usage_apps > 0, np.nan),
            ]
        ),
        axis=0,
    )
    disturbance = np.where(np.isfinite(disturbance), disturbance, 0.0)

    screen_on_bool = np.where(np.isfinite(screen_on), screen_on > 0.1, False)
    active_bool = np.where(np.isfinite(step_sum), step_sum > 0, False) | np.where(np.isfinite(still), still < 0.5, False)
    bright_bool = np.where(np.isfinite(dark), dark < 0.5, False)

    before = np.arange(0, best_start)
    last_screen = before[screen_on_bool[before]][-1] if len(before) and screen_on_bool[before].any() else np.nan
    last_active = before[active_bool[before]][-1] if len(before) and active_bool[before].any() else np.nan
    last_bright = before[bright_bool[before]][-1] if len(before) and bright_bool[before].any() else np.nan

    duration_minutes = (best_end - best_start + 1) * freq_minutes
    estimated_sleep_minutes = float(np.sum(score_smooth[block_idx]) * freq_minutes)
    outside_mask = np.ones(n, dtype=bool)
    outside_mask[block_idx] = False
    outside_score = float(np.nanmean(score_smooth[outside_mask])) if outside_mask.any() else np.nan

    result = {
        "sleepblock_start_idx": float(best_start),
        "sleepblock_end_idx": float(best_end),
        "sleepblock_start_hour_from_18": float(best_start * freq_minutes / 60.0),
        "sleepblock_end_hour_from_18": float((best_end + 1) * freq_minutes / 60.0),
        "sleepblock_duration_minutes": float(duration_minutes),
        "sleepblock_estimated_sleep_minutes": estimated_sleep_minutes,
        "sleepblock_efficiency_proxy": float(estimated_sleep_minutes / max(duration_minutes, 1.0)),
        "sleepblock_score_mean": float(np.nanmean(score_smooth[block_idx])),
        "sleepblock_score_std": float(np.nanstd(score_smooth[block_idx])),
        "sleepblock_score_margin": float(np.nanmean(score_smooth[block_idx]) - outside_score) if np.isfinite(outside_score) else np.nan,
        "sleepblock_coverage_mean": float(np.nanmean(coverage[block_idx])),
        "sleepblock_disturbance_ratio": float(np.nanmean(disturbance[block_idx])),
        "sleepblock_inner_disturbance_ratio": float(np.nanmean(disturbance[inner_idx])),
        "sleepblock_waso_bins": float(np.sum((score_smooth[inner_idx] < 0.45) | (disturbance[inner_idx] > 0.5))),
        "sleepblock_screen_on_bins": float(np.sum(screen_on_bool[block_idx])),
        "sleepblock_active_bins": float(np.sum(active_bool[block_idx])),
        "sleepblock_bright_bins": float(np.sum(bright_bool[block_idx])),
        "sleepblock_longest_high_score_run": float(longest_true_run(score_smooth[block_idx] >= threshold) * freq_minutes),
        "sleepblock_fragmentation_index": float(np.sum(candidate[block_idx][1:] != candidate[block_idx][:-1])) if len(block_idx) > 1 else 0.0,
        "sleepblock_onset_from_last_screen_minutes": float((best_start - last_screen) * freq_minutes) if np.isfinite(last_screen) else np.nan,
        "sleepblock_onset_from_last_active_minutes": float((best_start - last_active) * freq_minutes) if np.isfinite(last_active) else np.nan,
        "sleepblock_onset_from_last_bright_minutes": float((best_start - last_bright) * freq_minutes) if np.isfinite(last_bright) else np.nan,
        "sleepblock_pre60_screen_on_ratio": float(np.nanmean(screen_on[pre_idx])) if len(pre_idx) else np.nan,
        "sleepblock_pre60_activity_ratio": float(np.nanmean(1.0 - still[pre_idx])) if len(pre_idx) else np.nan,
        "sleepblock_pre60_light_dark_ratio": float(np.nanmean(dark[pre_idx])) if len(pre_idx) else np.nan,
        "sleepblock_pre60_usage_app_sum": float(np.nansum(usage_apps[pre_idx])) if len(pre_idx) else np.nan,
        "sleepblock_sensor_coverage_all": float(np.nanmean(coverage)),
        "sleepblock_threshold": float(threshold),
    }
    return result


def summarize_sleep_episode_v2(group: pd.DataFrame, freq_minutes: int = 5) -> Dict[str, float]:
    g = group.sort_values("bin_idx").reset_index(drop=True)
    n = len(g)

    def arr(col: str) -> np.ndarray:
        return pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float) if col in g else np.full(n, np.nan)

    def safe_mean(values: np.ndarray) -> float:
        return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan

    def safe_sum(values: np.ndarray) -> float:
        return float(np.nansum(values)) if np.isfinite(values).any() else np.nan

    screen_on = arr("sleepblock_screen_on_mean")
    step_sum = arr("sleepblock_step_sum")
    speed = arr("sleepblock_speed_mean")
    still = arr("sleepblock_activity_still_ratio")
    charging = arr("sleepblock_charging_mean")
    usage_apps = arr("sleepblock_usage_app_count")
    usage_time = arr("sleepblock_usage_total_time_sum")
    calories = arr("sleepblock_calories_sum")
    hr_mean = arr("sleepblock_hr_mean")
    hr_std = arr("sleepblock_hr_std")
    m_dark = arr("sleepblock_mlight_dark_ratio")
    w_dark = arr("sleepblock_wlight_dark_ratio")
    m_bright = arr("sleepblock_mlight_bright_ratio")
    w_bright = arr("sleepblock_wlight_bright_ratio")

    screen_off = np.where(np.isfinite(screen_on), 1.0 - np.clip(screen_on, 0, 1), np.nan)
    zero_step = np.where(np.isfinite(step_sum), (step_sum <= 0).astype(float), np.nan)
    low_speed = np.where(np.isfinite(speed), (speed <= 0.1).astype(float), np.nan)
    still = np.where(np.isfinite(still), np.clip(still, 0, 1), np.nan)
    dark = np.nanmean(np.vstack([m_dark, w_dark]), axis=0)
    bright = np.nanmean(np.vstack([m_bright, w_bright]), axis=0)
    no_usage = np.where(np.isfinite(usage_apps), (usage_apps <= 0).astype(float), np.nan)
    charging = np.where(np.isfinite(charging), np.clip(charging, 0, 1), np.nan)
    subject_hr_median = np.nanmedian(hr_mean)
    low_hr = np.where(
        np.isfinite(hr_mean) & np.isfinite(subject_hr_median),
        (hr_mean <= subject_hr_median).astype(float),
        np.nan,
    )
    time_prior = g["is_core_sleep_time"].to_numpy(dtype=float)

    components = np.vstack([screen_off, zero_step, still, low_speed, dark, low_hr, no_usage, charging, time_prior])
    weights = np.asarray([2.2, 1.25, 1.25, 0.55, 1.10, 0.85, 1.00, 0.35, 0.50], dtype=float)
    valid = np.isfinite(components)
    weighted_sum = np.nansum(components * weights[:, None], axis=0)
    weight_sum = np.sum(valid * weights[:, None], axis=0)
    score = np.divide(weighted_sum, weight_sum, out=np.full(n, np.nan), where=weight_sum > 0)
    score = np.where(np.isfinite(score), score, 0.0)
    score_smooth = pd.Series(score).rolling(5, center=True, min_periods=1).mean().to_numpy()
    coverage = weight_sum / weights.sum()

    quiet_components = np.vstack(
        [
            np.where(np.isfinite(screen_off), screen_off >= 0.75, np.nan),
            np.where(np.isfinite(zero_step), zero_step >= 0.5, np.nan),
            np.where(np.isfinite(still), still >= 0.70, np.nan),
            np.where(np.isfinite(dark), dark >= 0.55, np.nan),
            np.where(np.isfinite(no_usage), no_usage >= 0.5, np.nan),
        ]
    )
    quiet_ratio = np.nanmean(quiet_components.astype(float), axis=0)
    quiet_ratio = np.where(np.isfinite(quiet_ratio), quiet_ratio, 0.0)

    core_mask = time_prior >= 0.5
    core_scores = score_smooth[core_mask & (coverage >= 0.20)]
    if len(core_scores):
        threshold = max(0.54, min(0.76, float(np.nanquantile(core_scores, 0.55))))
    else:
        threshold = max(0.54, min(0.76, float(np.nanquantile(score_smooth, 0.60))))

    candidate = (score_smooth >= threshold) & (coverage >= 0.22)
    candidate |= (quiet_ratio >= 0.70) & core_mask & (coverage >= 0.18)
    candidate = fill_short_gaps(candidate, max_gap=3)
    segments = extract_segments(candidate, min_len=max(6, int(60 / freq_minutes)))

    interval_score = score_smooth + 0.25 * quiet_ratio + 0.10 * time_prior - 0.10 * (1.0 - np.clip(coverage, 0, 1))
    if segments:
        def segment_value(seg: Tuple[int, int]) -> float:
            start, end = seg
            idx = np.arange(start, end + 1)
            duration_hours = len(idx) * freq_minutes / 60.0
            duration_penalty = max(0.0, duration_hours - 10.5) * 0.20 + max(0.0, 3.0 - duration_hours) * 0.35
            start_clock = float(g.loc[start, "clock_hour"])
            wake_clock = float(g.loc[end, "clock_hour"])
            early_start_penalty = 0.25 if 18.0 <= start_clock < 20.0 else 0.0
            late_wake_penalty = 0.25 if 11.0 <= wake_clock < 18.0 else 0.0
            return (
                safe_mean(interval_score[idx])
                + 0.04 * min(duration_hours, 8.5)
                - duration_penalty
                - early_start_penalty
                - late_wake_penalty
            )

        best_start, best_end = max(segments, key=segment_value)
        duration_bins = best_end - best_start + 1
        if duration_bins > int(12 * 60 / freq_minutes):
            original_start = best_start
            best_start, best_end = best_scored_interval(
                interval_score[best_start : best_end + 1],
                min_len=int(4 * 60 / freq_minutes),
                max_len=int(10.5 * 60 / freq_minutes),
            )
            best_start += original_start
            best_end += original_start
    else:
        best_start, best_end = best_scored_interval(
            interval_score,
            min_len=int(4 * 60 / freq_minutes),
            max_len=int(10.5 * 60 / freq_minutes),
        )

    block_idx = np.arange(best_start, best_end + 1)
    inner_idx = block_idx[(block_idx >= best_start + int(30 / freq_minutes)) & (block_idx <= best_end - int(30 / freq_minutes))]
    if len(inner_idx) == 0:
        inner_idx = block_idx
    pre30_idx = np.arange(max(0, best_start - int(30 / freq_minutes)), best_start)
    pre60_idx = np.arange(max(0, best_start - int(60 / freq_minutes)), best_start)
    pre120_idx = np.arange(max(0, best_start - int(120 / freq_minutes)), best_start)
    post60_idx = np.arange(best_end + 1, min(n, best_end + 1 + int(60 / freq_minutes)))

    screen_on_bool = np.where(np.isfinite(screen_on), screen_on > 0.1, False)
    active_bool = (
        np.where(np.isfinite(step_sum), step_sum > 0, False)
        | np.where(np.isfinite(speed), speed > 0.1, False)
        | np.where(np.isfinite(still), still < 0.60, False)
    )
    bright_bool = np.where(np.isfinite(bright), bright > 0.15, False) | np.where(np.isfinite(dark), dark < 0.45, False)
    usage_bool = np.where(np.isfinite(usage_apps), usage_apps > 0, False) | np.where(np.isfinite(usage_time), usage_time > 0, False)
    disturbance = np.nanmean(
        np.vstack([screen_on_bool.astype(float), active_bool.astype(float), bright_bool.astype(float), usage_bool.astype(float)]),
        axis=0,
    )
    disturbance = np.where(np.isfinite(disturbance), disturbance, 0.0)
    awake_inner = (score_smooth[inner_idx] < 0.48) | (disturbance[inner_idx] > 0.50)

    before = np.arange(0, best_start)
    last_screen = before[screen_on_bool[before]][-1] if len(before) and screen_on_bool[before].any() else np.nan
    last_active = before[active_bool[before]][-1] if len(before) and active_bool[before].any() else np.nan
    last_usage = before[usage_bool[before]][-1] if len(before) and usage_bool[before].any() else np.nan
    first_screen_after = post60_idx[screen_on_bool[post60_idx]][0] if len(post60_idx) and screen_on_bool[post60_idx].any() else np.nan
    first_active_after = post60_idx[active_bool[post60_idx]][0] if len(post60_idx) and active_bool[post60_idx].any() else np.nan

    start_hour_from_18 = best_start * freq_minutes / 60.0
    end_hour_from_18 = (best_end + 1) * freq_minutes / 60.0
    duration_minutes = (best_end - best_start + 1) * freq_minutes
    estimated_sleep_minutes = float(np.sum(np.clip(score_smooth[block_idx], 0, 1)) * freq_minutes)
    hr_block = hr_mean[block_idx]
    hr_pre = hr_mean[pre60_idx] if len(pre60_idx) else np.asarray([], dtype=float)

    result = {
        "sleepv2_start_idx": float(best_start),
        "sleepv2_end_idx": float(best_end),
        "sleepv2_start_hour_from_18": float(start_hour_from_18),
        "sleepv2_end_hour_from_18": float(end_hour_from_18),
        "sleepv2_midpoint_hour_from_18": float((start_hour_from_18 + end_hour_from_18) / 2.0),
        "sleepv2_start_clock_hour": float(g.loc[best_start, "clock_hour"]),
        "sleepv2_wake_clock_hour": float((18.0 + end_hour_from_18) % 24.0),
        "sleepv2_duration_minutes": float(duration_minutes),
        "sleepv2_estimated_sleep_minutes": estimated_sleep_minutes,
        "sleepv2_efficiency_proxy": float(estimated_sleep_minutes / max(duration_minutes, 1.0)),
        "sleepv2_score_mean": safe_mean(score_smooth[block_idx]),
        "sleepv2_score_std": float(np.nanstd(score_smooth[block_idx])) if np.isfinite(score_smooth[block_idx]).any() else np.nan,
        "sleepv2_score_q10": float(np.nanquantile(score_smooth[block_idx], 0.10)) if np.isfinite(score_smooth[block_idx]).any() else np.nan,
        "sleepv2_score_q90": float(np.nanquantile(score_smooth[block_idx], 0.90)) if np.isfinite(score_smooth[block_idx]).any() else np.nan,
        "sleepv2_quiet_ratio_mean": safe_mean(quiet_ratio[block_idx]),
        "sleepv2_coverage_mean": safe_mean(coverage[block_idx]),
        "sleepv2_disturbance_ratio": safe_mean(disturbance[block_idx]),
        "sleepv2_inner_disturbance_ratio": safe_mean(disturbance[inner_idx]),
        "sleepv2_waso_minutes": float(np.sum(awake_inner) * freq_minutes),
        "sleepv2_awake_episode_count": float(count_true_segments(awake_inner)),
        "sleepv2_longest_awake_minutes": float(longest_true_run(awake_inner) * freq_minutes),
        "sleepv2_screen_on_minutes": float(np.sum(screen_on_bool[block_idx]) * freq_minutes),
        "sleepv2_screen_unlock_count": float(count_true_segments(screen_on_bool[inner_idx])),
        "sleepv2_active_minutes": float(np.sum(active_bool[block_idx]) * freq_minutes),
        "sleepv2_active_burst_count": float(count_true_segments(active_bool[inner_idx])),
        "sleepv2_bright_minutes": float(np.sum(bright_bool[block_idx]) * freq_minutes),
        "sleepv2_bright_burst_count": float(count_true_segments(bright_bool[inner_idx])),
        "sleepv2_usage_minutes_proxy": float(np.sum(usage_bool[block_idx]) * freq_minutes),
        "sleepv2_usage_burst_count": float(count_true_segments(usage_bool[inner_idx])),
        "sleepv2_step_sum": safe_sum(step_sum[block_idx]),
        "sleepv2_calories_sum": safe_sum(calories[block_idx]),
        "sleepv2_charging_ratio": safe_mean(charging[block_idx]),
        "sleepv2_hr_mean": safe_mean(hr_block),
        "sleepv2_hr_std_mean": safe_mean(hr_std[block_idx]),
        "sleepv2_hr_min": float(np.nanmin(hr_block)) if np.isfinite(hr_block).any() else np.nan,
        "sleepv2_hr_max": float(np.nanmax(hr_block)) if np.isfinite(hr_block).any() else np.nan,
        "sleepv2_hr_pre60_minus_sleep": safe_mean(hr_pre) - safe_mean(hr_block) if len(hr_pre) else np.nan,
        "sleepv2_onset_from_last_screen_minutes": float((best_start - last_screen) * freq_minutes) if np.isfinite(last_screen) else np.nan,
        "sleepv2_onset_from_last_active_minutes": float((best_start - last_active) * freq_minutes) if np.isfinite(last_active) else np.nan,
        "sleepv2_onset_from_last_usage_minutes": float((best_start - last_usage) * freq_minutes) if np.isfinite(last_usage) else np.nan,
        "sleepv2_wake_to_first_screen_minutes": float((first_screen_after - best_end) * freq_minutes) if np.isfinite(first_screen_after) else np.nan,
        "sleepv2_wake_to_first_active_minutes": float((first_active_after - best_end) * freq_minutes) if np.isfinite(first_active_after) else np.nan,
        "sleepv2_pre30_screen_on_ratio": safe_mean(screen_on[pre30_idx]) if len(pre30_idx) else np.nan,
        "sleepv2_pre60_screen_on_ratio": safe_mean(screen_on[pre60_idx]) if len(pre60_idx) else np.nan,
        "sleepv2_pre120_screen_on_ratio": safe_mean(screen_on[pre120_idx]) if len(pre120_idx) else np.nan,
        "sleepv2_pre60_active_ratio": safe_mean(active_bool[pre60_idx].astype(float)) if len(pre60_idx) else np.nan,
        "sleepv2_pre60_dark_ratio": safe_mean(dark[pre60_idx]) if len(pre60_idx) else np.nan,
        "sleepv2_pre60_usage_time_sum": safe_sum(usage_time[pre60_idx]) if len(pre60_idx) else np.nan,
        "sleepv2_post60_screen_on_ratio": safe_mean(screen_on[post60_idx]) if len(post60_idx) else np.nan,
        "sleepv2_post60_active_ratio": safe_mean(active_bool[post60_idx].astype(float)) if len(post60_idx) else np.nan,
        "sleepv2_post60_dark_ratio": safe_mean(dark[post60_idx]) if len(post60_idx) else np.nan,
        "sleepv2_threshold": float(threshold),
        "sleepv2_candidate_segment_count": float(len(segments)),
        "sleepv2_sensor_coverage_all": safe_mean(coverage),
    }
    return result


def process_sleep_block_features(data_dir: Path, base: pd.DataFrame) -> pd.DataFrame:
    grid = sleep_block_aggregates(data_dir, base, freq_minutes=5)
    rows = []
    for (subject_id, date), group in grid.groupby(KEYS, sort=False):
        row = {"subject_id": subject_id, "date": date}
        row.update(summarize_sleep_block(group, freq_minutes=5))
        rows.append(row)
    features = pd.DataFrame(rows)
    if features.empty:
        return pd.DataFrame(columns=KEYS)

    for col in [
        "sleepblock_duration_minutes",
        "sleepblock_estimated_sleep_minutes",
        "sleepblock_efficiency_proxy",
        "sleepblock_start_hour_from_18",
        "sleepblock_end_hour_from_18",
        "sleepblock_disturbance_ratio",
        "sleepblock_waso_bins",
    ]:
        if col in features:
            med = features.groupby("subject_id")[col].transform("median")
            features[f"{col}_subject_diff"] = features[col] - med
    return features


def process_sleep_episode_v2_features(data_dir: Path, base: pd.DataFrame) -> pd.DataFrame:
    grid = sleep_block_aggregates(data_dir, base, freq_minutes=5)
    rows = []
    for (subject_id, date), group in grid.groupby(KEYS, sort=False):
        row = {"subject_id": subject_id, "date": date}
        row.update(summarize_sleep_episode_v2(group, freq_minutes=5))
        rows.append(row)
    features = pd.DataFrame(rows)
    if features.empty:
        return pd.DataFrame(columns=KEYS)

    regularity_cols = [
        "sleepv2_start_hour_from_18",
        "sleepv2_end_hour_from_18",
        "sleepv2_midpoint_hour_from_18",
        "sleepv2_duration_minutes",
        "sleepv2_efficiency_proxy",
        "sleepv2_waso_minutes",
        "sleepv2_disturbance_ratio",
        "sleepv2_screen_on_minutes",
        "sleepv2_active_minutes",
        "sleepv2_hr_mean",
    ]
    for col in regularity_cols:
        if col not in features:
            continue
        value = pd.to_numeric(features[col], errors="coerce")
        med = value.groupby(features["subject_id"]).transform("median")
        q25 = value.groupby(features["subject_id"]).transform(lambda x: x.quantile(0.25))
        q75 = value.groupby(features["subject_id"]).transform(lambda x: x.quantile(0.75))
        iqr = (q75 - q25).replace(0, np.nan)
        features[f"{col}_subject_diff"] = value - med
        features[f"{col}_subject_absdiff"] = (value - med).abs()
        features[f"{col}_subject_z"] = (value - med) / iqr
    return features


def add_cross_sleep_proxy_features(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()

    def col(name: str) -> pd.Series:
        return pd.to_numeric(out[name], errors="coerce") if name in out.columns else pd.Series(np.nan, index=out.index)

    out["sleep_proxy_quiet_ratio"] = np.nanmean(
        np.vstack(
            [
                col("mScreenStatus_screen_off_episode_ratio"),
                col("wPedo_stationary_episode_ratio"),
                col("mActivity_still_episode_ratio"),
                col("mLight_dark_episode_ratio"),
                col("wLight_dark_episode_ratio"),
            ]
        ),
        axis=0,
    )
    out["sleep_proxy_disturbance_ratio"] = np.nanmean(
        np.vstack(
            [
                col("mScreenStatus_screen_on_episode_ratio"),
                col("wPedo_moving_episode_ratio"),
                col("mActivity_active_episode_ratio"),
                col("mLight_bright_episode_ratio"),
                col("wLight_bright_episode_ratio"),
            ]
        ),
        axis=0,
    )
    out["sleep_proxy_rest_run"] = np.nanmean(
        np.vstack(
            [
                col("mScreenStatus_screen_off_episode_longest_run"),
                col("wPedo_stationary_episode_longest_run"),
                col("mActivity_still_episode_longest_run"),
                col("mLight_dark_episode_longest_run"),
                col("wLight_dark_episode_longest_run"),
            ]
        ),
        axis=0,
    )
    out["sleep_proxy_screen_after_midnight"] = col("mScreenStatus_episode_night_m_screen_use_sum")
    out["sleep_proxy_movement_after_midnight"] = col("wPedo_episode_night_step_sum") + col("mActivity_episode_night_m_activity_sum")
    return out


def add_subject_relative_features(
    features: pd.DataFrame,
    max_subject_features: int = 260,
    max_rolling_features: int = 90,
    include_rolling: bool = True,
) -> pd.DataFrame:
    out = features.copy()
    drop_cols = set(TARGETS + ["is_train", "row_id", "sleep_date", "lifelog_date", "date"])
    candidate_cols = [c for c in out.columns if c not in drop_cols and c != "subject_id"]

    numeric = out[candidate_cols].apply(pd.to_numeric, errors="coerce")
    valid_cols = []
    for c in numeric.columns:
        s = numeric[c]
        if s.notna().sum() >= 200 and s.nunique(dropna=True) >= 8:
            valid_cols.append(c)

    if not valid_cols:
        return out

    variances = numeric[valid_cols].var(skipna=True).replace([np.inf, -np.inf], np.nan).fillna(0)
    selected = variances.sort_values(ascending=False).head(max_subject_features).index.tolist()
    subject_group = out["subject_id"]
    relative_parts: Dict[str, pd.Series] = {}

    for c in selected:
        s = numeric[c]
        med = s.groupby(subject_group).transform("median")
        q75 = s.groupby(subject_group).transform(lambda x: x.quantile(0.75))
        q25 = s.groupby(subject_group).transform(lambda x: x.quantile(0.25))
        iqr = (q75 - q25).replace(0, np.nan)
        relative_parts[f"{c}_subj_diff"] = s - med
        relative_parts[f"{c}_subj_z"] = (s - med) / iqr

    if relative_parts:
        out = pd.concat([out, pd.DataFrame(relative_parts, index=out.index)], axis=1).copy()

    if not include_rolling:
        return out

    rolling_pool = [
        c
        for c in selected
        if any(token in c.lower() for token in ["episode", "screen", "charging", "activity", "step", "light", "heart", "sleep_proxy"])
    ][:max_rolling_features]
    ordered = out.sort_values(["subject_id", "lifelog_date"]).copy()
    rolling_parts: Dict[str, pd.Series] = {}
    for c in rolling_pool:
        s = pd.to_numeric(ordered[c], errors="coerce")
        shifted = s.groupby(ordered["subject_id"]).shift(1)
        rolling_parts[f"{c}_prev_diff"] = s - shifted
        rolling_parts[f"{c}_roll3_mean"] = shifted.groupby(ordered["subject_id"]).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        rolling_parts[f"{c}_roll7_mean"] = shifted.groupby(ordered["subject_id"]).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)

    if rolling_parts:
        ordered = pd.concat([ordered, pd.DataFrame(rolling_parts, index=ordered.index)], axis=1).copy()

    added_cols = [c for c in ordered.columns if c not in out.columns]
    if added_cols:
        out = out.merge(ordered[["subject_id", "lifelog_date", "row_id", "is_train"] + added_cols],
                        on=["subject_id", "lifelog_date", "row_id", "is_train"], how="left")
    return out


def build_features(
    data_dir: Path,
    top_n: int,
    feature_route: str = "full",
    disable_subject_rolling: bool = False,
    disable_raw_id_features: bool = False,
    stable_high_card_summary_only: bool = False,
    enable_sleep_v2: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, sample, combined = load_tables(data_dir)
    base = add_calendar_features(combined)
    keys = base[KEYS].drop_duplicates()
    include_subject_rolling = not disable_subject_rolling and feature_route not in {"norelroll", "stable"}
    include_raw_id_features = not disable_raw_id_features and not stable_high_card_summary_only

    print("Building sensor features...")
    frames = [
        process_scalar_status(data_dir, keys, "ch2025_mACStatus.parquet", "m_charging", "mACStatus", True),
        process_scalar_status(data_dir, keys, "ch2025_mActivity.parquet", "m_activity", "mActivity", True),
        process_scalar_status(data_dir, keys, "ch2025_mScreenStatus.parquet", "m_screen_use", "mScreenStatus", True),
        process_light(data_dir, keys, "ch2025_mLight.parquet", "m_light", "mLight"),
        process_light(data_dir, keys, "ch2025_wLight.parquet", "w_light", "wLight"),
        process_pedo(data_dir, keys),
        process_hr(data_dir, keys),
        process_gps(data_dir, keys),
        process_wifi(data_dir, keys, top_n=top_n, include_raw_id_features=include_raw_id_features),
        process_ble(data_dir, keys, top_n=top_n, include_raw_id_features=include_raw_id_features),
        process_ambience(data_dir, keys, top_n=top_n),
        process_usage(data_dir, keys, top_n=top_n, include_raw_id_features=include_raw_id_features),
        process_episode_scalar(data_dir, base, "ch2025_mACStatus.parquet", "m_charging", "mACStatus", True),
        process_episode_scalar(data_dir, base, "ch2025_mActivity.parquet", "m_activity", "mActivity", True),
        process_episode_scalar(data_dir, base, "ch2025_mScreenStatus.parquet", "m_screen_use", "mScreenStatus", True),
        process_episode_light(data_dir, base, "ch2025_mLight.parquet", "m_light", "mLight"),
        process_episode_light(data_dir, base, "ch2025_wLight.parquet", "w_light", "wLight"),
        process_episode_pedo(data_dir, base),
        process_episode_hr(data_dir, base),
        process_sleep_block_features(data_dir, base),
    ]
    if enable_sleep_v2:
        frames.append(process_sleep_episode_v2_features(data_dir, base))

    features = merge_feature_frames(base, frames)
    features = add_cross_sleep_proxy_features(features)
    features = add_subject_relative_features(features, include_rolling=include_subject_rolling)
    return train, sample, features


def make_design_matrix(features: pd.DataFrame) -> pd.DataFrame:
    drop_cols = set(
        TARGETS
        + [
            "is_train",
            "row_id",
            "sleep_date",
            "lifelog_date",
            "date",
        ]
    )
    numeric_cols = [col for col in features.columns if col not in drop_cols and col != "subject_id"]
    numeric = features[numeric_cols].copy()
    for col in numeric.columns:
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")

    subjects = pd.get_dummies(features["subject_id"], prefix="subject", dtype=float)
    X = pd.concat([numeric, subjects], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    X.columns = [clean_token(c, max_len=96) for c in X.columns]
    return X


def target_feature_tokens(target: str) -> List[str]:
    common = [
        "lifelog_",
        "sleep_dow",
        "elapsed_day",
        "subject_num",
        "subject_seq",
        "subject_elapsed",
        "subject_",
        "te_subject_",
        "time_",
    ]
    sleep_tokens = [
        "sleepblock",
        "sleepv2",
        "sleep_proxy",
        "episode",
        "mscreenstatus",
        "wpedo",
        "mactivity",
        "mlight",
        "wlight",
        "whr",
        "macstatus",
    ]
    behavior_tokens = [
        "mscreenstatus",
        "mactivity",
        "wpedo",
        "mlight",
        "wlight",
        "whr",
        "musagestats_total_time",
        "musagestats_app_nunique",
        "musagestats_app_count",
        "mgps",
        "mambience_prob",
        "mambience_label_nunique",
        "mambience_prob_entropy",
        "mwifi_rssi",
        "mwifi_strong_signal",
        "mwifi_bssid_nunique",
        "mble_rssi",
        "mble_strong_signal",
        "mble_address_nunique",
        "macstatus",
    ]
    if target.startswith("S"):
        return common + sleep_tokens
    if target == "Q1":
        return common + sleep_tokens + behavior_tokens
    return common + behavior_tokens


def is_high_cardinality_feature(col: str) -> bool:
    name = col.lower()
    raw_markers = [
        "musagestats_app_name_",
        "mwifi_bssid_",
        "mble_address_",
        "mambience_label_",
        "mambience_top_label_",
    ]
    return any(marker in name for marker in raw_markers)


def select_target_design_matrix(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    target: str,
    mode: str,
    max_features: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    if mode == "all":
        cols = list(X_train.columns)
        return X_train, X_test, cols

    tokens = target_feature_tokens(target)
    selected = [
        col
        for col in X_train.columns
        if any(token in col.lower() for token in tokens) and not is_high_cardinality_feature(col)
    ]
    if not selected:
        selected = list(X_train.columns)

    always_keep = [
        col
        for col in selected
        if col.startswith("subject_")
        or col.startswith("te_subject_")
        or col.startswith("time_")
        or any(token in col for token in ["lifelog_", "sleep_dow", "elapsed_day", "subject_seq", "subject_elapsed"])
    ]
    max_features = max(50, int(max_features))
    if len(selected) > max_features:
        numeric = X_train[selected].apply(pd.to_numeric, errors="coerce")
        variances = numeric.var(skipna=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        keep_set = set(always_keep)
        remaining = [col for col in variances.sort_values(ascending=False).index if col not in keep_set]
        selected = always_keep + remaining[: max(0, max_features - len(always_keep))]

    selected = [col for col in selected if col in X_train.columns]
    return X_train[selected].copy(), X_test[selected].copy(), selected


def make_model_specs(seed: int, n_jobs: int, diagnostic: bool = False) -> List[Tuple[str, Pipeline]]:
    tree_estimators = 90 if diagnostic else 180
    forest_estimators = 0 if diagnostic else 120
    hist_iter = 0 if diagnostic else 120

    specs: List[Tuple[str, Pipeline]] = [
        (
            "logistic",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=0.8,
                            max_iter=2500,
                            class_weight="balanced",
                            random_state=seed,
                            solver="lbfgs",
                        ),
                    ),
                ]
            ),
        ),
        (
            "extra_trees",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    (
                        "model",
                        ExtraTreesClassifier(
                            n_estimators=tree_estimators,
                            min_samples_leaf=3,
                            max_features="sqrt",
                            class_weight="balanced_subsample",
                            random_state=seed,
                            n_jobs=n_jobs,
                        ),
                    ),
                ]
            ),
        )
    ]

    if forest_estimators:
        specs.append(
        (
            "random_forest",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=forest_estimators,
                            min_samples_leaf=4,
                            max_features="sqrt",
                            class_weight="balanced_subsample",
                            random_state=seed + 17,
                            n_jobs=n_jobs,
                        ),
                    ),
                ]
            ),
        ))

    if hist_iter:
        specs.append(
        (
            "hist_gb",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            max_iter=hist_iter,
                            learning_rate=0.04,
                            max_leaf_nodes=15,
                            l2_regularization=0.08,
                            random_state=seed + 31,
                        ),
                    ),
                ]
            ),
        ))

    if LGBMClassifier is not None:
        specs.append(
            (
                "lightgbm",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "model",
                            LGBMClassifier(
                                n_estimators=90 if diagnostic else 180,
                                learning_rate=0.035,
                                num_leaves=7,
                                min_child_samples=12,
                                subsample=0.85,
                                colsample_bytree=0.70,
                                reg_alpha=0.05,
                                reg_lambda=2.0,
                                class_weight="balanced",
                                random_state=seed + 53,
                                n_jobs=n_jobs,
                                verbose=-1,
                            ),
                        ),
                    ]
                ),
            )
        )

    return specs


def predict_positive(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    pred = model.predict_proba(X)
    classes = model.named_steps["model"].classes_
    if len(classes) == 1:
        return np.ones(len(X)) if int(classes[0]) == 1 else np.zeros(len(X))
    pos_idx = int(np.where(classes == 1)[0][0])
    return pred[:, pos_idx]


def smoothed_subject_means(
    train_subjects: np.ndarray,
    y: np.ndarray,
    apply_subjects: np.ndarray,
    smoothing: float = 5.0,
) -> np.ndarray:
    prior = float(np.mean(y))
    stats = pd.DataFrame({"subject_id": train_subjects, "y": y}).groupby("subject_id")["y"].agg(["sum", "count"])
    encoded = []
    for subject in apply_subjects:
        if subject in stats.index:
            total = float(stats.loc[subject, "sum"])
            count = float(stats.loc[subject, "count"])
            encoded.append((total + prior * smoothing) / (count + smoothing))
        else:
            encoded.append(prior)
    return np.asarray(encoded, dtype=float)


def loo_subject_encoding(
    subjects: np.ndarray,
    y: np.ndarray,
    smoothing: float = 5.0,
) -> np.ndarray:
    prior = float(np.mean(y))
    stats = pd.DataFrame({"subject_id": subjects, "y": y}).groupby("subject_id")["y"].agg(["sum", "count"])
    encoded = np.empty(len(y), dtype=float)
    for i, subject in enumerate(subjects):
        total = float(stats.loc[subject, "sum"])
        count = float(stats.loc[subject, "count"])
        denom = count - 1.0 + smoothing
        encoded[i] = prior if denom <= 0 else (total - y[i] + prior * smoothing) / denom
    return encoded


def weighted_time_label_mean(
    ref_subjects: np.ndarray,
    ref_dates: np.ndarray,
    ref_y: np.ndarray,
    query_subjects: np.ndarray,
    query_dates: np.ndarray,
    prior: float,
    k: int = 8,
    power: float = 1.0,
    smoothing: float = 4.0,
) -> np.ndarray:
    ref_dates = pd.to_datetime(ref_dates)
    query_dates = pd.to_datetime(query_dates)
    ref_days = pd.Series(ref_dates).astype("int64").to_numpy() / 86_400_000_000_000
    query_days = pd.Series(query_dates).astype("int64").to_numpy() / 86_400_000_000_000
    out = np.empty(len(query_subjects), dtype=float)

    by_subject: Dict[str, np.ndarray] = {}
    for subject in np.unique(ref_subjects):
        by_subject[str(subject)] = np.where(ref_subjects == subject)[0]

    for i, subject in enumerate(query_subjects):
        idx = by_subject.get(str(subject))
        if idx is None or len(idx) == 0:
            out[i] = prior
            continue
        dist = np.abs(ref_days[idx] - query_days[i])
        order = np.argsort(dist)[: min(k, len(idx))]
        chosen = idx[order]
        weights = 1.0 / np.power(dist[order] + 1.0, power)
        local = float(np.sum(ref_y[chosen] * weights) / np.sum(weights))
        out[i] = (local * len(chosen) + prior * smoothing) / (len(chosen) + smoothing)
    return out


def loo_time_label_encoding(
    subjects: np.ndarray,
    dates: np.ndarray,
    y: np.ndarray,
    k: int = 8,
    power: float = 1.0,
    smoothing: float = 4.0,
) -> np.ndarray:
    prior = float(np.mean(y))
    dates = pd.to_datetime(dates)
    days = pd.Series(dates).astype("int64").to_numpy() / 86_400_000_000_000
    out = np.empty(len(y), dtype=float)

    for i, subject in enumerate(subjects):
        idx = np.where((subjects == subject) & (np.arange(len(y)) != i))[0]
        if len(idx) == 0:
            out[i] = prior
            continue
        dist = np.abs(days[idx] - days[i])
        order = np.argsort(dist)[: min(k, len(idx))]
        chosen = idx[order]
        weights = 1.0 / np.power(dist[order] + 1.0, power)
        local = float(np.sum(y[chosen] * weights) / np.sum(weights))
        out[i] = (local * len(chosen) + prior * smoothing) / (len(chosen) + smoothing)
    return out


def temporal_component_names(target: str) -> List[str]:
    names = TIME_COMPONENT_RECIPES.get(target, ["default"])
    return list(dict.fromkeys(names))


def temporal_label_components(
    ref_subjects: np.ndarray,
    ref_dates: np.ndarray,
    ref_y: np.ndarray,
    query_subjects: np.ndarray,
    query_dates: np.ndarray,
    target: str,
    prior: Optional[float] = None,
    exclude_self: bool = False,
) -> Dict[str, np.ndarray]:
    prior = float(np.mean(ref_y)) if prior is None else float(prior)
    names = temporal_component_names(target)
    ref_dates = pd.to_datetime(ref_dates)
    query_dates = pd.to_datetime(query_dates)
    ref_days = pd.Series(ref_dates).astype("int64").to_numpy() / 86_400_000_000_000
    query_days = pd.Series(query_dates).astype("int64").to_numpy() / 86_400_000_000_000
    out = {name: np.full(len(query_subjects), prior, dtype=float) for name in names}

    if "default" in out:
        params = TIME_LABEL_PARAMS.get(target, {"k": 8, "power": 1.0, "smoothing": 4.0})
        if exclude_self:
            out["default"] = loo_time_label_encoding(ref_subjects, ref_dates, ref_y, **params)
        else:
            out["default"] = weighted_time_label_mean(
                ref_subjects,
                ref_dates,
                ref_y,
                query_subjects,
                query_dates,
                prior=prior,
                **params,
            )

    abs_specs = {
        "abs_k2": {"k": 2, "power": 0.5, "smoothing": 4.0},
        "abs_k8": {"k": 8, "power": 0.5, "smoothing": 4.0},
        "abs_k40": {"k": 40, "power": 0.5, "smoothing": 4.0},
        "abs_k80": {"k": 80, "power": 0.5, "smoothing": 4.0},
    }
    for name, params in abs_specs.items():
        if name not in out:
            continue
        if exclude_self:
            out[name] = loo_time_label_encoding(ref_subjects, ref_dates, ref_y, **params)
        else:
            out[name] = weighted_time_label_mean(
                ref_subjects,
                ref_dates,
                ref_y,
                query_subjects,
                query_dates,
                prior=prior,
                **params,
            )

    if "subject" in out:
        out["subject"] = (
            loo_subject_encoding(ref_subjects, ref_y)
            if exclude_self
            else smoothed_subject_means(ref_subjects, ref_y, query_subjects)
        )

    directional_names = {"nearest", "prev", "next", "prev_next", "linear", "run_same"} & set(out)
    if not directional_names:
        return {name: clip_prob(values) for name, values in out.items()}

    by_subject: Dict[str, np.ndarray] = {}
    for subject in np.unique(ref_subjects):
        by_subject[str(subject)] = np.where(ref_subjects == subject)[0]

    for i, subject in enumerate(query_subjects):
        idx = by_subject.get(str(subject))
        if idx is None or len(idx) == 0:
            continue
        if exclude_self:
            idx = idx[idx != i]
            if len(idx) == 0:
                continue
        dist = np.abs(ref_days[idx] - query_days[i])
        nearest_idx = idx[int(np.argmin(dist))]
        if "nearest" in out:
            out["nearest"][i] = (float(ref_y[nearest_idx]) + prior) / 2.0

        prev_idx = idx[ref_days[idx] < query_days[i]]
        next_idx = idx[ref_days[idx] > query_days[i]]
        prev_label: Optional[float] = None
        next_label: Optional[float] = None
        prev_dist: Optional[float] = None
        next_dist: Optional[float] = None
        if len(prev_idx):
            chosen = prev_idx[int(np.argmax(ref_days[prev_idx]))]
            prev_label = float(ref_y[chosen])
            prev_dist = float(query_days[i] - ref_days[chosen])
            if "prev" in out:
                out["prev"][i] = (prev_label + prior) / 2.0
        if len(next_idx):
            chosen = next_idx[int(np.argmin(ref_days[next_idx]))]
            next_label = float(ref_y[chosen])
            next_dist = float(ref_days[chosen] - query_days[i])
            if "next" in out:
                out["next"][i] = (next_label + prior) / 2.0

        if prev_label is not None and next_label is not None:
            inv_prev = 1.0 / (float(prev_dist) + 1.0)
            inv_next = 1.0 / (float(next_dist) + 1.0)
            local = (prev_label * inv_prev + next_label * inv_next) / (inv_prev + inv_next)
            if "prev_next" in out:
                out["prev_next"][i] = (2.0 * local + 2.0 * prior) / 4.0
            total_dist = float(prev_dist) + float(next_dist)
            linear = (
                (next_label * float(prev_dist) + prev_label * float(next_dist)) / total_dist
                if total_dist > 0
                else 0.5 * (prev_label + next_label)
            )
            if "linear" in out:
                out["linear"][i] = (2.0 * linear + 2.0 * prior) / 4.0
            if "run_same" in out and prev_label == next_label:
                out["run_same"][i] = 0.85 if prev_label == 1.0 else 0.15
        elif prev_label is not None:
            fallback = (prev_label + 1.5 * prior) / 2.5
            if "prev_next" in out:
                out["prev_next"][i] = fallback
            if "linear" in out:
                out["linear"][i] = fallback
        elif next_label is not None:
            fallback = (next_label + 1.5 * prior) / 2.5
            if "prev_next" in out:
                out["prev_next"][i] = fallback
            if "linear" in out:
                out["linear"][i] = fallback

    return {name: clip_prob(values) for name, values in out.items()}


def optimize_blend_weights(
    y: np.ndarray,
    model_pred: np.ndarray,
    subject_pred: np.ndarray,
    time_pred: np.ndarray,
    max_time_neighbor: float = 1.0,
    min_model: float = 0.0,
) -> Tuple[np.ndarray, float]:
    best_weights = np.asarray([0.70, 0.15, 0.15], dtype=float)
    best_loss = math.inf
    grid = np.linspace(0.0, 1.0, 21)
    for w_model in grid:
        if w_model + 1e-9 < min_model:
            continue
        for w_subject in grid:
            w_time = 1.0 - w_model - w_subject
            if w_time < -1e-9:
                continue
            if w_time > max_time_neighbor + 1e-9:
                continue
            pred = clip_prob(w_model * model_pred + w_subject * subject_pred + w_time * time_pred)
            loss = float(log_loss(y, pred, labels=[0, 1]))
            if loss < best_loss:
                best_loss = loss
                best_weights = np.asarray([w_model, w_subject, max(0.0, w_time)], dtype=float)
    return best_weights, best_loss


def blend_component_predictions(components: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    return clip_prob(
        weights.get("model", 0.0) * components["model"]
        + weights.get("subject", 0.0) * components["subject"]
        + weights.get("time_neighbor", 0.0) * components["time_neighbor"]
    )


def shrink_to_prior(pred: np.ndarray, prior: float, prior_weight: float) -> np.ndarray:
    return clip_prob((1.0 - prior_weight) * pred + prior_weight * prior)


def optimize_prior_shrink(
    y: np.ndarray,
    pred: np.ndarray,
    max_prior_weight: float = 0.30,
) -> Tuple[float, float]:
    prior = float(np.mean(y))
    best_weight = 0.0
    best_loss = math.inf
    for prior_weight in np.linspace(0.0, max_prior_weight, 31):
        shrunk = shrink_to_prior(pred, prior, float(prior_weight))
        loss = float(log_loss(y, shrunk, labels=[0, 1]))
        if loss < best_loss:
            best_loss = loss
            best_weight = float(prior_weight)
    return best_weight, best_loss


def optimize_two_way_blend(
    y: np.ndarray,
    base_pred: np.ndarray,
    alt_pred: np.ndarray,
    max_alt_weight: float = 1.0,
) -> Tuple[float, float]:
    best_weight = 0.0
    best_loss = math.inf
    for alt_weight in np.linspace(0.0, max_alt_weight, 41):
        pred = clip_prob((1.0 - alt_weight) * base_pred + alt_weight * alt_pred)
        loss = float(log_loss(y, pred, labels=[0, 1]))
        if loss < best_loss:
            best_loss = loss
            best_weight = float(alt_weight)
    return best_weight, best_loss


def weights_to_dict(weights: np.ndarray) -> Dict[str, float]:
    return {
        "model": float(weights[0]),
        "subject": float(weights[1]),
        "time_neighbor": float(weights[2]),
    }


def stratified_splits(y: np.ndarray, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    counts = np.bincount(y.astype(int), minlength=2)
    min_count = int(counts.min())
    if min_count >= 2:
        n_splits = min(5, min_count)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(y)), y))
    splitter = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y))))


def blocked_time_splits(
    subjects: np.ndarray,
    dates: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    gap_days: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    dates = pd.to_datetime(dates)
    day_values = pd.Series(dates).astype("int64").to_numpy() / 86_400_000_000_000
    indices = np.arange(len(y))
    chunks_by_subject: Dict[str, List[np.ndarray]] = {}
    for subject in np.unique(subjects):
        idx = indices[subjects == subject]
        idx = idx[np.argsort(day_values[idx])]
        chunks_by_subject[str(subject)] = [chunk for chunk in np.array_split(idx, n_splits) if len(chunk)]

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_splits):
        valid_parts = []
        for subject, chunks in chunks_by_subject.items():
            if fold < len(chunks):
                valid_parts.append(chunks[fold])
        if not valid_parts:
            continue
        valid_idx = np.concatenate(valid_parts)
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[valid_idx] = False
        for subject in np.unique(subjects[valid_idx]):
            subject_valid = valid_idx[subjects[valid_idx] == subject]
            if len(subject_valid) == 0:
                continue
            subject_idx = indices[subjects == subject]
            min_day = float(np.min(day_values[subject_valid]))
            max_day = float(np.max(day_values[subject_valid]))
            gap_mask = (
                (day_values[subject_idx] >= min_day - gap_days)
                & (day_values[subject_idx] <= max_day + gap_days)
            )
            train_mask[subject_idx[gap_mask]] = False
        train_idx = indices[train_mask]
        if len(train_idx) == 0 or len(valid_idx) == 0:
            continue
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[valid_idx])) < 2:
            continue
        splits.append((train_idx, valid_idx))
    return splits


def average_model_predictions(
    specs: List[Tuple[str, Pipeline]],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, np.ndarray]]:
    preds_by_name: Dict[str, np.ndarray] = {}
    model_weights: Dict[str, float] = {}
    for name, spec in specs:
        model = clone(spec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
            p = clip_prob(predict_positive(model, X_valid))
        preds_by_name[name] = p
        model_weights[name] = 1.0 / len(specs)
    return np.mean(list(preds_by_name.values()), axis=0), model_weights, preds_by_name


def weighted_named_predictions(preds_by_name: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    out = np.zeros(len(next(iter(preds_by_name.values()))), dtype=float)
    total = 0.0
    for name, pred in preds_by_name.items():
        weight = float(weights.get(name, 0.0))
        out += weight * pred
        total += weight
    if total <= 0:
        return clip_prob(np.mean(list(preds_by_name.values()), axis=0))
    return clip_prob(out / total)


def optimize_named_prediction_weights(
    y: np.ndarray,
    preds_by_name: Dict[str, np.ndarray],
    step: float = 0.05,
) -> Tuple[Dict[str, float], float]:
    names = list(preds_by_name)
    if len(names) == 1:
        pred = preds_by_name[names[0]]
        return {names[0]: 1.0}, float(log_loss(y, pred, labels=[0, 1]))

    units = int(round(1.0 / step))
    best_weights = {name: 1.0 / len(names) for name in names}
    best_loss = math.inf

    def search(idx: int, remaining: int, current: List[int]) -> None:
        nonlocal best_weights, best_loss
        if idx == len(names) - 1:
            weights_units = current + [remaining]
            weights = {name: units_value / units for name, units_value in zip(names, weights_units)}
            pred = weighted_named_predictions(preds_by_name, weights)
            loss = float(log_loss(y, pred, labels=[0, 1]))
            if loss < best_loss:
                best_loss = loss
                best_weights = weights
            return
        for value in range(remaining + 1):
            search(idx + 1, remaining - value, current + [value])

    search(0, units, [])
    return best_weights, best_loss


def make_stack_model_specs(seed: int) -> List[Tuple[str, Pipeline]]:
    return [
        (
            "stack_logistic_c03",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=0.3,
                            max_iter=2000,
                            random_state=seed + 701,
                            solver="lbfgs",
                        ),
                    ),
                ]
            ),
        ),
        (
            "stack_logistic_c10",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=1.0,
                            max_iter=2000,
                            random_state=seed + 709,
                            solver="lbfgs",
                        ),
                    ),
                ]
            ),
        ),
        (
            "stack_hist_gb",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            max_iter=80,
                            learning_rate=0.035,
                            max_leaf_nodes=7,
                            l2_regularization=0.25,
                            random_state=seed + 719,
                        ),
                    ),
                ]
            ),
        ),
    ]


def make_stack_features(
    predictions: Dict[str, pd.DataFrame],
    target: str,
    include_variants: Sequence[str] = ("conservative", "aggressive", "low_time"),
) -> pd.DataFrame:
    parts: Dict[str, np.ndarray] = {}
    for variant in include_variants:
        frame = predictions[variant]
        for other in TARGETS:
            p = frame[other].to_numpy(dtype=float)
            parts[f"{variant}_{other}_p"] = p
            parts[f"{variant}_{other}_logit"] = logit_prob(p)
    base = predictions["conservative"][target].to_numpy(dtype=float)
    parts["base_confidence"] = np.abs(base - 0.5)
    parts["base_entropy"] = -(base * np.log(clip_prob(base)) + (1.0 - base) * np.log(clip_prob(1.0 - base)))
    return pd.DataFrame(parts)


def cross_target_stack_predictions(
    train_features: pd.DataFrame,
    sample: pd.DataFrame,
    oof_predictions: Dict[str, pd.DataFrame],
    test_predictions: Dict[str, pd.DataFrame],
    seed: int,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, object], Dict[str, pd.DataFrame]]:
    submissions = {
        "stack": sample.copy(),
        "stack_safe": sample.copy(),
    }
    oof_outputs = {
        "stack": pd.DataFrame(index=np.arange(len(train_features))),
        "stack_safe": pd.DataFrame(index=np.arange(len(train_features))),
    }
    reports: Dict[str, object] = {}
    specs = make_stack_model_specs(seed)

    for target in TARGETS:
        y = train_features[target].astype(int).to_numpy()
        Z_train = make_stack_features(oof_predictions, target)
        Z_test = make_stack_features(test_predictions, target)
        meta_oof = np.zeros(len(y), dtype=float)
        splits = stratified_splits(y, seed + 911)

        for tr_idx, va_idx in splits:
            fold_preds = []
            for _, spec in specs:
                model = clone(spec)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(Z_train.iloc[tr_idx], y[tr_idx])
                    fold_preds.append(clip_prob(predict_positive(model, Z_train.iloc[va_idx])))
            meta_oof[va_idx] = clip_prob(np.mean(fold_preds, axis=0))

        final_preds = []
        for _, spec in specs:
            model = clone(spec)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(Z_train, y)
                final_preds.append(clip_prob(predict_positive(model, Z_test)))
        meta_test = clip_prob(np.mean(final_preds, axis=0))

        base_oof = oof_predictions["conservative"][target].to_numpy(dtype=float)
        base_test = test_predictions["conservative"][target].to_numpy(dtype=float)
        stack_weight, stack_loss = optimize_two_way_blend(y, base_oof, meta_oof, max_alt_weight=1.0)
        safe_weight, safe_loss = optimize_two_way_blend(y, base_oof, meta_oof, max_alt_weight=0.35)

        submissions["stack"][target] = clip_prob((1.0 - stack_weight) * base_test + stack_weight * meta_test)
        submissions["stack_safe"][target] = clip_prob((1.0 - safe_weight) * base_test + safe_weight * meta_test)
        oof_outputs["stack"][target] = clip_prob((1.0 - stack_weight) * base_oof + stack_weight * meta_oof)
        oof_outputs["stack_safe"][target] = clip_prob((1.0 - safe_weight) * base_oof + safe_weight * meta_oof)
        reports[target] = {
            "base_logloss": float(log_loss(y, base_oof, labels=[0, 1])),
            "meta_logloss": float(log_loss(y, meta_oof, labels=[0, 1])),
            "stack_logloss": stack_loss,
            "stack_safe_logloss": safe_loss,
            "stack_weight": stack_weight,
            "stack_safe_weight": safe_weight,
        }

    reports["mean_stack_logloss"] = float(np.mean([reports[t]["stack_logloss"] for t in TARGETS]))
    reports["mean_stack_safe_logloss"] = float(np.mean([reports[t]["stack_safe_logloss"] for t in TARGETS]))
    reports["note"] = (
        "Second-stage target stacking uses first-stage OOF probabilities for train rows "
        "and full-train first-stage probabilities for test rows."
    )
    return submissions, reports, oof_outputs


def binary_patterns(n_targets: int) -> np.ndarray:
    return np.asarray(
        [[(value >> bit) & 1 for bit in range(n_targets)] for value in range(2 ** n_targets)],
        dtype=float,
    )


def pattern_log_ratio(y_matrix: np.ndarray, alpha: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    patterns = binary_patterns(y_matrix.shape[1])
    pattern_ids = (y_matrix.astype(int) * (2 ** np.arange(y_matrix.shape[1]))).sum(axis=1)
    counts = np.bincount(pattern_ids, minlength=len(patterns)).astype(float)
    joint = (counts + alpha) / (counts.sum() + alpha * len(patterns))
    marginals = np.clip(y_matrix.mean(axis=0), EPS, 1.0 - EPS)
    independent = np.prod(
        np.where(patterns == 1, marginals[None, :], 1.0 - marginals[None, :]),
        axis=1,
    )
    return patterns, np.log(np.clip(joint, EPS, None)) - np.log(np.clip(independent, EPS, None))


def apply_dependency_adjustment(
    pred_matrix: np.ndarray,
    patterns: np.ndarray,
    log_ratio: np.ndarray,
    strength: float,
) -> np.ndarray:
    p = clip_prob(pred_matrix)
    log_p = np.log(p)
    log_not = np.log(1.0 - p)
    scores = (
        np.matmul(log_p, patterns.T)
        + np.matmul(log_not, (1.0 - patterns).T)
        + strength * log_ratio[None, :]
    )
    scores -= scores.max(axis=1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=1, keepdims=True)
    return clip_prob(np.matmul(probs, patterns))


def tune_dependency_adjustment(
    y_matrix: np.ndarray,
    pred_df: pd.DataFrame,
    seed: int,
    strengths: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
    blend_grid: Sequence[float] = tuple(np.linspace(0.0, 1.0, 21)),
) -> Tuple[np.ndarray, Dict[str, object]]:
    base = pred_df[TARGETS].to_numpy(dtype=float)
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed + 1301)
    best_pred = base.copy()
    best_report: Dict[str, object] = {
        "strength": 0.0,
        "blend": 0.0,
        "mean_logloss": float(np.mean([log_loss(y_matrix[:, j], base[:, j], labels=[0, 1]) for j in range(len(TARGETS))])),
    }
    best_loss = float(best_report["mean_logloss"])

    for strength in strengths:
        adjusted = np.zeros_like(base)
        for tr_idx, va_idx in splitter.split(base):
            patterns, ratio = pattern_log_ratio(y_matrix[tr_idx], alpha=1.0)
            adjusted[va_idx] = apply_dependency_adjustment(base[va_idx], patterns, ratio, strength)
        for blend in blend_grid:
            candidate = clip_prob((1.0 - blend) * base + blend * adjusted)
            target_losses = [
                float(log_loss(y_matrix[:, j], candidate[:, j], labels=[0, 1]))
                for j in range(len(TARGETS))
            ]
            mean_loss = float(np.mean(target_losses))
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_pred = candidate
                best_report = {
                    "strength": float(strength),
                    "blend": float(blend),
                    "mean_logloss": mean_loss,
                    "target_logloss": {target: target_losses[j] for j, target in enumerate(TARGETS)},
                }
    if "target_logloss" not in best_report:
        best_report["target_logloss"] = {
            target: float(log_loss(y_matrix[:, j], base[:, j], labels=[0, 1]))
            for j, target in enumerate(TARGETS)
        }
    return best_pred, best_report


def build_dependency_submission(
    train_features: pd.DataFrame,
    sample: pd.DataFrame,
    oof_predictions: Dict[str, pd.DataFrame],
    test_predictions: Dict[str, pd.DataFrame],
    base_variant: str,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    y_matrix = train_features[TARGETS].astype(int).to_numpy()
    _, report = tune_dependency_adjustment(y_matrix, oof_predictions[base_variant], seed)
    patterns, ratio = pattern_log_ratio(y_matrix, alpha=1.0)
    test_base = test_predictions[base_variant][TARGETS].to_numpy(dtype=float)
    adjusted_test = apply_dependency_adjustment(test_base, patterns, ratio, float(report["strength"]))
    final_test = clip_prob((1.0 - float(report["blend"])) * test_base + float(report["blend"]) * adjusted_test)
    submission = sample.copy()
    for idx, target in enumerate(TARGETS):
        submission[target] = final_test[:, idx]
    report["base_variant"] = base_variant
    return submission, report


def append_prediction_features(
    X: pd.DataFrame,
    predictions: Dict[str, pd.DataFrame],
    variants: Sequence[str] = ("conservative", "stack", "stack_safe"),
) -> pd.DataFrame:
    out = X.copy()
    for variant in variants:
        if variant not in predictions:
            continue
        frame = predictions[variant]
        for target in TARGETS:
            p = frame[target].to_numpy(dtype=float)
            out[f"pred_{variant}_{target}"] = p
            out[f"pred_{variant}_{target}_logit"] = logit_prob(p)
    return out


def train_stage2_full_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    train_features: pd.DataFrame,
    sample: pd.DataFrame,
    subjects_train: np.ndarray,
    subjects_test: np.ndarray,
    dates_train: np.ndarray,
    dates_test: np.ndarray,
    oof_predictions: Dict[str, pd.DataFrame],
    test_predictions: Dict[str, pd.DataFrame],
    seed: int,
    n_jobs: int,
    time_blend_cap: float,
    low_time_blend_cap: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    X2_train = append_prediction_features(X_train, oof_predictions)
    X2_test = append_prediction_features(X_test, test_predictions)
    submission = sample.copy()
    oof_frame = pd.DataFrame(index=np.arange(len(train_features)))
    reports: Dict[str, object] = {}

    for target in TARGETS:
        print(f"Stage2 target {target}", flush=True)
        y = train_features[target].astype(int).to_numpy()
        main = cv_target(
            X2_train,
            y,
            subjects_train,
            dates_train,
            target,
            seed + 1601,
            n_jobs,
            time_blend_cap=time_blend_cap,
            low_time_blend_cap=low_time_blend_cap,
        )
        reports[target] = {k: v for k, v in main.items() if k not in {"oof", "oof_variants"}}
        oof_frame[target] = np.asarray(main["oof_variants"]["conservative"], dtype=float)
        components, _ = fit_final_target(
            X2_train,
            y,
            X2_test,
            subjects_train,
            subjects_test,
            dates_train,
            dates_test,
            target,
            seed + 1601,
            n_jobs,
            model_blend_weights=main["model_blend_weights"],
            time_component_weights=main["time_component_weights"],
        )
        submission[target] = blend_component_predictions(components, main["blend_weights"])
        print(f"  stage2 CV logloss: {main['logloss']:.6f}", flush=True)

    reports["mean_logloss"] = float(np.mean([reports[target]["logloss"] for target in TARGETS]))
    return submission, oof_frame, reports


def cv_target(
    X: pd.DataFrame,
    y: np.ndarray,
    subjects: np.ndarray,
    dates: np.ndarray,
    target: str,
    seed: int,
    n_jobs: int,
    groups: Optional[np.ndarray] = None,
    custom_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    splitter_name: Optional[str] = None,
    diagnostic_models: bool = False,
    time_blend_cap: float = 0.55,
    low_time_blend_cap: float = 0.25,
    blend_weight: float = 0.85,
) -> Dict[str, object]:
    if custom_splits is not None:
        splits = custom_splits
        splitter_name = splitter_name or "custom"
    elif groups is None:
        splits = stratified_splits(y, seed)
        splitter_name = "stratified_kfold"
    else:
        n_splits = min(5, len(np.unique(groups)))
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(X, y, groups=groups))
        splitter_name = "group_kfold_subject"

    model_oof = np.zeros(len(y), dtype=float)
    subject_oof = np.zeros(len(y), dtype=float)
    time_oof = np.zeros(len(y), dtype=float)
    fold_reports = []
    specs = make_model_specs(seed, n_jobs, diagnostic=groups is not None or diagnostic_models)
    model_oof_by_name: Dict[str, np.ndarray] = {name: np.zeros(len(y), dtype=float) for name, _ in specs}
    time_oof_by_name: Dict[str, np.ndarray] = {
        name: np.zeros(len(y), dtype=float) for name in temporal_component_names(target)
    }
    time_params = TIME_LABEL_PARAMS.get(target, {"k": 8, "power": 1.0, "smoothing": 4.0})

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        y_tr = y[tr_idx]
        tr_subjects = subjects[tr_idx]
        va_subjects = subjects[va_idx]
        tr_dates = dates[tr_idx]
        va_dates = dates[va_idx]
        tr_te = loo_subject_encoding(tr_subjects, y_tr)
        va_te = smoothed_subject_means(tr_subjects, y_tr, va_subjects)
        tr_time_components = temporal_label_components(
            tr_subjects,
            tr_dates,
            y_tr,
            tr_subjects,
            tr_dates,
            target,
            prior=float(np.mean(y_tr)),
            exclude_self=True,
        )
        va_time_components = temporal_label_components(
            tr_subjects,
            tr_dates,
            y_tr,
            va_subjects,
            va_dates,
            target,
            prior=float(np.mean(y_tr)),
            exclude_self=False,
        )

        X_tr = X.iloc[tr_idx].copy()
        X_va = X.iloc[va_idx].copy()
        te_col = f"te_subject_{target.lower()}"
        X_tr[te_col] = tr_te
        X_va[te_col] = va_te
        for component_name in temporal_component_names(target):
            time_oof_by_name[component_name][va_idx] = va_time_components[component_name]
        model_time_components = (
            temporal_component_names(target)
            if target in TIME_COMPONENT_MODEL_FEATURE_TARGETS
            else ["default"]
        )
        for component_name in model_time_components:
            time_col = f"time_{component_name}_{target.lower()}"
            X_tr[time_col] = tr_time_components[component_name]
            X_va[time_col] = va_time_components[component_name]

        model_pred, weights, fold_model_preds = average_model_predictions(specs, X_tr, y_tr, X_va)
        for name, pred in fold_model_preds.items():
            model_oof_by_name[name][va_idx] = pred
        model_oof[va_idx] = model_pred
        subject_oof[va_idx] = va_te
        time_oof[va_idx] = va_time_components["default"]
        fixed_pred = clip_prob(blend_weight * model_pred + (1.0 - blend_weight) * va_te)
        fold_loss = float(log_loss(y[va_idx], fixed_pred, labels=[0, 1]))
        fold_reports.append(
            {
                "fold": fold,
                "n_valid": int(len(va_idx)),
                "positive_rate": float(np.mean(y[va_idx])),
                "fixed_blend_logloss": fold_loss,
                "model_weights": weights,
            }
        )

    model_blend_weights, model_blend_loss = optimize_named_prediction_weights(y, model_oof_by_name)
    model_oof = weighted_named_predictions(model_oof_by_name, model_blend_weights)
    time_component_weights, time_component_loss = optimize_named_prediction_weights(y, time_oof_by_name)
    time_oof = weighted_named_predictions(time_oof_by_name, time_component_weights)
    aggressive_weights, aggressive_loss = optimize_blend_weights(
        y, model_oof, subject_oof, time_oof, max_time_neighbor=1.0
    )
    conservative_weights, conservative_loss = optimize_blend_weights(
        y, model_oof, subject_oof, time_oof, max_time_neighbor=time_blend_cap
    )
    low_time_weights, low_time_loss = optimize_blend_weights(
        y, model_oof, subject_oof, time_oof, max_time_neighbor=low_time_blend_cap
    )
    notime_weights, notime_loss = optimize_blend_weights(
        y, model_oof, subject_oof, time_oof, max_time_neighbor=0.0
    )
    aggressive_oof = clip_prob(
        aggressive_weights[0] * model_oof
        + aggressive_weights[1] * subject_oof
        + aggressive_weights[2] * time_oof
    )
    oof = clip_prob(
        conservative_weights[0] * model_oof
        + conservative_weights[1] * subject_oof
        + conservative_weights[2] * time_oof
    )
    low_time_oof = clip_prob(
        low_time_weights[0] * model_oof
        + low_time_weights[1] * subject_oof
        + low_time_weights[2] * time_oof
    )
    notime_oof = clip_prob(
        notime_weights[0] * model_oof
        + notime_weights[1] * subject_oof
        + notime_weights[2] * time_oof
    )
    ensemble_oof = clip_prob(0.50 * oof + 0.25 * aggressive_oof + 0.25 * low_time_oof)
    prior_shrink_weight, shrunk_loss = optimize_prior_shrink(y, oof)
    shrunk_oof = shrink_to_prior(oof, float(np.mean(y)), prior_shrink_weight)

    return {
        "target": target,
        "splitter": splitter_name,
        "logloss": float(log_loss(y, oof, labels=[0, 1])),
        "component_logloss": {
            "model": float(log_loss(y, clip_prob(model_oof), labels=[0, 1])),
            "subject": float(log_loss(y, clip_prob(subject_oof), labels=[0, 1])),
            "time_neighbor": float(log_loss(y, clip_prob(time_oof), labels=[0, 1])),
            "time_default": float(log_loss(y, clip_prob(time_oof_by_name["default"]), labels=[0, 1])),
            "by_time_component": {
                name: float(log_loss(y, clip_prob(pred), labels=[0, 1]))
                for name, pred in time_oof_by_name.items()
            },
            "model_equal_average": float(log_loss(y, clip_prob(weighted_named_predictions(model_oof_by_name, {name: 1.0 for name in model_oof_by_name})), labels=[0, 1])),
            "by_model": {
                name: float(log_loss(y, clip_prob(pred), labels=[0, 1]))
                for name, pred in model_oof_by_name.items()
            },
        },
        "model_blend_weights": model_blend_weights,
        "model_blend_logloss": model_blend_loss,
        "time_component_weights": time_component_weights,
        "time_component_logloss": time_component_loss,
        "blend_weights": weights_to_dict(conservative_weights),
        "aggressive_blend_weights": weights_to_dict(aggressive_weights),
        "low_time_blend_weights": weights_to_dict(low_time_weights),
        "notime_blend_weights": weights_to_dict(notime_weights),
        "aggressive_logloss": aggressive_loss,
        "conservative_logloss": conservative_loss,
        "low_time_logloss": low_time_loss,
        "notime_logloss": notime_loss,
        "ensemble_logloss": float(log_loss(y, ensemble_oof, labels=[0, 1])),
        "prior_shrink_weight": prior_shrink_weight,
        "shrunk_logloss": shrunk_loss,
        "time_label_params": time_params,
        "folds": fold_reports,
        "oof": oof.tolist(),
        "oof_variants": {
            "conservative": oof.tolist(),
            "aggressive": aggressive_oof.tolist(),
            "low_time": low_time_oof.tolist(),
            "notime": notime_oof.tolist(),
            "shrunk": shrunk_oof.tolist(),
            "ensemble": ensemble_oof.tolist(),
        },
    }


def fit_final_target(
    X_train: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    train_subjects: np.ndarray,
    test_subjects: np.ndarray,
    train_dates: np.ndarray,
    test_dates: np.ndarray,
    target: str,
    seed: int,
    n_jobs: int,
    model_blend_weights: Optional[Dict[str, float]] = None,
    time_component_weights: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, object]]]:
    tr_te = loo_subject_encoding(train_subjects, y)
    te_te = smoothed_subject_means(train_subjects, y, test_subjects)
    tr_time_components = temporal_label_components(
        train_subjects,
        train_dates,
        y,
        train_subjects,
        train_dates,
        target,
        prior=float(np.mean(y)),
        exclude_self=True,
    )
    te_time_components = temporal_label_components(
        train_subjects,
        train_dates,
        y,
        test_subjects,
        test_dates,
        target,
        prior=float(np.mean(y)),
        exclude_self=False,
    )
    te_col = f"te_subject_{target.lower()}"

    X_tr = X_train.copy()
    X_te = X_test.copy()
    X_tr[te_col] = tr_te
    X_te[te_col] = te_te
    model_time_components = (
        temporal_component_names(target)
        if target in TIME_COMPONENT_MODEL_FEATURE_TARGETS
        else ["default"]
    )
    for component_name in model_time_components:
        time_col = f"time_{component_name}_{target.lower()}"
        X_tr[time_col] = tr_time_components[component_name]
        X_te[time_col] = te_time_components[component_name]

    specs = make_model_specs(seed, n_jobs)
    preds_by_name: Dict[str, np.ndarray] = {}
    importances: List[Dict[str, object]] = []

    for name, spec in specs:
        model = clone(spec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y)
            preds_by_name[name] = clip_prob(predict_positive(model, X_te))

        if name in {"extra_trees", "random_forest", "lightgbm"}:
            fitted_model = model.named_steps["model"]
            imputer = model.named_steps["imputer"]
            try:
                feature_names = imputer.get_feature_names_out(X_tr.columns)
            except Exception:
                feature_names = np.asarray(X_tr.columns)
            for feature, importance in zip(feature_names, fitted_model.feature_importances_):
                importances.append(
                    {
                        "target": target,
                        "model": name,
                        "feature": str(feature),
                        "importance": float(importance),
                    }
                )

    if model_blend_weights is None:
        model_pred = np.mean(list(preds_by_name.values()), axis=0)
    else:
        model_pred = weighted_named_predictions(preds_by_name, model_blend_weights)
    components = {
        "model": clip_prob(model_pred),
        "subject": clip_prob(te_te),
        "time_neighbor": weighted_named_predictions(
            te_time_components,
            time_component_weights or {"default": 1.0},
        ),
    }
    return components, importances


def train_and_predict(
    features: pd.DataFrame,
    sample: pd.DataFrame,
    seed: int,
    n_jobs: int,
    skip_group_cv: bool,
    skip_block_cv: bool,
    time_blend_cap: float,
    low_time_blend_cap: float,
    block_gap_days: int,
    skip_meta_models: bool = False,
    target_feature_mode: str = "all",
    max_target_features: int = 900,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, object], pd.DataFrame]:
    is_train = features["is_train"].to_numpy() == 1
    train_features = features[is_train].sort_values("row_id").reset_index(drop=True)
    test_features = features[~is_train].sort_values("row_id").reset_index(drop=True)

    X_all = make_design_matrix(pd.concat([train_features, test_features], ignore_index=True))
    X_train = X_all.iloc[: len(train_features)].reset_index(drop=True)
    X_test = X_all.iloc[len(train_features) :].reset_index(drop=True)
    subjects_train = train_features["subject_id"].to_numpy()
    subjects_test = test_features["subject_id"].to_numpy()
    dates_train = train_features["lifelog_date"].to_numpy()
    dates_test = test_features["lifelog_date"].to_numpy()

    submissions = {
        "conservative": sample.copy(),
        "aggressive": sample.copy(),
        "low_time": sample.copy(),
        "ensemble": sample.copy(),
        "notime": sample.copy(),
        "shrunk": sample.copy(),
    }
    oof_predictions: Dict[str, pd.DataFrame] = {
        name: pd.DataFrame(index=np.arange(len(train_features)))
        for name in ["conservative", "aggressive", "low_time", "ensemble", "notime", "shrunk"]
    }
    main_reports: Dict[str, object] = {}
    group_reports: Dict[str, object] = {}
    blocked_reports: Dict[str, object] = {}
    all_importances: List[Dict[str, object]] = []
    target_feature_reports: Dict[str, object] = {}

    print(f"Training {len(TARGETS)} binary targets with {X_train.shape[1]} base features...", flush=True)
    for target in TARGETS:
        print(f"Target {target}", flush=True)
        y = train_features[target].astype(int).to_numpy()
        X_target_train, X_target_test, selected_cols = select_target_design_matrix(
            X_train,
            X_test,
            target,
            mode=target_feature_mode,
            max_features=max_target_features,
        )
        target_feature_reports[target] = {
            "mode": target_feature_mode,
            "n_features": int(X_target_train.shape[1]),
            "max_target_features": int(max_target_features),
            "sample_features": selected_cols[:25],
        }
        if target_feature_mode != "all":
            print(f"  target feature subset: {X_target_train.shape[1]} features ({target_feature_mode})", flush=True)
        main = cv_target(
            X_target_train,
            y,
            subjects_train,
            dates_train,
            target,
            seed,
            n_jobs,
            time_blend_cap=time_blend_cap,
            low_time_blend_cap=low_time_blend_cap,
        )
        for variant, values in main["oof_variants"].items():
            oof_predictions[variant][target] = np.asarray(values, dtype=float)
        main_reports[target] = {k: v for k, v in main.items() if k not in {"oof", "oof_variants"}}
        print(f"  stratified conservative CV logloss: {main['logloss']:.6f}", flush=True)
        print(
            "  aggressive/conservative/low-time/notime/shrunk/ensemble: "
            f"{main['aggressive_logloss']:.6f} / {main['conservative_logloss']:.6f} / "
            f"{main['low_time_logloss']:.6f} / {main['notime_logloss']:.6f} / "
            f"{main['shrunk_logloss']:.6f} / {main['ensemble_logloss']:.6f}",
            flush=True,
        )
        print(f"  conservative blend weights: {main['blend_weights']}", flush=True)

        if not skip_group_cv:
            group = cv_target(
                X_target_train,
                y,
                subjects_train,
                dates_train,
                target,
                seed,
                n_jobs,
                groups=subjects_train,
                diagnostic_models=True,
                time_blend_cap=time_blend_cap,
                low_time_blend_cap=low_time_blend_cap,
            )
            group_reports[target] = {k: v for k, v in group.items() if k != "oof"}
            print(f"  subject GroupKFold logloss: {group['logloss']:.6f}", flush=True)

        if not skip_block_cv:
            block_splits = blocked_time_splits(
                subjects_train,
                dates_train,
                y,
                n_splits=5,
                gap_days=block_gap_days,
            )
            if block_splits:
                blocked = cv_target(
                    X_target_train,
                    y,
                    subjects_train,
                    dates_train,
                    target,
                    seed,
                    n_jobs,
                    custom_splits=block_splits,
                    splitter_name=f"blocked_time_gap_{block_gap_days}d",
                    diagnostic_models=True,
                    time_blend_cap=time_blend_cap,
                    low_time_blend_cap=low_time_blend_cap,
                )
                blocked_reports[target] = {k: v for k, v in blocked.items() if k != "oof"}
                print(f"  blocked time CV logloss: {blocked['logloss']:.6f}", flush=True)
            else:
                blocked_reports[target] = {"splitter": f"blocked_time_gap_{block_gap_days}d", "error": "no valid folds"}
                print("  blocked time CV skipped: no valid folds", flush=True)

        components, importances = fit_final_target(
            X_target_train,
            y,
            X_target_test,
            subjects_train,
            subjects_test,
            dates_train,
            dates_test,
            target,
            seed,
            n_jobs,
            model_blend_weights=main["model_blend_weights"],
            time_component_weights=main["time_component_weights"],
        )
        submissions["conservative"][target] = blend_component_predictions(components, main["blend_weights"])
        submissions["aggressive"][target] = blend_component_predictions(components, main["aggressive_blend_weights"])
        submissions["low_time"][target] = blend_component_predictions(components, main["low_time_blend_weights"])
        submissions["notime"][target] = blend_component_predictions(components, main["notime_blend_weights"])
        submissions["ensemble"][target] = clip_prob(
            0.50 * submissions["conservative"][target].to_numpy()
            + 0.25 * submissions["aggressive"][target].to_numpy()
            + 0.25 * submissions["low_time"][target].to_numpy()
        )
        submissions["shrunk"][target] = shrink_to_prior(
            submissions["conservative"][target].to_numpy(),
            float(np.mean(y)),
            float(main["prior_shrink_weight"]),
        )
        all_importances.extend(importances)

    if skip_meta_models:
        stack_reports = {"note": "Skipped because skip_meta_models=True."}
        joint_report = {"note": "Skipped because skip_meta_models=True."}
        stack_joint_report = {"note": "Skipped because skip_meta_models=True."}
        stage2_report = {"note": "Skipped because skip_meta_models=True."}
        stage2_joint_report = {"note": "Skipped because skip_meta_models=True."}
        print("Skipping stack/joint/stage2 meta-model diagnostics.", flush=True)
    else:
        print("Training cross-target stacking layer...", flush=True)
        stack_submissions, stack_reports, stack_oof_predictions = cross_target_stack_predictions(
            train_features=train_features,
            sample=sample,
            oof_predictions=oof_predictions,
            test_predictions=submissions,
            seed=seed,
        )
        submissions.update(stack_submissions)
        oof_predictions.update(stack_oof_predictions)
        print(
            "  stack/safe stack CV logloss: "
            f"{stack_reports['mean_stack_logloss']:.6f} / {stack_reports['mean_stack_safe_logloss']:.6f}",
            flush=True,
        )
        print("Applying joint target dependency adjustment...", flush=True)
        joint_submission, joint_report = build_dependency_submission(
            train_features=train_features,
            sample=sample,
            oof_predictions=oof_predictions,
            test_predictions=submissions,
            base_variant="conservative",
            seed=seed,
        )
        stack_joint_submission, stack_joint_report = build_dependency_submission(
            train_features=train_features,
            sample=sample,
            oof_predictions=oof_predictions,
            test_predictions=submissions,
            base_variant="stack",
            seed=seed,
        )
        submissions["joint"] = joint_submission
        submissions["stack_joint"] = stack_joint_submission
        print(
            "  joint conservative/stack CV logloss: "
            f"{joint_report['mean_logloss']:.6f} / {stack_joint_report['mean_logloss']:.6f}",
            flush=True,
        )
        print("Training full feature stage2 stacking model...", flush=True)
        stage2_submission, stage2_oof, stage2_report = train_stage2_full_model(
            X_train=X_train,
            X_test=X_test,
            train_features=train_features,
            sample=sample,
            subjects_train=subjects_train,
            subjects_test=subjects_test,
            dates_train=dates_train,
            dates_test=dates_test,
            oof_predictions=oof_predictions,
            test_predictions=submissions,
            seed=seed,
            n_jobs=n_jobs,
            time_blend_cap=time_blend_cap,
            low_time_blend_cap=low_time_blend_cap,
        )
        submissions["stage2"] = stage2_submission
        oof_predictions["stage2"] = stage2_oof
        stage2_joint_submission, stage2_joint_report = build_dependency_submission(
            train_features=train_features,
            sample=sample,
            oof_predictions=oof_predictions,
            test_predictions=submissions,
            base_variant="stage2",
            seed=seed,
        )
        submissions["stage2_joint"] = stage2_joint_submission
        print(
            "  stage2/stage2-joint CV logloss: "
            f"{stage2_report['mean_logloss']:.6f} / {stage2_joint_report['mean_logloss']:.6f}",
            flush=True,
        )

    main_losses = [float(main_reports[t]["logloss"]) for t in TARGETS]
    aggressive_losses = [float(main_reports[t]["aggressive_logloss"]) for t in TARGETS]
    low_time_losses = [float(main_reports[t]["low_time_logloss"]) for t in TARGETS]
    notime_losses = [float(main_reports[t]["notime_logloss"]) for t in TARGETS]
    shrunk_losses = [float(main_reports[t]["shrunk_logloss"]) for t in TARGETS]
    ensemble_losses = [float(main_reports[t]["ensemble_logloss"]) for t in TARGETS]
    report = {
        "seed": seed,
        "n_train": int(len(train_features)),
        "n_test": int(len(test_features)),
        "n_base_features": int(X_train.shape[1]),
        "targets": TARGETS,
        "submission_policy": {
            "submission.csv": "conservative time-neighbor-capped blend",
            "submission_aggressive.csv": "random-CV optimized blend without time-neighbor cap",
            "submission_low_time.csv": "low time-neighbor cap blend",
            "submission_ensemble.csv": "0.50 conservative + 0.25 aggressive + 0.25 low-time",
            "submission_notime.csv": "model + subject blend with time-neighbor weight forced to zero",
            "submission_shrunk.csv": "conservative blend shrunk toward each target train prior",
            "submission_stack.csv": "cross-target second-stage stack with OOF-optimized blend",
            "submission_stack_safe.csv": "cross-target second-stage stack with meta blend capped at 0.35",
            "submission_joint.csv": "joint target dependency adjustment applied to conservative predictions",
            "submission_stack_joint.csv": "joint target dependency adjustment applied to stacked predictions",
            "submission_stage2.csv": "full feature second-stage model using first-stage OOF prediction features",
            "submission_stage2_joint.csv": "joint target dependency adjustment applied to stage2 predictions",
            "time_blend_cap": time_blend_cap,
            "low_time_blend_cap": low_time_blend_cap,
            "block_gap_days": block_gap_days,
        },
        "label_rates": {t: float(train_features[t].mean()) for t in TARGETS},
        "main_cv": main_reports,
        "mean_main_cv_logloss": float(np.mean(main_losses)),
        "mean_aggressive_cv_logloss": float(np.mean(aggressive_losses)),
        "mean_low_time_cv_logloss": float(np.mean(low_time_losses)),
        "mean_notime_cv_logloss": float(np.mean(notime_losses)),
        "mean_shrunk_cv_logloss": float(np.mean(shrunk_losses)),
        "mean_ensemble_cv_logloss": float(np.mean(ensemble_losses)),
        "stacking_cv": stack_reports,
        "joint_cv": {
            "conservative": joint_report,
            "stack": stack_joint_report,
            "stage2": stage2_joint_report,
        },
        "stage2_cv": stage2_report,
        "group_cv": group_reports,
        "blocked_time_cv": blocked_reports,
        "target_feature_selection": target_feature_reports,
    }
    importance_df = pd.DataFrame(all_importances)
    return submissions, report, importance_df


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    if list(submission.columns) != list(sample.columns):
        raise ValueError("Submission columns do not match sample submission columns.")
    if len(submission) != len(sample):
        raise ValueError("Submission row count does not match sample submission row count.")
    for col in TARGETS:
        if not submission[col].between(0, 1).all():
            raise ValueError(f"Column {col} contains values outside [0, 1].")


def prediction_distance_frame(candidate: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cand_values = candidate[TARGETS].to_numpy(dtype=float)
    ref_values = reference[TARGETS].to_numpy(dtype=float)
    diff = cand_values - ref_values
    abs_diff = np.abs(diff)
    for j, target in enumerate(TARGETS):
        target_abs = abs_diff[:, j]
        target_diff = diff[:, j]
        rows.append(
            {
                "target": target,
                "mean_abs_diff": float(np.mean(target_abs)),
                "max_abs_diff": float(np.max(target_abs)),
                "mean_signed_diff": float(np.mean(target_diff)),
                "pct_abs_diff_gt_0p05": float(np.mean(target_abs > 0.05)),
                "pct_abs_diff_gt_0p10": float(np.mean(target_abs > 0.10)),
                "pct_abs_diff_gt_0p20": float(np.mean(target_abs > 0.20)),
                "candidate_mean": float(candidate[target].mean()),
                "reference_mean": float(reference[target].mean()),
            }
        )
    rows.append(
        {
            "target": "__overall__",
            "mean_abs_diff": float(np.mean(abs_diff)),
            "max_abs_diff": float(np.max(abs_diff)),
            "mean_signed_diff": float(np.mean(diff)),
            "pct_abs_diff_gt_0p05": float(np.mean(abs_diff > 0.05)),
            "pct_abs_diff_gt_0p10": float(np.mean(abs_diff > 0.10)),
            "pct_abs_diff_gt_0p20": float(np.mean(abs_diff > 0.20)),
            "candidate_mean": float(np.mean(cand_values)),
            "reference_mean": float(np.mean(ref_values)),
        }
    )
    return pd.DataFrame(rows)


def validate_submission_identity(frame: pd.DataFrame, sample: pd.DataFrame, name: str) -> None:
    id_cols = [col for col in sample.columns if col not in TARGETS]
    if list(frame.columns) != list(sample.columns):
        raise ValueError(f"{name} columns do not match sample submission columns.")
    if len(frame) != len(sample):
        raise ValueError(f"{name} row count does not match sample submission row count.")
    if not frame[id_cols].equals(sample[id_cols]):
        raise ValueError(f"{name} id/date columns do not match sample submission order.")


def run_public_blend(args: argparse.Namespace) -> None:
    if args.public_blend_base is None or args.public_blend_alt is None:
        raise ValueError("--public-blend-base and --public-blend-alt are required with --public-blend-weight.")
    if args.public_blend_weight is None:
        raise ValueError("--public-blend-weight is required for public blend mode.")

    data_dir = Path(args.data_dir)
    sample = pd.read_csv(data_dir / "ch2026_submission_sample.csv")
    base = pd.read_csv(args.public_blend_base)
    alt = pd.read_csv(args.public_blend_alt)
    validate_submission_identity(base, sample, "public blend base")
    validate_submission_identity(alt, sample, "public blend alt")

    weight = float(args.public_blend_weight)
    out = sample.copy()
    blended = (1.0 - weight) * base[TARGETS].to_numpy(dtype=float) + weight * alt[TARGETS].to_numpy(dtype=float)
    out[TARGETS] = clip_prob(blended)
    validate_submission(out, sample)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.public_blend_report_out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    report = {
        "mode": "public_blend",
        "base": args.public_blend_base,
        "alt": args.public_blend_alt,
        "weight": weight,
        "out": args.out,
        "rows": int(len(out)),
        "target_min": float(out[TARGETS].min().min()),
        "target_max": float(out[TARGETS].max().max()),
        "target_means": {target: float(out[target].mean()) for target in TARGETS},
    }

    if args.prediction_reference:
        reference = pd.read_csv(args.prediction_reference)
        validate_submission_identity(reference, sample, "prediction reference")
        distance = prediction_distance_frame(out, reference)
        Path(args.prediction_distance_out).parent.mkdir(parents=True, exist_ok=True)
        distance.to_csv(args.prediction_distance_out, index=False)
        overall = distance[distance["target"] == "__overall__"].iloc[0].to_dict()
        report["prediction_reference"] = args.prediction_reference
        report["prediction_distance_out"] = args.prediction_distance_out
        report["prediction_distance_overall"] = overall

    with open(args.public_blend_report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved {args.out}", flush=True)
    print(f"Saved {args.public_blend_report_out}", flush=True)
    if args.prediction_reference:
        print(f"Saved {args.prediction_distance_out}", flush=True)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(args.n_jobs))
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\.impute\._base")
    np.random.seed(args.seed)
    data_dir = Path(args.data_dir)

    if args.public_blend_weight is not None:
        run_public_blend(args)
        return

    effective_top_n = args.high_card_top_n if args.high_card_top_n is not None else args.top_n
    if args.feature_route == "stable" and args.high_card_top_n is None:
        effective_top_n = 5
    stable_summary_only = args.stable_high_card_summary_only or args.feature_route == "stable"
    skip_meta_models = args.skip_meta_models or args.feature_route == "stable"

    train, sample, features = build_features(
        data_dir,
        top_n=effective_top_n,
        feature_route=args.feature_route,
        disable_subject_rolling=args.disable_subject_rolling,
        disable_raw_id_features=args.disable_raw_id_features,
        stable_high_card_summary_only=stable_summary_only,
        enable_sleep_v2=args.enable_sleep_v2,
    )
    print(f"Feature table shape: {features.shape}", flush=True)

    submissions, report, importance_df = train_and_predict(
        features=features,
        sample=sample,
        seed=args.seed,
        n_jobs=args.n_jobs,
        skip_group_cv=args.skip_group_cv,
        skip_block_cv=args.skip_block_cv,
        time_blend_cap=args.time_blend_cap,
        low_time_blend_cap=args.low_time_blend_cap,
        block_gap_days=args.block_gap_days,
        skip_meta_models=skip_meta_models,
        target_feature_mode=args.target_feature_mode,
        max_target_features=args.max_target_features,
    )
    for submission in submissions.values():
        validate_submission(submission, sample)

    report["feature_route_config"] = {
        "feature_route": args.feature_route,
        "disable_subject_rolling": bool(args.disable_subject_rolling),
        "subject_rolling_enabled": bool(not args.disable_subject_rolling and args.feature_route not in {"norelroll", "stable"}),
        "top_n": int(args.top_n),
        "high_card_top_n": None if args.high_card_top_n is None else int(args.high_card_top_n),
        "effective_high_card_top_n": int(effective_top_n),
        "disable_raw_id_features": bool(args.disable_raw_id_features),
        "stable_high_card_summary_only": bool(stable_summary_only),
        "enable_sleep_v2": bool(args.enable_sleep_v2),
        "target_feature_mode": args.target_feature_mode,
        "max_target_features": int(args.max_target_features),
        "skip_meta_models": bool(skip_meta_models),
    }

    output_paths = [
        args.out,
        args.aggressive_out,
        args.low_time_out,
        args.ensemble_out,
        args.notime_out,
        args.shrunk_out,
        args.stack_out,
        args.stack_safe_out,
        args.joint_out,
        args.stack_joint_out,
        args.stage2_out,
        args.stage2_joint_out,
        args.report_out,
        args.importance_out,
    ]
    for path in output_paths:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    submission_outputs = [
        ("conservative", args.out),
        ("aggressive", args.aggressive_out),
        ("low_time", args.low_time_out),
        ("ensemble", args.ensemble_out),
        ("notime", args.notime_out),
        ("shrunk", args.shrunk_out),
        ("stack", args.stack_out),
        ("stack_safe", args.stack_safe_out),
        ("joint", args.joint_out),
        ("stack_joint", args.stack_joint_out),
        ("stage2", args.stage2_out),
        ("stage2_joint", args.stage2_joint_out),
    ]
    saved_outputs = []
    for variant, path in submission_outputs:
        if variant not in submissions:
            continue
        submissions[variant].to_csv(path, index=False)
        saved_outputs.append(path)

    if args.prediction_reference:
        reference = pd.read_csv(args.prediction_reference)
        sample_csv = pd.read_csv(data_dir / "ch2026_submission_sample.csv")
        validate_submission_identity(reference, sample_csv, "prediction reference")
        candidate = pd.read_csv(args.out)
        validate_submission_identity(candidate, sample_csv, "candidate output")
        distance = prediction_distance_frame(candidate, reference)
        Path(args.prediction_distance_out).parent.mkdir(parents=True, exist_ok=True)
        distance.to_csv(args.prediction_distance_out, index=False)
        report["prediction_reference"] = args.prediction_reference
        report["prediction_distance_out"] = args.prediction_distance_out
        report["prediction_distance_overall"] = distance[distance["target"] == "__overall__"].iloc[0].to_dict()

    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if not importance_df.empty:
        summary = (
            importance_df.groupby("feature", as_index=False)["importance"]
            .mean()
            .sort_values("importance", ascending=False)
        )
        importance_df = importance_df.merge(summary.rename(columns={"importance": "mean_importance"}), on="feature")
        importance_df.sort_values(["mean_importance", "target", "model"], ascending=[False, True, True]).to_csv(
            args.importance_out, index=False
        )

    for path in saved_outputs:
        print(f"Saved {path}", flush=True)
    print(f"Saved {args.report_out}", flush=True)
    if not importance_df.empty:
        print(f"Saved {args.importance_out}", flush=True)
    if args.prediction_reference:
        print(f"Saved {args.prediction_distance_out}", flush=True)
    print(f"Mean stratified CV logloss: {report['mean_main_cv_logloss']:.6f}", flush=True)


if __name__ == "__main__":
    main()
