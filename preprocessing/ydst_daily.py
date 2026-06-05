from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ydst"
OUTPUT_DIR = PROJECT_ROOT / "data" / "preprocessed"
OUTPUT_PATH = OUTPUT_DIR / "ydst_seoul_daily.csv"

OPEN_START_HOUR = 9.0


def scheduled_close_hour(month: int) -> float:
    if month in {1, 2, 11, 12}:
        return 17.0
    if month in {6, 7, 8}:
        return 18.5
    return 18.0


def scheduled_last_entry_hour(month: int) -> float:
    if month in {1, 2, 11, 12}:
        return 16.0
    if month in {6, 7, 8}:
        return 17.5
    return 17.0


def find_first_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise KeyError(f"None of the candidate columns exist: {candidates}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess all raw YDST hourly files into one Seoul daily dataset."
    )
    parser.add_argument("--input-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def clean_number(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(r"\s+", "", regex=True)
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "-": pd.NA})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def preprocess_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="cp949")
    datetime_col = find_first_column(df, ["시간", "일시"])
    result = pd.DataFrame(
        {
            "station": clean_number(df["지점"]).astype("Int64"),
            "datetime": pd.to_datetime(df[datetime_col], errors="coerce"),
            "ydst_pm10": clean_number(df["1시간평균 미세먼지농도(㎍/㎥)"]),
            "source_file": path.name,
        }
    )
    result = result.dropna(subset=["datetime"])
    result["date"] = result["datetime"].dt.normalize()
    result["hour"] = result["datetime"].dt.hour.astype(float)
    result["month"] = result["datetime"].dt.month
    result["close_hour"] = result["month"].map(scheduled_close_hour)
    result["last_entry_hour"] = result["month"].map(scheduled_last_entry_hour)
    result["is_open_hour_nominal"] = (
        (result["hour"] >= OPEN_START_HOUR) & (result["hour"] < result["close_hour"])
    ).astype(int)
    result["is_entry_hour_nominal"] = (
        (result["hour"] >= OPEN_START_HOUR) & (result["hour"] < result["last_entry_hour"])
    ).astype(int)
    return result


def main() -> None:
    args = parse_args()
    files = sorted(path for path in args.input_dir.glob("*.csv") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No CSV files found in {args.input_dir}")

    frames = [preprocess_file(path) for path in files]
    hourly = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["datetime", "source_file"])
        .drop_duplicates(subset=["datetime"], keep="last")
        .reset_index(drop=True)
    )

    daily = (
        hourly.groupby("date", as_index=False)
        .agg(
            station=("station", "first"),
            ydst_pm10_all_hours=("ydst_pm10", "mean"),
            ydst_pm10_count_all_hours=("ydst_pm10", "count"),
            ydst_pm10_open_hours=(
                "ydst_pm10",
                lambda values: values[hourly.loc[values.index, "is_open_hour_nominal"] == 1].mean(),
            ),
            ydst_pm10_count_open_hours=(
                "is_open_hour_nominal",
                "sum",
            ),
            ydst_pm10_entry_hours=(
                "ydst_pm10",
                lambda values: values[hourly.loc[values.index, "is_entry_hour_nominal"] == 1].mean(),
            ),
            ydst_pm10_count_entry_hours=(
                "is_entry_hour_nominal",
                "sum",
            ),
            close_hour=("close_hour", "first"),
            last_entry_hour=("last_entry_hour", "first"),
            source_files=("source_file", lambda values: ",".join(sorted(set(values)))),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["day"] = daily["date"].dt.day
    daily["dow"] = daily["date"].dt.weekday
    daily["dow_name"] = daily["date"].dt.day_name().map(
        {
            "Monday": "월",
            "Tuesday": "화",
            "Wednesday": "수",
            "Thursday": "목",
            "Friday": "금",
            "Saturday": "토",
            "Sunday": "일",
        }
    )
    daily["is_weekend"] = daily["dow"].isin([5, 6]).astype(int)
    daily["is_regular_closed_tuesday"] = (daily["dow"] == 1).astype(int)
    daily["scheduled_open_hours_nominal"] = daily["close_hour"] - OPEN_START_HOUR
    daily["scheduled_entry_hours_nominal"] = daily["last_entry_hour"] - OPEN_START_HOUR
    daily["ydst_pm10_high"] = (daily["ydst_pm10_open_hours"] >= 81).fillna(False).astype(int)
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")

    ordered_columns = [
        "date",
        "year",
        "month",
        "day",
        "dow",
        "dow_name",
        "is_weekend",
        "is_regular_closed_tuesday",
        "station",
        "close_hour",
        "last_entry_hour",
        "scheduled_open_hours_nominal",
        "scheduled_entry_hours_nominal",
        "ydst_pm10_all_hours",
        "ydst_pm10_count_all_hours",
        "ydst_pm10_open_hours",
        "ydst_pm10_count_open_hours",
        "ydst_pm10_entry_hours",
        "ydst_pm10_count_entry_hours",
        "ydst_pm10_high",
        "source_files",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    daily[ordered_columns].to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"files read  : {len(files):,}")
    print(f"hourly rows : {len(hourly):,}")
    print(f"daily rows  : {len(daily):,}")
    print(f"date range  : {daily['date'].min()} ~ {daily['date'].max()}")
    print(f"output path : {args.output}")


if __name__ == "__main__":
    main()
