from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "visitors"
OUTPUT_DIR = PROJECT_ROOT / "data" / "preprocessed"
OUTPUT_PATH = OUTPUT_DIR / "visitors_gbg_daily.csv"
HOLIDAY_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "holiday_calendar_2009_2026.csv"

GBG_COLUMN_CANDIDATES = {
    "paid": ["경복궁(유료)"],
    "free": ["경복궁(무료)"],
    "domestic": ["경복궁(내국인)"],
    "english": ["경복궁(영어권)"],
    "japanese": ["경복궁(일본어권)", "경복궁(일어권)"],
    "chinese": ["경복궁(중국어권)", "경복궁(중어권)"],
    "other_foreign": ["경복궁(기타외국인)"],
}

OTHER_PALACE_VISITOR_COLUMNS = {
    "cdg_visitors": {
        "paid": ["창덕궁(유료)"],
        "free": ["창덕궁(무료)"],
    },
    "cgg_visitors": {
        "paid": ["창경궁(유료)"],
        "free": ["창경궁(무료)"],
    },
    "dsg_visitors": {
        "paid": ["덕수궁(유료)"],
        "free": ["덕수궁(무료)"],
    },
}

VISITOR_VALUE_COLUMNS = [
    "visitors",
    "cdg_visitors",
    "cgg_visitors",
    "dsg_visitors",
    "paid",
    "free",
    "domestic",
    "english",
    "japanese",
    "chinese",
    "other_foreign",
    "foreign_total",
]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess all raw 4-palace visitor files into a Gyeongbokgung-only daily dataset."
    )
    parser.add_argument("--input-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--holiday-file", type=Path, default=HOLIDAY_RAW_PATH)
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def clean_number_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace('"', "", regex=False)
        .str.replace("'", "", regex=False)
        .str.replace("=", "", regex=False)
        .str.replace(r"\s+", "", regex=True)
    )
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "-": pd.NA})
    return pd.to_numeric(cleaned, errors="coerce").fillna(0).astype(int)


def find_first_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise KeyError(f"None of the candidate columns exist: {candidates}")


def extract_year_from_name(path: Path) -> int:
    match = re.search(r"(20\d{2})", path.name)
    if not match:
        raise ValueError(f"Could not extract year from filename: {path.name}")
    return int(match.group(1))


def load_holiday_dates(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"공휴일 raw 파일이 없습니다: {path}. 먼저 `venv/bin/python data/holiday_scraper.py`를 실행하세요."
        )

    holiday_df = pd.read_csv(path, encoding="utf-8-sig")
    if "date" not in holiday_df.columns:
        raise ValueError(f"`date` 컬럼이 없습니다: {path}")

    if "is_holiday" in holiday_df.columns:
        holiday_df = holiday_df[holiday_df["is_holiday"] == 1]

    return set(pd.to_datetime(holiday_df["date"]).dt.strftime("%Y-%m-%d"))


def add_missing_calendar_days(df: pd.DataFrame) -> pd.DataFrame:
    full_index = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    result = df.set_index("date").reindex(full_index).rename_axis("date").reset_index()

    inserted_mask = result["visitors"].isna()
    for column in VISITOR_VALUE_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0).astype(int)

    result["source_year"] = result["source_year"].fillna(result["date"].dt.year).astype(int)
    result["source_file"] = result["source_file"].fillna("filled_missing_dates")

    result["year"] = result["date"].dt.year
    result["month"] = result["date"].dt.month
    result["day"] = result["date"].dt.day
    result["dow"] = result["date"].dt.weekday
    result["dow_name"] = result["date"].dt.day_name().map(
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
    result["is_weekend"] = result["dow"].isin([5, 6]).astype(int)
    result["is_regular_closed_tuesday"] = (result["dow"] == 1).astype(int)
    result["closed"] = (result["visitors"] == 0).astype(int)
    result["others_closed"] = (
        (result["cdg_visitors"] == 0).astype(int)
        + (result["cgg_visitors"] == 0).astype(int)
        + (result["dsg_visitors"] == 0).astype(int)
    ).astype(int)
    result["close_hour"] = result["month"].map(scheduled_close_hour)
    result["last_entry_hour"] = result["month"].map(scheduled_last_entry_hour)
    result["scheduled_open_hours_nominal"] = result["close_hour"] - 9.0
    result["scheduled_entry_hours_nominal"] = result["last_entry_hour"] - 9.0

    print(f"missing days filled with 0: {int(inserted_mask.sum()):,}")
    return result


def preprocess_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="cp949")
    df.columns = [normalize_text(column) for column in df.columns]

    date_column = find_first_column(df, ["일자", "날짜", "일시"])
    result = pd.DataFrame({"date": pd.to_datetime(df[date_column], errors="coerce")})
    if result["date"].isna().any():
        raise ValueError(f"Invalid dates found in {path.name}")

    for output_name, candidates in GBG_COLUMN_CANDIDATES.items():
        source_column = find_first_column(df, candidates)
        result[output_name] = clean_number_series(df[source_column])

    for output_name, palace_candidates in OTHER_PALACE_VISITOR_COLUMNS.items():
        paid_column = find_first_column(df, palace_candidates["paid"])
        free_column = find_first_column(df, palace_candidates["free"])
        result[output_name] = (
            clean_number_series(df[paid_column]) + clean_number_series(df[free_column])
        )

    result["foreign_total"] = (
        result["english"] + result["japanese"] + result["chinese"] + result["other_foreign"]
    )
    result["visitors"] = result["paid"] + result["free"]
    result["source_file"] = path.name
    result["source_year"] = extract_year_from_name(path)

    result["year"] = result["date"].dt.year
    result["month"] = result["date"].dt.month
    result["day"] = result["date"].dt.day
    result["dow"] = result["date"].dt.weekday
    result["dow_name"] = result["date"].dt.day_name().map(
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
    result["is_weekend"] = result["dow"].isin([5, 6]).astype(int)
    result["is_regular_closed_tuesday"] = (result["dow"] == 1).astype(int)
    result["closed"] = (result["visitors"] == 0).astype(int)
    result["others_closed"] = (
        (result["cdg_visitors"] == 0).astype(int)
        + (result["cgg_visitors"] == 0).astype(int)
        + (result["dsg_visitors"] == 0).astype(int)
    ).astype(int)
    result["d_holiday"] = 0
    result["close_hour"] = result["month"].map(scheduled_close_hour)
    result["last_entry_hour"] = result["month"].map(scheduled_last_entry_hour)
    result["scheduled_open_hours_nominal"] = result["close_hour"] - 9.0
    result["scheduled_entry_hours_nominal"] = result["last_entry_hour"] - 9.0

    ordered_columns = [
        "date",
        "year",
        "month",
        "day",
        "dow",
        "dow_name",
        "is_weekend",
        "is_regular_closed_tuesday",
        "closed",
        "others_closed",
        "d_holiday",
        "close_hour",
        "last_entry_hour",
        "scheduled_open_hours_nominal",
        "scheduled_entry_hours_nominal",
        "visitors",
        "cdg_visitors",
        "cgg_visitors",
        "dsg_visitors",
        "paid",
        "free",
        "domestic",
        "english",
        "japanese",
        "chinese",
        "other_foreign",
        "foreign_total",
        "source_year",
        "source_file",
    ]
    return result[ordered_columns]


def main() -> None:
    args = parse_args()
    files = sorted(path for path in args.input_dir.glob("*.csv") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No CSV files found in {args.input_dir}")

    holiday_dates = load_holiday_dates(args.holiday_file)
    frames = [preprocess_file(path) for path in files]
    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["date", "source_year"])
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    combined = add_missing_calendar_days(combined)
    combined["d_holiday"] = combined["date"].dt.strftime("%Y-%m-%d").isin(holiday_dates).astype(int)
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"files read  : {len(files):,}")
    print(f"rows saved  : {len(combined):,}")
    print(f"date range  : {combined['date'].min()} ~ {combined['date'].max()}")
    print(f"output path : {args.output}")


if __name__ == "__main__":
    main()
