from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "asos"
OUTPUT_DIR = PROJECT_ROOT / "data" / "preprocessed"
OUTPUT_PATH = OUTPUT_DIR / "asos_seoul_daily.csv"


def find_first_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise KeyError(f"None of the candidate columns exist: {candidates}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess all raw ASOS daily files into one Seoul daily dataset."
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
    sunshine_col = find_first_column(df, ["합계 일조 시간(hr)", "합계 일조시간(hr)"])
    solar_col = find_first_column(df, ["합계 일사(MJ/m2)", "합계 일사량(MJ/m2)"])
    selected = pd.DataFrame(
        {
            "station": clean_number(df["지점"]).astype("Int64"),
            "date": pd.to_datetime(df["일시"], errors="coerce"),
            "tavg": clean_number(df["평균기온(°C)"]),
            "tmin": clean_number(df["최저기온(°C)"]),
            "tmax": clean_number(df["최고기온(°C)"]),
            "rain": clean_number(df["일강수량(mm)"]),
            "wind_avg": clean_number(df["평균 풍속(m/s)"]),
            "humidity_avg": clean_number(df["평균 상대습도(%)"]),
            "sunshine": clean_number(df[sunshine_col]),
            "cloud_avg": clean_number(df["평균 전운량(1/10)"]),
            "solar_total": clean_number(df[solar_col]),
            "snow_fresh_max": clean_number(df["일 최심신적설(cm)"]),
            "snow_depth_max": clean_number(df["일 최심적설(cm)"]),
            "snow_fresh_3h_total": clean_number(df["합계 3시간 신적설(cm)"]),
            "source_file": path.name,
        }
    )
    selected["rain"] = selected["rain"].fillna(0)
    selected["snow_fresh_max"] = selected["snow_fresh_max"].fillna(0)
    selected["snow_depth_max"] = selected["snow_depth_max"].fillna(0)
    selected["snow_fresh_3h_total"] = selected["snow_fresh_3h_total"].fillna(0)
    return selected


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
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
    result["rain_yn"] = (result["rain"] > 0).astype(int)
    result["snow_yn"] = (
        (result["snow_fresh_max"] > 0)
        | (result["snow_depth_max"] > 0)
        | (result["snow_fresh_3h_total"] > 0)
    ).astype(int)
    result["pleasant"] = result["tavg"].between(5, 20, inclusive="both").fillna(False).astype(int)

    streak = []
    current = 0
    for flag in result["pleasant"].tolist():
        current = current + 1 if flag == 1 else 0
        streak.append(current)
    result["pleasant_streak"] = streak
    return result


def main() -> None:
    args = parse_args()
    files = sorted(path for path in args.input_dir.glob("*.csv") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No CSV files found in {args.input_dir}")

    frames = [preprocess_file(path) for path in files]
    combined = (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["date"])
        .sort_values(["date", "source_file"])
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    combined = add_derived_columns(combined)
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")

    ordered_columns = [
        "date",
        "year",
        "month",
        "day",
        "dow",
        "dow_name",
        "is_weekend",
        "station",
        "tavg",
        "tmin",
        "tmax",
        "rain",
        "rain_yn",
        "snow_fresh_max",
        "snow_depth_max",
        "snow_fresh_3h_total",
        "snow_yn",
        "pleasant",
        "pleasant_streak",
        "wind_avg",
        "humidity_avg",
        "sunshine",
        "cloud_avg",
        "solar_total",
        "source_file",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined[ordered_columns].to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"files read  : {len(files):,}")
    print(f"rows saved  : {len(combined):,}")
    print(f"date range  : {combined['date'].min()} ~ {combined['date'].max()}")
    print(f"output path : {args.output}")


if __name__ == "__main__":
    main()
