from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "holiday_calendar_2009_2026.csv"
HOLIDAY_API_URL = "https://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Korean public holidays from KASI API and save to data/raw."
    )
    parser.add_argument("--start-year", type=int, default=2009)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=RAW_OUTPUT_PATH)
    return parser.parse_args()


def load_service_key(project_root: Path) -> str:
    env_path = project_root / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "SERVICE_KEY":
                return unquote(value.strip().strip('"').strip("'"))

    env_value = os.getenv("SERVICE_KEY")
    if env_value:
        return unquote(env_value)

    raise ValueError("SERVICE_KEY를 찾을 수 없습니다. .env 또는 환경변수에 설정하세요.")


def parse_items(response: requests.Response) -> list[dict[str, object]]:
    content_type = response.headers.get("content-type", "")

    if "json" in content_type.lower():
        payload = response.json()
        body = payload.get("response", {}).get("body", {})
        items_container = body.get("items", {})
        if isinstance(items_container, str):
            return []
        items = items_container.get("item", [])
        if isinstance(items, dict):
            items = [items]
        if isinstance(items, str):
            return []
        return items

    root = ET.fromstring(response.text)
    items: list[dict[str, object]] = []
    for item in root.findall(".//item"):
        items.append(
            {
                "dateKind": item.findtext("dateKind"),
                "dateName": item.findtext("dateName"),
                "isHoliday": item.findtext("isHoliday"),
                "locdate": item.findtext("locdate"),
                "seq": item.findtext("seq"),
            }
        )
    return items


def fetch_holidays(start_year: int, end_year: int, service_key: str) -> pd.DataFrame:
    session = requests.Session()
    rows: list[dict[str, object]] = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            response = session.get(
                HOLIDAY_API_URL,
                params={
                    "ServiceKey": service_key,
                    "solYear": str(year),
                    "solMonth": f"{month:02d}",
                    "_type": "json",
                    "numOfRows": "100",
                },
                timeout=30,
            )
            response.raise_for_status()

            for item in parse_items(response):
                locdate = str(item.get("locdate", "") or "").strip()
                if len(locdate) != 8:
                    continue
                rows.append(
                    {
                        "date": f"{locdate[:4]}-{locdate[4:6]}-{locdate[6:8]}",
                        "locdate": locdate,
                        "year": int(locdate[:4]),
                        "month": int(locdate[4:6]),
                        "day": int(locdate[6:8]),
                        "date_name": str(item.get("dateName", "") or "").strip(),
                        "is_holiday": 1 if str(item.get("isHoliday", "") or "").strip() == "Y" else 0,
                        "date_kind": str(item.get("dateKind", "") or "").strip(),
                        "seq": str(item.get("seq", "") or "").strip(),
                    }
                )

    holiday_df = pd.DataFrame(rows)
    if holiday_df.empty:
        raise ValueError("공휴일 API 응답이 비어 있습니다.")

    holiday_df = (
        holiday_df.sort_values(["date", "seq"])
        .drop_duplicates(subset=["date", "date_name"], keep="first")
        .sort_values(["date", "date_name"])
        .reset_index(drop=True)
    )
    return holiday_df


def main() -> None:
    args = parse_args()
    service_key = load_service_key(PROJECT_ROOT)
    holiday_df = fetch_holidays(args.start_year, args.end_year, service_key)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    holiday_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"rows saved  : {len(holiday_df):,}")
    print(f"date range  : {holiday_df['date'].min()} ~ {holiday_df['date'].max()}")
    print(f"output path : {args.output}")


if __name__ == "__main__":
    main()
