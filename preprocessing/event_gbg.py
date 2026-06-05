# ./preprocessing/event_gbg.py

import re
import json
import argparse
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# 0. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "kh_royal_culture_events.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "preprocessed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


PALACES = ["경복궁", "창덕궁", "덕수궁", "창경궁", "경희궁", "종묘"]

PROGRAM_TYPE_GROUP_MAP = {
    "체험": "참여형",
    "교육": "참여형",
    "공연": "관람형",
    "재현": "관람형",
    "전시": "관람형",
    "공식행사": "행사형",
    "복합": "복합형",
}

KEYWORD_RULES = {
    "has_reward": ["리워드", "기념품", "상품", "선물"],
    "has_children": ["어린이", "아동", "가족", "키즈"],
    "has_night": ["야행", "밤", "야간", "한밤", "별빛"],
    "has_music": ["음악회", "공연", "연주", "국악", "콘서트"],
    "has_food": ["생과방", "다과", "음식", "차", "다례"],
    "has_guide": ["해설", "가이드", "투어", "탐방", "산책"],
    "has_hanbok": ["한복"],
    "has_reservation_notice": ["사전예약", "예약", "예매", "접수"],
    "has_fee_info": ["무료", "입장권", "입장료", "참가비", "유료"],
    "has_rain_notice": ["우천", "비", "순연"],
}


# ============================================================
# 1. 기본 유틸 함수
# ============================================================

def clean_text(value: Any) -> str:
    """문자열 공백 정리"""
    if pd.isna(value):
        return ""

    text = str(value)
    text = text.replace("\u00a0", " ")
    text = text.replace("", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_period(period: str) -> tuple[pd.Timestamp | pd.NaT, pd.Timestamp | pd.NaT]:
    """
    period 예시:
    - 2025-10-10~2025-10-10
    - 2025-04-26~2025-05-04
    """
    period = clean_text(period)

    if not period:
        return pd.NaT, pd.NaT

    match = re.search(
        r"(\d{4})[-.](\d{1,2})[-.](\d{1,2})\s*~\s*(\d{4})[-.](\d{1,2})[-.](\d{1,2})",
        period,
    )

    if not match:
        return pd.NaT, pd.NaT

    y1, m1, d1, y2, m2, d2 = match.groups()

    start_date = pd.to_datetime(f"{y1}-{m1}-{d1}", errors="coerce")
    end_date = pd.to_datetime(f"{y2}-{m2}-{d2}", errors="coerce")

    return start_date, end_date


def infer_season(start_date: pd.Timestamp | pd.NaT, title: str = "") -> str:
    """궁중문화축전 기준 봄/가을 구분"""
    title = clean_text(title)

    if "봄" in title:
        return "봄"
    if "가을" in title:
        return "가을"

    if pd.isna(start_date):
        return "미상"

    month = start_date.month

    if 3 <= month <= 6:
        return "봄"
    if 9 <= month <= 11:
        return "가을"

    return "기타"


def normalize_title(title: str) -> str:
    """행사명 기본 정제"""
    title = clean_text(title)

    # 앞쪽 연도/시즌 표현 일부 제거
    title = re.sub(r"^\d{4}년?\s*", "", title)
    title = re.sub(r"^(봄|가을)\s*", "", title)
    title = re.sub(r"^궁중문화축전\s*", "", title)

    return title.strip()


def make_title_base(title: str) -> str:
    """
    장소명이 붙은 프로그램을 묶기 위한 base title 생성.
    예:
    - 궁중놀이방-경복궁 -> 궁중놀이방
    - 경복궁 생과방 -> 생과방
    - 경복궁 수문장 교대의식 -> 수문장 교대의식
    """
    title = normalize_title(title)

    # suffix 형태 제거: -경복궁, _경복궁 등
    title = re.sub(r"\s*[-_/]\s*(경복궁|창덕궁|덕수궁|창경궁|경희궁|종묘)\s*$", "", title)

    # prefix 형태 제거: 경복궁 생과방
    title = re.sub(r"^(경복궁|창덕궁|덕수궁|창경궁|경희궁|종묘)\s+", "", title)

    return clean_text(title)


def normalize_program_type(row: pd.Series) -> str:
    """상세 유형 우선, 없으면 목록 유형 사용"""
    value = clean_text(row.get("program_type", ""))

    if not value:
        value = clean_text(row.get("program_type_from_list", ""))

    return value


def normalize_participation_type(value: str) -> str:
    value = clean_text(value)

    if not value:
        return "미상"

    if "사전" in value or "예약" in value or "예매" in value:
        return "사전예약"
    if "현장" in value:
        return "현장참여"
    if "온라인" in value:
        return "온라인"

    return value


def extract_first_time_info(text: str) -> dict:
    """
    time/schedule_summary에서 가장 먼저 등장하는 시간 정보 추출.
    완벽한 시간표 파싱보다 분석용 1차 변수 생성 목적.

    예:
    - 09:00 ~ 18:00
    - 10:00(20분) / 14:00(20분)
    """
    text = clean_text(text)

    result = {
        "has_time": 0,
        "start_time_first": "",
        "end_time_first": "",
        "duration_minutes_first": pd.NA,
    }

    if not text:
        return result

    times = re.findall(r"\d{1,2}:\d{2}", text)

    if times:
        result["has_time"] = 1
        result["start_time_first"] = times[0]

        if len(times) >= 2:
            result["end_time_first"] = times[1]

    duration_match = re.search(r"\((\d+)\s*분\)", text)
    if duration_match:
        result["duration_minutes_first"] = int(duration_match.group(1))

    return result


def extract_palaces_from_text(text: str) -> list[str]:
    """텍스트에서 궁/종묘명 추출"""
    text = clean_text(text)

    found = []
    for palace in PALACES:
        if palace in text:
            found.append(palace)

    return found


def make_place_list(row: pd.Series) -> list[str]:
    """
    place에서 궁궐명이 잡히면 그것을 사용.
    없으면 palace_group/detail_palace를 fallback으로 사용.
    """
    place = clean_text(row.get("place", ""))
    palace_group = clean_text(row.get("palace_group", ""))
    detail_palace = clean_text(row.get("detail_palace", ""))

    found = extract_palaces_from_text(place)

    if found:
        return found

    if palace_group:
        return [palace_group]

    if detail_palace:
        return [detail_palace]

    return []


def is_gyeongbokgung_event(row: pd.Series) -> bool:
    """경복궁 관련 행사 필터링"""
    fields = [
        row.get("palace_group", ""),
        row.get("detail_palace", ""),
        row.get("place", ""),
        row.get("title", ""),
        row.get("detail_title", ""),
    ]

    combined = " ".join(clean_text(x) for x in fields)
    return "경복궁" in combined


# ============================================================
# 2. 데이터 로드
# ============================================================

def load_raw_json(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    print(f"[INFO] Raw 데이터 로드 완료: {len(df):,} rows")
    print(f"[INFO] 입력 파일: {input_path}")

    return df


# ============================================================
# 3. events_clean 생성
# ============================================================

def build_events_clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # 문자열 컬럼 기본 정리
    text_cols = [
        "round_id",
        "round_label",
        "palace_group",
        "program_idx",
        "title",
        "program_type_from_list",
        "schedule_summary",
        "image_url_from_list",
        "image_alt",
        "detail_url",
        "detail_palace",
        "detail_title",
        "detail_category_in_title",
        "status",
        "program_type",
        "participation_type",
        "period",
        "time",
        "place",
        "capacity",
        "contact",
        "detail_text",
        "detail_image_url",
    ]

    for col in text_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].apply(clean_text)

    # 경복궁 행사만 필터링
    df = df[df.apply(is_gyeongbokgung_event, axis=1)].copy()

    print(f"[INFO] 경복궁 관련 행사 필터링 완료: {len(df):,} rows")

    # 기본 title 정리
    df["title_original"] = df["title"]
    df["title_clean"] = df["title"].apply(normalize_title)
    df["title_base"] = df["title"].apply(make_title_base)

    # 유형 정리
    df["program_type_clean"] = df.apply(normalize_program_type, axis=1)
    df["program_type_group"] = df["program_type_clean"].map(PROGRAM_TYPE_GROUP_MAP).fillna("기타")

    # 참여 방식 정리
    df["participation_type_clean"] = df["participation_type"].apply(normalize_participation_type)
    df["is_reservation_required"] = (df["participation_type_clean"] == "사전예약").astype(int)
    df["is_onsite"] = (df["participation_type_clean"] == "현장참여").astype(int)
    df["is_online"] = (df["participation_type_clean"] == "온라인").astype(int)

    # 날짜 파싱
    parsed_dates = df["period"].apply(parse_period)
    df["start_date"] = parsed_dates.apply(lambda x: x[0])
    df["end_date"] = parsed_dates.apply(lambda x: x[1])

    df["duration_days"] = (
        (df["end_date"] - df["start_date"]).dt.days + 1
    ).where(df["start_date"].notna() & df["end_date"].notna(), pd.NA)

    df["year"] = df["start_date"].dt.year.astype("Int64")
    df["month"] = df["start_date"].dt.month.astype("Int64")
    df["season"] = df.apply(
        lambda row: infer_season(row["start_date"], row["title_original"]),
        axis=1,
    )

    # 시간 정보
    time_source = (df["time"] + " " + df["schedule_summary"]).apply(clean_text)
    time_info = time_source.apply(extract_first_time_info).apply(pd.Series)

    df = pd.concat([df, time_info], axis=1)

    # 장소 정리
    df["place_clean"] = df["place"].replace("", pd.NA)
    df["place_clean"] = df["place_clean"].fillna(df["detail_palace"])
    df["place_clean"] = df["place_clean"].fillna(df["palace_group"])
    df["place_clean"] = df["place_clean"].fillna("")
    df["place_list"] = df.apply(make_place_list, axis=1)

    # 이미지 URL 통합
    df["image_url"] = df["detail_image_url"]
    df.loc[df["image_url"] == "", "image_url"] = df.loc[df["image_url"] == "", "image_url_from_list"]

    # 상세 텍스트 기반 키워드 변수
    full_text = (
        df["title_original"] + " "
        + df["program_type_clean"] + " "
        + df["participation_type_clean"] + " "
        + df["place"] + " "
        + df["detail_text"]
    ).apply(clean_text)

    for col, keywords in KEYWORD_RULES.items():
        pattern = "|".join(re.escape(k) for k in keywords)
        df[col] = full_text.str.contains(pattern, regex=True, na=False).astype(int)

    # 중복 제거
    df = df.drop_duplicates(subset=["round_id", "program_idx"], keep="first")

    # 분석에 자주 쓰는 컬럼 우선 배치
    preferred_cols = [
        "round_id",
        "round_label",
        "program_idx",
        "title_original",
        "title_clean",
        "title_base",
        "status",
        "program_type_clean",
        "program_type_group",
        "participation_type_clean",
        "is_reservation_required",
        "is_onsite",
        "is_online",
        "period",
        "start_date",
        "end_date",
        "duration_days",
        "year",
        "month",
        "season",
        "time",
        "schedule_summary",
        "has_time",
        "start_time_first",
        "end_time_first",
        "duration_minutes_first",
        "palace_group",
        "detail_palace",
        "place",
        "place_clean",
        "place_list",
        "capacity",
        "contact",
        "detail_text",
        "detail_url",
        "image_url",
        "image_alt",
        *KEYWORD_RULES.keys(),
    ]

    remain_cols = [col for col in df.columns if col not in preferred_cols]
    df = df[preferred_cols + remain_cols]

    print(f"[INFO] events_clean 생성 완료: {len(df):,} rows")

    return df


# ============================================================
# 4. events_long 생성
# ============================================================

def build_events_long(events_clean: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in events_clean.iterrows():
        place_list = row.get("place_list", [])

        if not isinstance(place_list, list) or len(place_list) == 0:
            place_list = ["경복궁"]

        # event_gbg이므로 경복궁만 남김
        if "경복궁" in place_list:
            place_list = ["경복궁"]
        else:
            continue

        for place_single in place_list:
            rows.append({
                "round_id": row["round_id"],
                "round_label": row["round_label"],
                "program_idx": row["program_idx"],
                "title_original": row["title_original"],
                "title_clean": row["title_clean"],
                "title_base": row["title_base"],
                "place_single": place_single,
                "program_type_clean": row["program_type_clean"],
                "program_type_group": row["program_type_group"],
                "participation_type_clean": row["participation_type_clean"],
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "duration_days": row["duration_days"],
                "year": row["year"],
                "month": row["month"],
                "season": row["season"],
                "has_time": row["has_time"],
                "start_time_first": row["start_time_first"],
                "end_time_first": row["end_time_first"],
                "duration_minutes_first": row["duration_minutes_first"],
                "detail_url": row["detail_url"],
                "image_url": row["image_url"],
                **{col: row[col] for col in KEYWORD_RULES.keys()},
            })

    df_long = pd.DataFrame(rows)

    print(f"[INFO] events_long 생성 완료: {len(df_long):,} rows")

    return df_long


# ============================================================
# 5. event_dates 생성
# ============================================================

def build_event_dates(events_long: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in events_long.iterrows():
        start_date = row["start_date"]
        end_date = row["end_date"]

        if pd.isna(start_date) or pd.isna(end_date):
            continue

        date_range = pd.date_range(start=start_date, end=end_date, freq="D")

        for date in date_range:
            rows.append({
                "date": date,
                "round_id": row["round_id"],
                "round_label": row["round_label"],
                "program_idx": row["program_idx"],
                "title_original": row["title_original"],
                "title_clean": row["title_clean"],
                "title_base": row["title_base"],
                "place_single": row["place_single"],
                "program_type_clean": row["program_type_clean"],
                "program_type_group": row["program_type_group"],
                "participation_type_clean": row["participation_type_clean"],
                "year": date.year,
                "month": date.month,
                "day": date.day,
                "weekday": date.day_name(),
                "weekday_kr": ["월", "화", "수", "목", "금", "토", "일"][date.weekday()],
                "is_weekend": int(date.weekday() >= 5),
                "season": row["season"],
                "has_time": row["has_time"],
                "start_time_first": row["start_time_first"],
                "end_time_first": row["end_time_first"],
                "duration_minutes_first": row["duration_minutes_first"],
                **{col: row[col] for col in KEYWORD_RULES.keys()},
            })

    df_dates = pd.DataFrame(rows)

    print(f"[INFO] event_dates 생성 완료: {len(df_dates):,} rows")

    return df_dates


# ============================================================
# 6. 날짜별 feature 생성
# ============================================================

def build_daily_features(event_dates: pd.DataFrame) -> pd.DataFrame:
    if event_dates.empty:
        return pd.DataFrame()

    base = (
        event_dates
        .groupby("date")
        .agg(
            event_count=("program_idx", "nunique"),
            program_instance_count=("program_idx", "count"),
            title_base_count=("title_base", "nunique"),
            has_time_count=("has_time", "sum"),
            reservation_event_count=("participation_type_clean", lambda x: (x == "사전예약").sum()),
            onsite_event_count=("participation_type_clean", lambda x: (x == "현장참여").sum()),
            online_event_count=("participation_type_clean", lambda x: (x == "온라인").sum()),
            night_event_count=("has_night", "sum"),
            children_event_count=("has_children", "sum"),
            music_event_count=("has_music", "sum"),
            food_event_count=("has_food", "sum"),
            guide_event_count=("has_guide", "sum"),
            reward_event_count=("has_reward", "sum"),
            hanbok_event_count=("has_hanbok", "sum"),
        )
        .reset_index()
    )

    # 유형별 count
    type_counts = pd.crosstab(
        event_dates["date"],
        event_dates["program_type_clean"],
    ).reset_index()

    type_counts = type_counts.rename(
        columns={
            col: f"type_{col}_count"
            for col in type_counts.columns
            if col != "date"
        }
    )

    # 참여 방식별 count
    participation_counts = pd.crosstab(
        event_dates["date"],
        event_dates["participation_type_clean"],
    ).reset_index()

    participation_counts = participation_counts.rename(
        columns={
            col: f"participation_{col}_count"
            for col in participation_counts.columns
            if col != "date"
        }
    )

    daily = base.merge(type_counts, on="date", how="left")
    daily = daily.merge(participation_counts, on="date", how="left")

    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["day"] = daily["date"].dt.day
    daily["weekday"] = daily["date"].dt.day_name()
    daily["weekday_kr"] = daily["date"].apply(lambda x: ["월", "화", "수", "목", "금", "토", "일"][x.weekday()])
    daily["is_weekend"] = daily["date"].dt.weekday.ge(5).astype(int)
    daily["season"] = daily["date"].apply(lambda x: infer_season(x, ""))

    daily = daily.sort_values("date").reset_index(drop=True)

    print(f"[INFO] daily_features 생성 완료: {len(daily):,} rows")

    return daily


# ============================================================
# 7. 저장
# ============================================================

def save_dataframe(df: pd.DataFrame, filename: str) -> None:
    path = OUTPUT_DIR / filename

    df_to_save = df.copy()

    # list 컬럼은 CSV 저장 가능하도록 문자열화
    for col in df_to_save.columns:
        if df_to_save[col].apply(lambda x: isinstance(x, list)).any():
            df_to_save[col] = df_to_save[col].apply(
                lambda x: ", ".join(x) if isinstance(x, list) else x
            )

    df_to_save.to_csv(path, index=False, encoding="utf-8-sig")

    print(f"[SAVE] {path}")


def save_outputs(
    events_clean: pd.DataFrame,
    events_long: pd.DataFrame,
    event_dates: pd.DataFrame,
    daily_features: pd.DataFrame,
) -> None:
    save_dataframe(events_clean, "events_gbg_clean.csv")
    save_dataframe(events_long, "events_gbg_long.csv")
    save_dataframe(event_dates, "event_dates_gbg.csv")
    save_dataframe(daily_features, "daily_event_features_gbg.csv")


# ============================================================
# 8. 실행부
# ============================================================

def main(input_path: str | None = None) -> None:
    if input_path is None:
        input_path = DEFAULT_INPUT_PATH
    else:
        input_path = Path(input_path)

    df_raw = load_raw_json(Path(input_path))

    events_clean = build_events_clean(df_raw)
    events_long = build_events_long(events_clean)
    event_dates = build_event_dates(events_long)
    daily_features = build_daily_features(event_dates)

    save_outputs(
        events_clean=events_clean,
        events_long=events_long,
        event_dates=event_dates,
        daily_features=daily_features,
    )

    print("\n[DONE] 경복궁 행사 전처리 완료")
    print(f"- events_clean: {len(events_clean):,} rows")
    print(f"- events_long: {len(events_long):,} rows")
    print(f"- event_dates: {len(event_dates):,} rows")
    print(f"- daily_features: {len(daily_features):,} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="raw JSON 파일 경로. 기본값: data/raw/kh_royal_culture_events.json",
    )

    args = parser.parse_args()
    main(input_path=args.input)