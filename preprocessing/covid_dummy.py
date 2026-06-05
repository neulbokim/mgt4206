# ============================================================
# 코로나 더미 변수 생성
# output: ./data/preprocessed/covid_dummy_daily.csv
# ============================================================

import os
import pandas as pd


def main():
    # 프로젝트 루트 기준 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)

    output_dir = os.path.join(project_root, "data", "preprocessed")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "covid_dummy_daily.csv")

    # 기존 전처리 데이터 범위와 맞춤
    start_date = pd.Timestamp("2009-01-01")
    end_date = pd.Timestamp("2025-12-31")

    # 코로나 시작일: 국내 첫 확진일
    covid_start = pd.Timestamp("2020-01-20")

    # 종료 기준 3종
    covid_v1_end = pd.Timestamp("2023-05-31")  # 위기경보 해제·종식 선언
    covid_v2_end = pd.Timestamp("2023-01-30")  # 실내 마스크 의무 해제
    covid_v3_end = pd.Timestamp("2022-04-18")  # 사회적 거리두기 전면 해제

    df = pd.DataFrame({
        "date": pd.date_range(start_date, end_date, freq="D")
    })

    df["covid_v1"] = (
        (df["date"] >= covid_start) &
        (df["date"] <= covid_v1_end)
    ).astype(int)

    df["covid_v2"] = (
        (df["date"] >= covid_start) &
        (df["date"] <= covid_v2_end)
    ).astype(int)

    df["covid_v3"] = (
        (df["date"] >= covid_start) &
        (df["date"] <= covid_v3_end)
    ).astype(int)

    # 설명용 메타 컬럼
    df["covid_start_date"] = covid_start.strftime("%Y-%m-%d")
    df["covid_v1_end_date"] = covid_v1_end.strftime("%Y-%m-%d")
    df["covid_v2_end_date"] = covid_v2_end.strftime("%Y-%m-%d")
    df["covid_v3_end_date"] = covid_v3_end.strftime("%Y-%m-%d")

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"saved: {output_path}")
    print(df.shape)
    print(df[["covid_v1", "covid_v2", "covid_v3"]].sum())


if __name__ == "__main__":
    main()