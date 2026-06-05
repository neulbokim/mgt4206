"""
한국형 관광기후지수(KTCI) 계산기
논문: 김남조·김상태(2014), "한국형 관광기후지수(KTCI)의 개발에 관한 연구"

KTCI 공식 (계절별):
- 연중: KTCI = 2(2.59Cd + 2.09Ca + 3.22P + 1.16W + 0.95S)
- 봄:   KTCI = 2(2.58Cd + 1.93Ca + 3.04P + 1.31W + 1.14S)
- 여름: KTCI = 2(3.07Cd + 1.90Ca + 3.27P + 0.90W + 0.86S)
- 가을: KTCI = 2(2.37Cd + 2.18Ca + 3.43P + 1.04W + 0.98S)
- 겨울: KTCI = 2(2.27Cd + 2.39Ca + 3.07P + 1.46W + 0.81S)

변수:
- Cd: 열적쾌적성(한낮) — 최고기온 + 최저상대습도 기반 Effective Temperature
- Ca: 열적쾌적성(일평균) — 평균기온 + 평균상대습도 기반 ET
- P : 강수량 점수 (0~5)
- W : 풍속 점수 (0~5)
- S : 일조/구름 점수 (0~5)
"""

import pandas as pd
import numpy as np
import glob
import os

# ─────────────────────────────────────────────
# 1. Effective Temperature(ET) 점수 계산
#    Mieczkowski(1985) 기준: 기온(℃)과 상대습도(%)를 ET로 변환 후 점수화
# ─────────────────────────────────────────────

def calc_ET(temp_c, rh_pct):
    """
    Effective Temperature(ET) 근사 계산
    Houghton & Yaglou(1923) 기반 간소화 공식
    ET ≈ T - 0.4 * (T - 10) * (1 - RH/100)
    """
    rh = np.clip(rh_pct, 0, 100)
    et = temp_c - 0.4 * (temp_c - 10) * (1 - rh / 100)
    return et


def et_to_score(et):
    """
    ET 값을 TCI 점수(-3 ~ 5)로 변환
    Mieczkowski(1985) 점수표 기반
    - 최적 ET 범위(17~24℃): 5점
    - 범위를 벗어날수록 감소
    - 논문 Table 1 기준: -20℃ ~ 48℃ → -3.0 ~ 5.0점
    """
    # 구간별 점수 매핑 (ET 상한 기준)
    # (ET_upper, score) — ET <= upper 이면 해당 점수
    score_table = [
        (-20, -3.0),
        (-16, -2.0),
        (-12, -1.0),
        (-8,   0.0),
        (-4,   1.0),
        (0,    1.5),
        (4,    2.0),
        (8,    2.5),
        (13,   3.0),
        (18,   3.5),
        (20,   4.0),
        (22,   4.5),
        (24,   5.0),  # 최적
        (26,   4.5),
        (28,   4.0),
        (30,   3.5),
        (32,   3.0),
        (34,   2.5),
        (36,   2.0),
        (38,   1.5),
        (40,   1.0),
        (42,   0.5),
        (44,   0.0),
        (46,  -1.0),
        (48,  -2.0),
    ]
    # ET > 48 → -3점
    if pd.isna(et):
        return np.nan
    for upper, score in score_table:
        if et <= upper:
            return score
    return -3.0


# ─────────────────────────────────────────────
# 2. 강수량 점수 (P)
# ─────────────────────────────────────────────

def precip_to_score(precip_mm):
    """
    월강수량(mm) → 점수(0~5)
    - 0mm       → 5.0
    - 15mm 증가마다 0.5 감소
    - 150mm 이상 → 0.0
    ※ 일별 데이터이므로 일강수량을 월강수량으로 환산 불가 →
      일강수량 기준 스케일 축소 적용 (150mm/월 ÷ 30일 ≈ 5mm/일)
      → 0mm→5점, 5mm 증가마다 0.5점 감소, 25mm이상→0점
    """
    if pd.isna(precip_mm):
        precip_mm = 0.0

    # 일강수량 기준 변환 (월기준 ÷ 30일 스케일)
    score = 5.0 - (precip_mm / 5.0) * 0.5
    return float(np.clip(score, 0.0, 5.0))


# ─────────────────────────────────────────────
# 3. 풍속 점수 (W)
# ─────────────────────────────────────────────

def wind_to_score(wind_ms):
    """
    평균풍속(m/s) → 점수(0~5)
    Mieczkowski(1985) 기준 그대로 적용
    """
    if pd.isna(wind_ms):
        return np.nan
    if wind_ms <= 0.8:
        return 5.0
    elif wind_ms <= 1.60:
        return 4.5
    elif wind_ms <= 2.51:
        return 4.0
    elif wind_ms <= 3.40:
        return 3.5
    elif wind_ms <= 5.50:
        return 3.0
    elif wind_ms <= 6.75:
        return 2.5
    elif wind_ms <= 8.00:
        return 2.0
    elif wind_ms <= 10.70:
        return 1.0
    else:
        return 0.0


# ─────────────────────────────────────────────
# 4. 일조/구름 점수 (S)
# ─────────────────────────────────────────────

def sunshine_to_score(sunshine_hr=None, cloud_10=None):
    """
    일조시간(hr) 또는 전운량(10분위) → 점수(0~5)
    - 일조시간 우선 사용
    - 없으면 전운량(0~10) 역변환: 구름 많을수록 점수 낮음
    """
    if sunshine_hr is not None and not pd.isna(sunshine_hr):
        # 일조시간 1시간 이하→0점, 1시간 증가마다 0.5점, 10시간이상→5점
        score = sunshine_hr * 0.5
        return float(np.clip(score, 0.0, 5.0))

    if cloud_10 is not None and not pd.isna(cloud_10):
        # 전운량 0(맑음)→5점, 10(흐림)→0점
        score = (10.0 - cloud_10) / 10.0 * 5.0
        return float(np.clip(score, 0.0, 5.0))

    return np.nan


# ─────────────────────────────────────────────
# 5. 계절 판별
#    - KTCI   : 월 기반 3개월 계절
#    - KTCI-a : 9일 이동평균 일평균기온 기반 기상학적 계절
# ─────────────────────────────────────────────

def get_season_by_month(month):
    """
    월 기반 계절 반환 (기후학적 기준)
    봄: 3~5월, 여름: 6~8월, 가을: 9~11월, 겨울: 12~2월
    """
    if month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    elif month in [9, 10, 11]:
        return 'autumn'
    else:
        return 'winter'


def find_stable_start(values, condition, end_index):
    """
    주어진 구간의 끝까지 조건을 계속 만족하는 최초 시작 인덱스 반환
    """
    for index in range(end_index):
        segment = values[index:end_index]
        segment = segment[~np.isnan(segment)]
        if len(segment) > 0 and np.all(condition(segment)):
            return index
    return None


def assign_temperature_based_seasons(daily):
    """
    9일 이동평균 일평균기온 기준으로 계절 재산정
    - 봄: 5℃ 이상으로 올라간 뒤 다시 떨어지지 않는 첫날
    - 여름: 20℃ 이상으로 올라간 뒤 다시 떨어지지 않는 첫날
    - 가을: 20℃ 미만으로 내려간 뒤 다시 올라가지 않는 첫날
    - 겨울: 5℃ 미만으로 내려간 뒤 다시 올라가지 않는 첫날
    """
    season_series = pd.Series(index=daily.index, dtype='object')

    for _, group in daily.groupby(daily['날짜'].dt.year, sort=True):
        group = group.sort_values('날짜')
        moving_avg = (
            group['평균기온']
            .rolling(window=9, center=True, min_periods=9)
            .mean()
            .to_numpy()
        )
        seasons = np.array(['winter'] * len(group), dtype=object)
        valid_positions = np.flatnonzero(~np.isnan(moving_avg))

        if len(valid_positions) == 0:
            season_series.loc[group.index] = seasons
            continue

        final_end = valid_positions[-1] + 1
        winter_start = find_stable_start(
            moving_avg, lambda arr: arr < 5.0, final_end
        )
        autumn_end = winter_start if winter_start is not None else final_end
        autumn_start = find_stable_start(
            moving_avg, lambda arr: arr < 20.0, autumn_end
        )
        summer_end = autumn_start if autumn_start is not None else autumn_end
        summer_start = find_stable_start(
            moving_avg, lambda arr: arr >= 20.0, summer_end
        )
        spring_end = summer_start if summer_start is not None else summer_end
        spring_start = find_stable_start(
            moving_avg, lambda arr: arr >= 5.0, spring_end
        )

        if spring_start is not None:
            seasons[spring_start:] = 'spring'
        if summer_start is not None:
            seasons[summer_start:] = 'summer'
        if autumn_start is not None:
            seasons[autumn_start:] = 'autumn'
        if winter_start is not None:
            seasons[winter_start:] = 'winter'

        season_series.loc[group.index] = seasons

    return season_series


# ─────────────────────────────────────────────
# 6. KTCI 계산 (계절별 가중치 적용)
# ─────────────────────────────────────────────

WEIGHTS = {
    'annual':  (2.59, 2.09, 3.22, 1.16, 0.95),
    'spring':  (2.58, 1.93, 3.04, 1.31, 1.14),
    'summer':  (3.07, 1.90, 3.27, 0.90, 0.86),
    'autumn':  (2.37, 2.18, 3.43, 1.04, 0.98),
    'winter':  (2.27, 2.39, 3.07, 1.46, 0.81),
}

WEATHER_COLUMNS = [
    '지점', '지점명', '일시', '기온(°C)', '강수량(mm)',
    '풍속(m/s)', '습도(%)', '일조(hr)', '전운량(10분위)'
]

REQUIRED_WEATHER_COLUMNS = set(WEATHER_COLUMNS)


def calc_ktci(cd, ca, p, w, s, season='annual'):
    """
    KTCI = 2 * (wCd*Cd + wCa*Ca + wP*P + wW*W + wS*S)
    각 변수 최대 5점 → 합산 최대 50 → ×2 = 100점
    """
    wCd, wCa, wP, wW, wS = WEIGHTS[season]
    score = 2.0 * (wCd * cd + wCa * ca + wP * p + wW * w + wS * s)
    return round(score, 2)


def calc_ktci_weighted(cd, ca, p, w, s, weights):
    """
    계절별 가중치를 직접 받아 KTCI 계산
    """
    w_cd, w_ca, w_p, w_w, w_s = weights
    score = 2.0 * (w_cd * cd + w_ca * ca + w_p * p + w_w * w + w_s * s)
    return round(score, 2)


def build_smoothed_season_weights(daily, season_col='계절_a', transition_days=14):
    """
    계절 전환 구간에서 이전/이후 계절 가중치를 선형 보간
    """
    weight_series = pd.Series(index=daily.index, dtype='object')

    for _, group in daily.groupby(daily['날짜'].dt.year, sort=True):
        group = group.sort_values('날짜')
        group_weights = [WEIGHTS[season] for season in group[season_col]]

        transition_points = []
        seasons = group[season_col].tolist()
        for position in range(1, len(group)):
            if seasons[position] != seasons[position - 1]:
                transition_points.append((position, seasons[position - 1], seasons[position]))

        for position, prev_season, next_season in transition_points:
            start = max(0, position - transition_days)
            end = min(len(group) - 1, position + transition_days)
            span = max(end - start, 1)

            prev_weights = np.array(WEIGHTS[prev_season], dtype=float)
            next_weights = np.array(WEIGHTS[next_season], dtype=float)

            for idx in range(start, end + 1):
                alpha = (idx - start) / span
                blended = tuple((1 - alpha) * prev_weights + alpha * next_weights)
                group_weights[idx] = blended

        weight_series.loc[group.index] = group_weights

    return weight_series


# ─────────────────────────────────────────────
# 7. 등급 분류
# ─────────────────────────────────────────────

def ktci_to_grade(ktci):
    """
    Mieczkowski(1985) 10단계 등급
    80~100: Ideal / Excellent / Very Good ...
    """
    if ktci >= 90:
        return 'Ideal'
    elif ktci >= 80:
        return 'Excellent'
    elif ktci >= 70:
        return 'Very Good'
    elif ktci >= 60:
        return 'Good'
    elif ktci >= 50:
        return 'Acceptable'
    elif ktci >= 40:
        return 'Marginal'
    elif ktci >= 30:
        return 'Unfavorable'
    elif ktci >= 20:
        return 'Very Unfavorable'
    elif ktci >= 10:
        return 'Extremely Unfavorable'
    else:
        return 'Impossible'


# ─────────────────────────────────────────────
# 8. 메인 처리 함수: 전처리된 일별 ASOS 데이터 사용
# ─────────────────────────────────────────────

ASOS_DAILY_COLUMNS = {
    'date',          # 날짜
    'month',         # 월
    'tavg',          # 평균기온(°C)
    'tmax',          # 최고기온(°C)
    'rain',          # 일강수량(mm)
    'wind_avg',      # 평균풍속(m/s)
    'humidity_avg',  # 평균습도(%)
    'sunshine',      # 일조시간(hr)
    'cloud_avg',     # 평균전운량(10분위)
}


def read_asos_daily_file(file_path):
    """
    ./data/preprocessed/asos_seoul_daily.csv 로딩

    예상 컬럼:
    - date, month, tavg, tmax, rain, wind_avg, humidity_avg, sunshine, cloud_avg

    주의:
    - 이 파일은 이미 일별 집계 데이터이므로 기존 기상청 시간별 CSV처럼
      일시/시각 기준으로 groupby하지 않는다.
    - 원자료에 '최저습도'가 없기 때문에 Cd 계산에는 humidity_avg를 대체값으로 사용한다.
    """
    df = pd.read_csv(file_path, encoding='utf-8-sig')

    missing = ASOS_DAILY_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"asos_seoul_daily.csv에 필요한 컬럼이 없습니다: {sorted(missing)}"
        )

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    numeric_cols = [
        'month', 'tavg', 'tmax', 'rain', 'wind_avg',
        'humidity_avg', 'sunshine', 'cloud_avg'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    return df


def process_asos_daily_file(file_path):
    """
    전처리된 서울 일별 ASOS CSV를 읽어 KTCI 계산
    """
    raw = read_asos_daily_file(file_path)

    daily = pd.DataFrame({
        '날짜': raw['date'],
        '월': raw['month'].fillna(raw['date'].dt.month).astype(int),
        '최고기온': raw['tmax'],
        '평균기온': raw['tavg'],
        # 일별 전처리 파일에는 최저습도가 없으므로 평균습도를 대체값으로 사용
        '최저습도': raw['humidity_avg'],
        '평균습도': raw['humidity_avg'],
        '강수량': raw['rain'].fillna(0),
        '풍속': raw['wind_avg'],
        '일조시간': raw['sunshine'],
        '전운량': raw['cloud_avg'],
    })

    # ── 각 점수 계산 ──────────────────────────
    # Cd: 한낮 열적쾌적성 점수
    # 원래는 최고기온 + 최저상대습도 기반이나,
    # 현재 일별 파일에는 최저습도가 없으므로 평균습도를 대체 적용한다.
    daily['ET_d'] = daily.apply(
        lambda r: calc_ET(r['최고기온'], r['최저습도']), axis=1
    )
    daily['Cd'] = daily['ET_d'].apply(et_to_score)

    # Ca: 일평균 열적쾌적성 점수
    daily['ET_a'] = daily.apply(
        lambda r: calc_ET(r['평균기온'], r['평균습도']), axis=1
    )
    daily['Ca'] = daily['ET_a'].apply(et_to_score)

    # P: 강수량 점수
    daily['P'] = daily['강수량'].apply(precip_to_score)

    # W: 풍속 점수
    daily['W'] = daily['풍속'].apply(wind_to_score)

    # S: 일조/구름 점수 (일조 우선, 없으면 전운량)
    def s_score(row):
        sun = row['일조시간']
        cloud = row['전운량']
        if not pd.isna(sun) and sun > 0:
            return sunshine_to_score(sunshine_hr=sun)
        return sunshine_to_score(cloud_10=cloud)

    daily['S'] = daily.apply(s_score, axis=1)

    # 결측값 보간
    for col in ['Cd', 'Ca', 'P', 'W', 'S']:
        daily[col] = daily[col].interpolate(method='linear').bfill().ffill()

    # 계절 판별
    daily['계절'] = daily['월'].apply(get_season_by_month)
    daily['계절_a'] = assign_temperature_based_seasons(daily)
    daily['가중치_a2'] = build_smoothed_season_weights(daily)

    # KTCI 계산
    daily['KTCI'] = daily.apply(
        lambda r: calc_ktci(r['Cd'], r['Ca'], r['P'], r['W'], r['S'], r['계절']),
        axis=1
    ).clip(0, 100).round(2)

    daily['KTCI-a'] = daily.apply(
        lambda r: calc_ktci(r['Cd'], r['Ca'], r['P'], r['W'], r['S'], r['계절_a']),
        axis=1
    ).clip(0, 100).round(2)

    daily['KTCI-a2'] = daily.apply(
        lambda r: calc_ktci_weighted(
            r['Cd'], r['Ca'], r['P'], r['W'], r['S'], r['가중치_a2']
        ),
        axis=1
    ).clip(0, 100).round(2)

    daily['KTCI-a_roll3'] = daily['KTCI-a'].rolling(window=3, min_periods=1).mean().round(2)
    daily['KTCI-a_roll7'] = daily['KTCI-a'].rolling(window=7, min_periods=1).mean().round(2)

    # 등급
    daily['등급'] = daily['KTCI'].apply(ktci_to_grade)
    daily['등급_a'] = daily['KTCI-a'].apply(ktci_to_grade)
    daily['등급_a2'] = daily['KTCI-a2'].apply(ktci_to_grade)

    return daily.drop(columns=['가중치_a2'])


def make_output(daily):
    """
    저장용 컬럼 정리
    """
    output = daily[[
        '날짜',
        'KTCI', '등급', '계절',
        'KTCI-a', '등급_a', '계절_a',
        'KTCI-a2', '등급_a2',
        'KTCI-a_roll3', 'KTCI-a_roll7',
        'Cd', 'Ca', 'P', 'W', 'S',
        'ET_d', 'ET_a',
        '최고기온', '평균기온', '최저습도', '평균습도',
        '강수량', '풍속', '일조시간', '전운량'
    ]].copy()

    output['날짜'] = pd.to_datetime(output['날짜']).dt.strftime('%Y-%m-%d')
    return output


# ─────────────────────────────────────────────
# 9. 실행
# ─────────────────────────────────────────────

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)

    input_path = os.path.join(
        project_root, 'data', 'preprocessed', 'asos_seoul_daily.csv'
    )
    output_path = os.path.join(
        project_root, 'data', 'preprocessed', 'ktci.csv'
    )

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

    print(f"입력 파일: {input_path}")
    print("KTCI 계산 중...")

    daily = process_asos_daily_file(input_path)
    output = make_output(daily)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"완료: {output_path}")
    print(f"총 {len(output):,}일")
    print("\n=== 결과 미리보기 ===")
    print(
        output[[
            '날짜', 'KTCI', '등급', '계절',
            'KTCI-a', '등급_a', '계절_a', 'KTCI-a2'
        ]].head(10).to_string(index=False)
    )

    print("\n=== KTCI 기술통계 ===")
    print(output['KTCI'].describe())
