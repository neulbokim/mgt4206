# 경복궁 방문객 시계열 분석 프로젝트

**경복궁 일별 방문객 수**를 **종속변수(y)**로 두고, 
**날씨·대기질·휴일·코로나·이벤트·KTCI 계열 변수**를 결합해 **SARIMAX 모델링**을 수행하는 프로젝트입니다.

## 디렉터리 구조

```text
code/
├─ data/
│  ├─ holiday_scraper.py
│  ├─ raw/
│  │  ├─ asos/
│  │  ├─ holiday_calendar_2009_2026.csv
│  │  ├─ visitors/
│  │  ├─ ydst/
│  │  ├─ kh_royal_culture_events.csv
│  │  └─ kh_royal_culture_events.json
│  ├─ preprocessed/
│  └─ modeling/
├─ preprocessing/
│  ├─ visitors_gbg.py
│  ├─ asos_daily.py
│  ├─ ydst_daily.py
│  ├─ covid_dummy.py
│  ├─ event_gbg.py
│  └─ ktci_calculator.py
├─ modeling/
│  ├─ eda_covid_clean.ipynb
│  └─ sarimax.ipynb
├─ TEAM_RUN_GUIDE.md
├─ requirements.txt
└─ README.md
```

## 1. 데이터 수집

### 원천 데이터

- `data/raw/visitors/`
  - 4대궁 관람객 수 현황 연도별 CSV
  - 경복궁 분석에 필요한 종속변수의 원천
- `data/raw/asos/`
  - 서울 ASOS 일자료 CSV
  - 기온, 강수, 적설, 풍속, 습도, 일조, 일사 등 기상 변수 원천
- `data/raw/ydst/`
  - 서울 시간단위 PM10 자료
  - 관람시간 기준 일평균 대기질 변수 원천
- `data/raw/kh_royal_culture_events.csv`
- `data/raw/kh_royal_culture_events.json`
  - 경복궁 관련 행사 원천
- `data/holiday_scraper.py`
  - 한국천문연구원 특일 정보 API에서 공휴일 raw 파일 수집
  - 결과 파일: `data/raw/holiday_calendar_2009_2026.csv`

### 변수 역할

- **종속변수**
  - `visitors`: 경복궁 일별 총 방문객 수

- **설명변수 / 독립변수 후보**
  - 기상: `tavg`, `tmin`, `tmax`, `rain`, `snow_*`, `pleasant`, `pleasant_streak`
  - 대기질: `pm10` (`ydst_pm10_open_hours`)
  - KTCI 계열: `ktci`, `ktci_a`, `ktci_a2`, `ktci_a_roll3`, `ktci_a_roll7`
  - 분해형 KTCI 성분: `Cd`, `Ca`, `P`, `W`, `S`

- **외생변수 후보**
  - 운영/휴일: `closed`, `others_closed`, `d_holiday`, `is_weekend`
  - 코로나: `covid_v1`, `covid_v2`, `covid_v3`
  - 이벤트: `event_day`, `event_count`, `night_event_count`
  - 연휴 파생: `offdays_left`, `long_break_3p`, `pre_holiday`

### 현재 분석 단위

- 공간 단위: `경복궁`
- 시간 단위: `일별`
- 분석 기간: `2009-01-01 ~ 2025-12-31`

## 2. 데이터 전처리

전처리 스크립트는 모두 `code/` 루트에서 실행한다.

### 전처리 스크립트

- `data/holiday_scraper.py`
  - 공휴일 raw 파일 수집
- `preprocessing/visitors_gbg.py`
  - 경복궁 방문객 데이터 생성
  - 누락된 방문객 달력 구간(`2020-05-30 ~ 2020-07-21`)은 `0`으로 채움
  - `closed`, `others_closed`, `d_holiday` 계산
- `preprocessing/asos_daily.py`
  - ASOS 일자료 정리
- `preprocessing/ydst_daily.py`
  - 시간단위 PM10을 관람시간 기준 일자료로 집계
- `preprocessing/covid_dummy.py`
  - 코로나 더미 생성
- `preprocessing/event_gbg.py`
  - 행사 변수 생성
- `preprocessing/ktci_calculator.py`
  - KTCI 계열 변수 생성

### 주요 산출물

- `data/raw/holiday_calendar_2009_2026.csv`
- `data/preprocessed/visitors_gbg_daily.csv`
- `data/preprocessed/asos_seoul_daily.csv`
- `data/preprocessed/ydst_seoul_daily.csv`
- `data/preprocessed/covid_dummy_daily.csv`
- `data/preprocessed/daily_event_features_gbg.csv`
- `data/preprocessed/ktci.csv`
- `data/preprocessed/eda_gbg_master.csv`

### 전처리 결과 요약

- `visitors_gbg_daily.csv`
  - 방문객, 요일, 휴일, 운영 변수 포함
  - 53일 누락 구간은 `filled_missing_dates`로 채워짐
- `asos_seoul_daily.csv`
  - 기온, 강수, 적설, 풍속, 습도, 일조, 일사 포함
- `ydst_seoul_daily.csv`
  - `ydst_pm10_open_hours`를 대표 PM10 변수로 사용
- `eda_gbg_master.csv`
  - EDA와 모델링이 공통으로 쓰는 통합 입력 파일

## 3. EDA

EDA 노트북은 `modeling/eda_covid_clean.ipynb`다.

### 목적

- 날짜 누락, 중복, 0 방문객 구간 확인
- 경복궁 방문객 시계열의 장기 추세와 계절성 확인
- 월별 / 요일별 / 코로나 구간별 수준 차이 확인
- 날씨, PM10, 이벤트, KTCI 계열과 방문객의 관계 점검
- 모델링에 넣을 변수 블록 정리

### 입력 / 출력

- 입력: 전처리 산출물 6종
  - `visitors_gbg_daily.csv`
  - `asos_seoul_daily.csv`
  - `ydst_seoul_daily.csv`
  - `covid_dummy_daily.csv`
  - `daily_event_features_gbg.csv`
  - `ktci.csv`
- 출력:
  - `data/preprocessed/eda_gbg_master.csv`

### 현재 EDA에서 확인하는 핵심 항목

- 방문객 시계열 + 이동평균
- 연도별 방문객 합계
- 월별 / 요일별 패턴
- 코로나 단계별 방문객 수준
- 결측률과 coverage
- 변수 블록 후보

## 4. 모델링

모델링 노트북은 `modeling/sarimax.ipynb`다.

### 목적

- SARIMAX 기반 예측 성능 비교
- baseline 대비 KTCI 계열이 얼마나 개선되는지 확인
- decomposition model과 rolling KTCI의 성능 비교
- 이벤트 변수 포함/미포함 비교
- 코로나 더미 사양 비교

### 입력

- `data/preprocessed/eda_gbg_master.csv`

### 현재 비교하는 모델

- `Baseline (Temperature)`
- `KTCI Model`
- `KTCI-a Model`
- `KTCI-a2 Model`
- `KTCI-a_roll3 Model`
- `KTCI-a_roll7 Model`
- `Decomposition Reduced Model`

### 현재 모델에서 쓰는 주요 외생변수

- 공통 통제
  - `rain`, `pm10`, `closed`, `others_closed`, `is_weekend`, `d_holiday`
- 연휴 파생
  - `offdays_left`, `long_break_3p`, `pre_holiday`
- 코로나
  - 기본 `covid_v3`, 비교 시 `covid_v1/v2/v3`
- 이벤트
  - `event_day`, `event_count`, `night_event_count`

### 진단 및 평가

- 정상성: `ADF`
- 공선성: `VIF`
- 성능: `RMSE`, `MAE`, `R²`, `Adjusted R²`
- 잔차 진단: `ACF`, `PACF`, `Ljung-Box`
- test 구간 시각화

## 환경 설정

```bash
# 가상환경 생성
python3 -m venv .venv

# 가상환경 활성화
source .venv/bin/activate # MacOS인 경우
.\.venv\Scripts\activate # Windows인 경우

# 패키지 설치
pip install -r requirements.txt
```

## 실행 가이드

팀원용 상세 실행 절차는 `TEAM_RUN_GUIDE.md`를 참고해주세요.

## 프로젝트 정보
- 2026-1학기 경영환경에서의시계열분석과예측(캡스톤디자인)(MGT4206) 팀 프로젝트
- 프로젝트 기간: 2026년 5월 14일 ~ 2026년 6월 23일
- 팀원: 
  - 서강대학교 국어국문학과 22학번 김현서 (neulbokim@sogang.ac.kr)
  - 서강대학교 경제학과 20학번 구준모 (kjms1019@gmail.com)
  - 서강대학교 미디어&엔터테인먼트학과 21학번 남윤서 (namseo4664@naver.com)
  - 서강대학교 경영학과 21학번 박준상 (junsang2sc@naver.com)
  - 서강대학교 경영학과 23학번 서정인 (jungin0307@sogang.ac.kr)
  - 서강대학교 경영학과 23학번 신연아 (ya_ehf1210@naver.com)