# 경복궁 방문객 시계열 분석 프로젝트

**경복궁 일별 방문객 수**를 **종속변수**로 두고,  
**날씨·대기질·휴일·코로나·KTCI 계열 변수**를 결합해 **SARIMAX 모델링**을 수행하는 프로젝트입니다.

- `modeling/timeseries.ipynb`: 발표용으로 정리한 핵심 노트북
- `modeling/timeseries_2015.ipynb`: 2015년 이후 구간만 사용한 발표용 변형 노트북
- `modeling/sarimax.ipynb`: 전체 실험과 비교를 포함한 상세 분석 노트북

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
│  ├─ preprocessed/
│  └─ modeling/
├─ preprocessing/
│  ├─ visitors_gbg.py
│  ├─ asos_daily.py
│  ├─ ydst_daily.py
│  ├─ covid_dummy.py
│  └─ ktci_calculator.py
├─ modeling/
│  ├─ granger_cointegration_tests.py
│  ├─ no_event_granger_cointegration_tests.py
│  ├─ no_event_sarimax_performance.py
│  ├─ sarimax.ipynb
│  ├─ timeseries.ipynb
│  └─ timeseries_presentation_outline.md
├─ data/
│  └─ eda.ipynb
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
- `preprocessing/ktci_calculator.py`
  - KTCI 계열 변수 생성

### 주요 산출물

- `data/raw/holiday_calendar_2009_2026.csv`
- `data/preprocessed/visitors_gbg_daily.csv`
- `data/preprocessed/asos_seoul_daily.csv`
- `data/preprocessed/ydst_seoul_daily.csv`
- `data/preprocessed/covid_dummy_daily.csv`
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

EDA 노트북은 `data/eda.ipynb`다.

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

모델링 노트북은 `modeling/timeseries.ipynb`다.

### 목적

- SARIMAX 기반 예측 성능 비교
- `baseline -> KTCI -> KTCI-a -> KTCI-a-roll7` 서사를 중심으로 성능 비교
- 한국형 계절 재정의가 예측력을 높이는지 확인
- ADF, VIF, Granger, 잔차 진단으로 모델 타당성 점검
- 공적분은 본문 메인 근거가 아니라 보조 확인용으로만 취급

### 입력

- `data/preprocessed/eda_gbg_master.csv`

### 현재 비교하는 모델

- `Baseline (Temperature)`
- `KTCI Model`
- `KTCI-a Model`
- `KTCI-a-roll7 Model`

### 보조 분석 스크립트

- `modeling/granger_cointegration_tests.py`
  - 전체 구간에서 Granger causality와 cointegration을 함께 점검
  - `outputs/granger_*`와 `outputs/cointegration_*` 요약 파일 생성
- `modeling/no_event_granger_cointegration_tests.py`
  - 이벤트 변수를 제외한 no-event 사양에서 Granger/cointegration 비교
- `modeling/no_event_sarimax_performance.py`
  - no-event 사양의 SARIMAX 성능 비교

### 주요 산출물

- `outputs/granger_lag_results_no_event.csv`
- `outputs/granger_summary_no_event.csv`
- `outputs/sarimax_scores_no_event.csv`
- `outputs/adf_results_no_event.csv`
- `outputs/cointegration_results_no_event.csv`
- `outputs/figures/00_no_event_model_storyline.png`
- `outputs/figures/01_no_event_granger_fstat.png`
- `outputs/figures/02_no_event_ktci_path_granger.png`
- `outputs/figures/03_no_event_decomposition_granger.png`
- `outputs/figures/04_no_event_cointegration_neglogp.png`
- `outputs/figures/05_no_event_cointegration_tstat.png`
- `outputs/figures/06_no_event_adf_stationarity.png`
- `outputs/figures/07_no_event_season_reassignment.png`
- `outputs/figures/08_no_event_sarimax_performance.png`

### 현재 모델에서 쓰는 주요 외생변수

- 공통 통제
  - `rain`, `pm10`, `closed`, `others_closed`, `is_weekend`, `d_holiday`
- 연휴 파생
  - `offdays_left`, `long_break_3p`, `pre_holiday`
- 코로나
  - 기본 `covid_v3`, 비교 시 `covid_v1/v2/v3`
- 이벤트

### 진단 및 평가

- 정상성: `ADF`
- 공선성: `VIF`
- 선행 설명력: `Granger causality`
- 보조 확인: `cointegration`
- 성능: `RMSE`, `MAE`, `R²`, `Adjusted R²`
- 잔차 진단: `ACF`, `PACF`, `Ljung-Box`
- test 구간 시각화

### 발표용 해석 원칙

- 공적분은 장기균형을 보는 검정이므로, 일별 예측 중심인 본 프로젝트의 메인 논거로 쓰지 않는다.
- ADF 결과로 핵심 변수들이 대체로 I(0)임을 확인하고, Granger test는 "어떤 변수가 먼저 설명력을 가지는가"를 보는 보조 도구로 사용한다.
- Granger 1등과 SARIMAX 1등이 달라도 이상하지 않다. 둘은 서로 다른 질문에 답하기 때문이다.
- SARIMAX 결과는 최종적으로 out-of-sample forecast accuracy로 해석한다.

### 현재 문서와 노트북의 역할 분담

- `README.md`
  - 프로젝트 전체 구조와 실행 흐름 정리
- `TEAM_RUN_GUIDE.md`
  - 재현 순서와 실행 체크리스트 정리
- `modeling/timeseries_presentation_outline.md`
  - 발표 슬라이드 문구와 스토리라인 정리
- `modeling/timeseries.ipynb`
  - 발표용 핵심 모델링 흐름
- `modeling/timeseries_2015.ipynb`
  - 2015년 이후 데이터만 사용한 발표용 변형
- `modeling/sarimax.ipynb`
  - 추가 비교와 세부 분석을 포함한 상세 버전

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
