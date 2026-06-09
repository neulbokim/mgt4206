# 팀원 실행 가이드

이 문서는 팀원이 로컬에서 같은 결과를 재현하기 위한 실행 절차만 정리한 문서입니다.

## 1. 작업 위치

- 프로젝트 루트:
  - `code/`

모든 명령은 이 디렉터리에서 실행한다.

## 2. 환경 준비

```bash
# 가상환경 생성
python3 -m venv .venv

# 가상환경 활성화
source .venv/bin/activate # MacOS인 경우
.\.venv\Scripts\activate # Windows인 경우

# 패키지 설치
pip install -r requirements.txt
```

## 3. `.env` 확인

프로젝트 루트 `code/.env`에 아래 키가 있어야 한다.

```env
SERVICE_KEY=...
```

이 키는 `data/holiday_scraper.py`가 공휴일 raw 데이터를 받을 때 사용한다.

## 4. 전처리 실행 순서

아래 순서대로 실행한다.

```bash
python3 data/holiday_scraper.py
python3 preprocessing/visitors_gbg.py
python3 preprocessing/asos_daily.py
python3 preprocessing/ydst_daily.py
python3 preprocessing/covid_dummy.py
python3 preprocessing/ktci_calculator.py
```

## 5. 전처리 후 확인할 파일

다음 파일이 생성되어야 한다.

- `data/raw/holiday_calendar_2009_2026.csv`
- `data/preprocessed/visitors_gbg_daily.csv`
- `data/preprocessed/asos_seoul_daily.csv`
- `data/preprocessed/ydst_seoul_daily.csv`
- `data/preprocessed/covid_dummy_daily.csv`
- `data/preprocessed/ktci.csv`

## 6. EDA 실행

열어야 할 노트북:

- `data/eda.ipynb`

실행 순서:

- 위에서부터 순서대로 전체 실행

생성 파일:

- `data/preprocessed/eda_gbg_master.csv`

EDA에서 먼저 확인할 것:

- 방문객 시계열 그래프
- 월별 / 요일별 패턴
- 코로나 구간 비교
- 결측률 표

## 7. 모델링 실행

필요하면 먼저 보조 분석 스크립트를 실행한다.

```bash
python3 modeling/granger_cointegration_tests.py
python3 modeling/no_event_granger_cointegration_tests.py
python3 modeling/no_event_sarimax_performance.py
```

이 스크립트들은 `outputs/` 아래의 `csv`와 `figures`를 생성한다.

열어야 할 노트북:

- `modeling/timeseries.ipynb`
- `modeling/timeseries_2015.ipynb` (2015년 이후 구간만 볼 때 사용)
- `modeling/sarimax.ipynb`

실행 순서:

- `eda_gbg_master.csv`가 최신인지 확인
- `modeling/timeseries.ipynb`를 먼저 실행
- 2015년 이후 버전이 필요하면 `modeling/timeseries_2015.ipynb`를 대신 실행
- 필요하면 `modeling/sarimax.ipynb`를 이어서 실행

주요 확인 항목:

- `ADF`
- `VIF`
- `Granger causality`
- `cointegration`은 백업 슬라이드용 보조 확인으로만 사용
- 성능표 `RMSE`, `MAE`, `R²`, `Adjusted R²`
- test 구간 실제값 vs 예측값 그래프
- 환절기 구간 확대 그래프
- 잔차 진단

발표 메인 해석:

- 공적분은 메인 슬라이드에서 제외한다.
- 핵심 변수들이 대체로 I(0)이므로, 장기균형보다 단기 예측 타당성이 더 중요하다.
- Granger test는 선행 설명력의 보조 증거로 쓰고, 최종 순위는 SARIMAX의 out-of-sample 성능으로 판단한다.
- `Granger` 1등과 `SARIMAX` 1등이 달라도 이상하지 않다. 둘은 서로 다른 질문에 답한다.

## 8. 현재 기본 해석

- 분석 대상은 `경복궁` 하나다.
- 종속변수는 `visitors`다.
- 공기질 변수는 `pm25`가 아니라 `pm10`이다.
- 방문객 누락 53일 구간은 전처리에서 `0`으로 채워져 있다.
- 공휴일 변수 `d_holiday`는 `data/raw/holiday_calendar_2009_2026.csv`를 기준으로 만든다.

## 9. 자주 생길 수 있는 문제

- **공휴일 파일이 없을 때**
  - `python3 data/holiday_scraper.py`를 먼저 실행

- **노트북에서 입력 파일이 없다고 나올 때**
  - 전처리 스크립트를 순서대로 다시 실행

- **EDA 시계열 그래프가 비정상일 때**
  - `data/preprocessed/eda_gbg_master.csv`를 다시 생성
  - `data/eda.ipynb`를 위에서부터 다시 실행

- **SARIMAX에서 결측 관련 에러가 날 때**
  - `data/eda.ipynb`를 다시 실행해 master를 갱신

## 10. 최소 재실행 루트

코드 수정 없이 결과만 다시 보고 싶을 때:

```bash
python3 data/holiday_scraper.py
python3 preprocessing/visitors_gbg.py
python3 preprocessing/asos_daily.py
python3 preprocessing/ydst_daily.py
python3 preprocessing/covid_dummy.py
python3 preprocessing/ktci_calculator.py
```

그 다음:

- `data/eda.ipynb`
- `modeling/sarimax.ipynb`

순서로 다시 실행하면 된다.
