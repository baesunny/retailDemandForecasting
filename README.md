# Retail Demand Forecasting

유통 POS 데이터를 활용해 면류·라면류 월별 판매 수요를 예측한 시계열 분석 프로젝트입니다. 2024년 2학기 **시계열 유통데이터활용공모전** 과제로 수행했으며, 제조업체 단위 예측의 한계를 확인한 뒤 클러스터 기반 하이브리드 모델로 전환하여 2023년 9~12월 테스트 구간에서 **월별 평균 MAPE 4.65%**를 달성했습니다.

## 프로젝트 목적

소매 유통 데이터는 SKU·제조업체 단위로 관측치가 희소해 단일 시계열 모델만으로는 안정적인 수요 예측이 어렵습니다. 본 프로젝트는 다음 질문에 답하는 것을 목표로 했습니다.

1. 제조업체 단위 예측이 왜 실패하는가?
2. 판매량 기반 클러스터링으로 예측 단위를 재구성하면 성능이 개선되는가?
3. 날씨·경제지수 등 외생변수와 SARIMAX, Prophet, 트리 모델, 앙상블 중 어떤 조합이 가장 효과적인가?

최종적으로 29개 제조업체를 4개 클러스터로 묶고, 클러스터별 최적 모델을 선정한 뒤 월별 총 수요로 집계하는 예측 파이프라인을 완성했습니다.

## 배경 및 문제 정의

| 항목 | 내용 |
| --- | --- |
| 분석 대상 | 면류·라면류 유통 판매 데이터 |
| 데이터 기간 | 2021-01 ~ 2023-12 (학습·검증), 2024년 미완성 데이터 제외 |
| 예측 대상 | 월별 `판매수량` |
| 외생변수 | 한파, 폭염, 호우, 생활물가지수, 음식료품 계절조정지수 |
| 검증 구간 | 2023-09 ~ 2023-12 (4개월 홀드아웃) |

초기 실험에서 제조업체 단위 SARIMAX·Prophet은 MAPE 105% 수준으로 예측이 불가능함을 확인했습니다. 원인은 제조업체·월 조합당 관측치가 적어 시계열 패턴을 학습하기 어렵기 때문입니다.

## 접근 방법

```text
final.csv
  → 전처리 (상품 메타 제거, 2024년 데이터 제외)
  → 29개 제조업체 greedy volume-balanced 4-cluster
  → 월×클러스터 집계 + 2023-05 cluster 4 이상치 보정
  → 클러스터별 모델 실험 (SARIMAX / Prophet / RF / XGB / LGBM / 앙상블)
  → 클러스터 예측값 합산 → 월별 총 수요 예측
```

공통 전처리 로직은 `src/preprocessing.py`에, 평가 함수는 `src/evaluation.py`에 분리했습니다.

### 실험 흐름

| 노트북 | 내용 | 대표 결과 |
| --- | --- | ---: |
| `01_baseline_sarimax` | 제조업체·중분류 SARIMAX 탐색 | 제조업체 단위 실패 |
| `02_prophet_manufacturer` | 제조업체 단위 Prophet | MAPE 105.53% |
| `03_prophet_cluster` | 클러스터 + Prophet | MAPE 38.61% |
| `04_sarimax_hyperparameter_tuning` | SARIMAX Grid Search / Optuna | MAPE 31~73% |
| `05_tree_models_no_exog` | 외생변수 제외 트리 모델 | RF MAPE 3.36% |
| `06_full_experiment_matrix` | 특성·모델·튜닝 전수 비교 | 최적 조합 선정 근거 |
| `07_final_aggregation` | 클러스터별 최종 모델 + 집계 | 월별 MAPE 4.65% |

## 최종 결과

### 클러스터별 최적 모델 (`07_final_aggregation`)

| Cluster | 최종 모델 | Test MAPE |
| ---: | --- | ---: |
| 1 | 날씨 + SARIMAX 25% + LightGBM 75% | 6.80% |
| 2 | 날씨 + SARIMAX + GridSearch | 5.74% |
| 3 | 날씨 + SARIMAX + GridSearch | 4.62% |
| 4 | 날씨 + 경제지수 + SARIMAX + GridSearch | 10.80% |

### 월별 총 수요 예측 (2023-09 ~ 2023-12)

| 월 | 실제 | 예측 | MAPE |
| --- | ---: | ---: | ---: |
| 2023-09 | 6,598 | 6,573 | 0.38% |
| 2023-10 | 7,522 | 6,694 | 11.01% |
| 2023-11 | 6,795 | 6,348 | 6.58% |
| 2023-12 | 6,848 | 6,892 | 0.64% |

- 4개월 절대 오차 합: 1,344건
- 월별 평균 MAPE: **4.65%**

## 사용 기술

- Python, Jupyter Notebook
- pandas, numpy, matplotlib
- statsmodels (SARIMAX), Prophet
- scikit-learn (GridSearchCV, RandomForest)
- XGBoost, LightGBM, Optuna

## 디렉토리 구조

```text
retailDemandForecasting/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md                         # final.csv 배치 안내
├── docs/
│   └── presentation/
│       └── [24-2학기]시계열_유통데이터활용공모전.pdf
├── notebooks/
│   ├── 01_baseline_sarimax.ipynb         # 초기 SARIMAX 실험
│   ├── 02_prophet_manufacturer.ipynb     # 제조업체 단위 Prophet
│   ├── 03_prophet_cluster.ipynb          # 클러스터 + Prophet
│   ├── 04_sarimax_hyperparameter_tuning.ipynb
│   ├── 05_tree_models_no_exog.ipynb      # 트리 모델 실험
│   ├── 06_full_experiment_matrix.ipynb   # 전체 조합 실험
│   └── 07_final_aggregation.ipynb        # 최종 모델 및 집계
└── src/
    ├── __init__.py
    ├── config.py                         # 경로, 기간, 외생변수 상수
    ├── preprocessing.py                  # 로드, 클러스터링, 집계, 이상치 보정
    └── evaluation.py                     # MAPE 평가 유틸
```

## 발표자료

| 파일 | 설명 |
| --- | --- |
| `docs/presentation/[24-2학기]시계열_유통데이터활용공모전.pdf` | 2024년 2학기 시계열 유통데이터활용공모전 제출 보고서 |

## 실행 방법

```bash
pip install -r requirements.txt
```

1. 공모전에서 제공된 `final.csv`를 `data/final.csv`에 배치합니다.
2. Jupyter Notebook을 실행합니다.
3. 실험 흐름은 `notebooks/01` → `07` 순서로 확인할 수 있습니다.
4. 최종 결과와 클러스터별 모델 선정은 `07_final_aggregation.ipynb`에서 확인합니다.

## 핵심 코드

전처리 파이프라인은 아래 한 줄로 재현할 수 있습니다.

```python
from src.preprocessing import build_modeling_dataset

assignments, monthly = build_modeling_dataset()
```

클러스터링, 월별 집계, 2023-05 이상치 보정까지 `src/preprocessing.py`에 포함되어 있습니다.
