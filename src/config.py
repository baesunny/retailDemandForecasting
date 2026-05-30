from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "final.csv"

N_CLUSTERS = 4
TRAIN_START = "2021-01-01"
TRAIN_END = "2023-08-01"
TEST_START = "2023-09-01"
TEST_END = "2023-12-01"

WEATHER_EXOG = ["한파", "폭염", "호우"]
ECONOMIC_EXOG = ["생활물가지수", "음식료품_계절조정지수"]
ALL_EXOG = WEATHER_EXOG + ECONOMIC_EXOG

OUTLIER_MONTH = "2023-05-01"
OUTLIER_CLUSTER = 4
