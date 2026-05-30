import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error


def mape_percent(actual, predicted) -> float:
    """Return MAPE as a percentage."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(mean_absolute_percentage_error(actual, predicted) * 100)


def monthly_mape_table(
    months: pd.Series,
    actual: pd.Series,
    predicted: pd.Series,
) -> pd.DataFrame:
    """Build a month-level comparison table with MAPE."""
    result = pd.DataFrame(
        {
            "판매월": months.values,
            "판매수량": actual.values,
            "예측판매수량": predicted.values,
        }
    )
    result["차이"] = (result["판매수량"] - result["예측판매수량"]).abs()
    result["mape"] = result.apply(
        lambda row: abs((row["판매수량"] - row["예측판매수량"]) / row["판매수량"]) * 100,
        axis=1,
    )
    return result
