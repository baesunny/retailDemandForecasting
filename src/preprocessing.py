from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ALL_EXOG,
    DATA_PATH,
    N_CLUSTERS,
    OUTLIER_CLUSTER,
    OUTLIER_MONTH,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
)


def load_raw_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the retail transaction dataset."""
    return pd.read_csv(path, encoding="utf-8-sig")


def prepare_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop product metadata and remove incomplete 2024 rows."""
    prepared = df.copy()
    prepared.drop(columns=["상품명", "대분류", "중분류"], inplace=True, errors="ignore")
    prepared["판매월"] = pd.to_datetime(prepared["판매월"], format="%Y-%m")
    return prepared[prepared["판매월"].dt.year != 2024].copy()


def cluster_manufacturers(
    df: pd.DataFrame,
    n_clusters: int = N_CLUSTERS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Greedy volume-balanced clustering of manufacturers.

    Returns manufacturer-level assignments and the dataframe with cluster labels.
    """
    manufacturer_sales = (
        df.groupby("제조업체")["판매수량"]
        .sum()
        .reset_index()
        .sort_values(by="판매수량", ascending=False)
        .reset_index(drop=True)
    )

    cluster_sales = [0] * n_clusters
    cluster_labels: list[int] = []

    for _, row in manufacturer_sales.iterrows():
        cluster_idx = int(np.argmin(cluster_sales))
        cluster_sales[cluster_idx] += row["판매수량"]
        cluster_labels.append(cluster_idx + 1)

    manufacturer_sales["군집"] = cluster_labels

    labeled = df.copy()
    cluster_map = dict(
        zip(manufacturer_sales["제조업체"], manufacturer_sales["군집"])
    )
    labeled["cluster"] = labeled["제조업체"].map(cluster_map)
    return manufacturer_sales, labeled


def aggregate_cluster_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sales and exogenous variables at month-cluster level."""
    aggregated = df.groupby(["판매월", "cluster"], as_index=False).agg(
        {
            "판매수량": "sum",
            "한파": "first",
            "폭염": "first",
            "호우": "first",
            "생활물가지수": "first",
            "음식료품_계절조정지수": "first",
        }
    )
    return fix_cluster_outlier(aggregated)


def fix_cluster_outlier(
    df: pd.DataFrame,
    month: str = OUTLIER_MONTH,
    cluster: int = OUTLIER_CLUSTER,
) -> pd.DataFrame:
    """Replace the known 2023-05 cluster outlier with the pre-test mean."""
    fixed = df.copy()
    baseline = fixed[
        (fixed["판매월"] < pd.Timestamp(TEST_START)) & (fixed["cluster"] == cluster)
    ]["판매수량"].mean()
    mask = (fixed["판매월"] == pd.Timestamp(month)) & (fixed["cluster"] == cluster)
    fixed.loc[mask, "판매수량"] = baseline
    return fixed


def split_cluster_data(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Split monthly cluster data into individual cluster frames."""
    return {
        cluster_id: df[df["cluster"] == cluster_id].reset_index(drop=True)
        for cluster_id in sorted(df["cluster"].unique())
    }


def split_train_test(
    cluster_df: pd.DataFrame,
    exog_vars: list[str] | None = None,
) -> tuple[pd.Series, pd.Series, pd.DataFrame | None, pd.DataFrame | None]:
    """Split target and exogenous variables into train/test windows."""
    exog_vars = exog_vars or ALL_EXOG
    train_mask = (cluster_df["판매월"] >= TRAIN_START) & (
        cluster_df["판매월"] <= TRAIN_END
    )
    test_mask = (cluster_df["판매월"] >= TEST_START) & (
        cluster_df["판매월"] <= TEST_END
    )

    train_y = cluster_df.loc[train_mask, "판매수량"]
    test_y = cluster_df.loc[test_mask, "판매수량"]
    train_x = cluster_df.loc[train_mask, exog_vars]
    test_x = cluster_df.loc[test_mask, exog_vars]
    return train_y, test_y, train_x, test_x


def build_modeling_dataset(path: Path | str = DATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end preprocessing pipeline used across experiment notebooks."""
    raw = load_raw_data(path)
    base = prepare_base_dataframe(raw)
    assignments, labeled = cluster_manufacturers(base)
    monthly = aggregate_cluster_monthly(labeled)
    return assignments, monthly
