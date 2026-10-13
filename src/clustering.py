"""
Customer clustering utilities for the
AI Customer Segmentation & Marketing Analytics project.
"""

from pathlib import Path

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
)


def evaluate_kmeans(
    scaled_data: pd.DataFrame,
    k_range: range = range(2, 9),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Evaluate K-Means clustering across multiple
    values of K.

    Metrics:
    - Inertia
    - Silhouette Score
    - Davies-Bouldin Index
    """

    if scaled_data.empty:
        raise ValueError(
            "Scaled dataset is empty."
        )

    if scaled_data.isna().any().any():
        raise ValueError(
            "Missing values detected in scaled data."
        )

    results = []

    for k in k_range:

        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=20,
        )

        labels = model.fit_predict(
            scaled_data
        )

        results.append({
            "K": k,
            "Inertia": model.inertia_,
            "SilhouetteScore": silhouette_score(
                scaled_data,
                labels,
            ),
            "DaviesBouldinIndex": (
                davies_bouldin_score(
                    scaled_data,
                    labels,
                )
            ),
        })

    return pd.DataFrame(results)


def fit_kmeans(
    scaled_data: pd.DataFrame,
    n_clusters: int,
    random_state: int = 42,
) -> tuple[KMeans, pd.Series]:
    """
    Fit the final K-Means model.

    Returns
    -------
    model : KMeans
        Fitted K-Means model.

    labels : pd.Series
        Cluster assignments.
    """

    if n_clusters < 2:
        raise ValueError(
            "n_clusters must be at least 2."
        )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20,
    )

    labels = model.fit_predict(
        scaled_data
    )

    labels = pd.Series(
        labels,
        index=scaled_data.index,
        name="Cluster",
    )

    return model, labels


def save_cluster_evaluation(
    evaluation_results: pd.DataFrame,
    output_path: str = "reports/cluster_evaluation.csv",
) -> None:
    """
    Save K-Means evaluation results to CSV.
    """

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_results.to_csv(
        path,
        index=False,
    )