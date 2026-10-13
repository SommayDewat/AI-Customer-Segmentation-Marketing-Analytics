"""
Machine learning preparation utilities for the
AI Customer Segmentation & Marketing Analytics project.
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# ============================================================
# Features used for clustering
# ============================================================

CLUSTERING_FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "TotalQuantity",
    "UniqueProducts",
    "AverageOrderValue",
    "AverageBasketSize",
    "CustomerLifetimeDays",
]


# ============================================================
# Features requiring log transformation
# ============================================================

LOG_TRANSFORM_FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "TotalQuantity",
    "UniqueProducts",
    "AverageOrderValue",
    "AverageBasketSize",
]


# ============================================================
# Features kept without log transformation
# ============================================================

NON_LOG_FEATURES = [
    "CustomerLifetimeDays",
]


# ============================================================
# Validate clustering features
# ============================================================

def validate_clustering_features(
    df: pd.DataFrame,
    features: list[str] = CLUSTERING_FEATURES,
) -> None:
    """
    Validate that all clustering features exist
    in the customer dataset.
    """

    missing_features = set(features) - set(df.columns)

    if missing_features:
        raise ValueError(
            "Missing clustering features: "
            f"{sorted(missing_features)}"
        )


# ============================================================
# Extract clustering features
# ============================================================

def get_clustering_features(
    df: pd.DataFrame,
    features: list[str] = CLUSTERING_FEATURES,
) -> pd.DataFrame:
    """
    Extract the features intended for clustering.
    """

    validate_clustering_features(
        df,
        features,
    )

    clustering_data = df[features].copy()

    return clustering_data


# ============================================================
# Analyze feature skewness
# ============================================================

def analyze_feature_skewness(
    df: pd.DataFrame,
    features: list[str] = CLUSTERING_FEATURES,
) -> pd.DataFrame:
    """
    Calculate skewness statistics for clustering features.
    """

    validate_clustering_features(
        df,
        features,
    )

    result = pd.DataFrame({
        "Feature": features,
        "Skewness": [
            df[feature].skew()
            for feature in features
        ],
    })

    result["AbsoluteSkewness"] = (
        result["Skewness"].abs()
    )

    result["HighlySkewed"] = (
        result["AbsoluteSkewness"] > 1
    )

    return (
        result
        .sort_values(
            "AbsoluteSkewness",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# Log transformation
# ============================================================

def log_transform_features(
    df: pd.DataFrame,
    log_features: list[str] = LOG_TRANSFORM_FEATURES,
    non_log_features: list[str] = NON_LOG_FEATURES,
) -> pd.DataFrame:
    """
    Apply log1p transformation to selected skewed features
    while retaining non-transformed features.
    """

    all_features = (
        log_features + non_log_features
    )

    missing_features = (
        set(all_features) - set(df.columns)
    )

    if missing_features:
        raise ValueError(
            "Missing features: "
            f"{sorted(missing_features)}"
        )

    data = df[all_features].copy()

    if data.isna().any().any():
        raise ValueError(
            "Missing values detected before "
            "log transformation."
        )

    for feature in log_features:

        if (data[feature] < 0).any():
            raise ValueError(
                f"Negative values detected in "
                f"{feature}."
            )

        data[feature] = np.log1p(
            data[feature]
        )

    return data


# ============================================================
# Feature scaling
# ============================================================

def scale_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize numerical features using StandardScaler.

    Returns
    -------
    scaled_data : pd.DataFrame
        Standardized feature matrix.

    scaler : StandardScaler
        Fitted scaler.
    """

    if df.isna().any().any():
        raise ValueError(
            "Missing values detected before scaling."
        )

    scaler = StandardScaler()

    scaled_values = scaler.fit_transform(df)

    scaled_data = pd.DataFrame(
        scaled_values,
        columns=df.columns,
        index=df.index,
    )

    return scaled_data, scaler