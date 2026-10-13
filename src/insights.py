"""
Business insights and customer segmentation utilities for the
AI Customer Segmentation & Marketing Analytics project.
"""

import pandas as pd


SEGMENT_NAMES = {
    0: "Dormant Low-Value Customers",
    1: "VIP Loyal Customers",
    2: "At-Risk Valuable Customers",
    3: "Engaged Growth Customers",
}


MARKETING_RECOMMENDATIONS = {
    "Dormant Low-Value Customers": (
        "Use low-cost automated reactivation campaigns, "
        "broad promotions, and email reminders. "
        "Avoid high-cost personalized campaigns unless engagement improves."
    ),
    "VIP Loyal Customers": (
        "Prioritize retention through VIP loyalty programs, "
        "exclusive offers, early product access, personalized recommendations, "
        "and premium customer experiences."
    ),
    "At-Risk Valuable Customers": (
        "Prioritize win-back campaigns using personalized offers, "
        "limited-time incentives, product recommendations, "
        "and targeted re-engagement messaging."
    ),
    "Engaged Growth Customers": (
        "Focus on cross-selling, product recommendations, "
        "bundles, loyalty rewards, and increasing purchase frequency "
        "to move customers toward the VIP segment."
    ),
}


def validate_cluster_data(
    df: pd.DataFrame,
    cluster_column: str = "Cluster",
) -> None:
    """
    Validate the customer dataset before generating insights.
    """

    required_columns = {
        "CustomerID",
        "Recency",
        "Frequency",
        "Monetary",
        "TotalQuantity",
        "UniqueProducts",
        "AverageOrderValue",
        "AverageBasketSize",
        "CustomerLifetimeDays",
        "RFMTotalScore",
        cluster_column,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing columns required for cluster insights: "
            f"{sorted(missing_columns)}"
        )


def profile_clusters(
    df: pd.DataFrame,
    cluster_column: str = "Cluster",
) -> pd.DataFrame:
    """
    Generate business profiles for each customer cluster.
    """

    validate_cluster_data(df, cluster_column)

    total_customers = len(df)
    total_revenue = df["Monetary"].sum()

    profile = (
        df.groupby(cluster_column)
        .agg(
            Customers=("CustomerID", "count"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean"),
            AvgQuantity=("TotalQuantity", "mean"),
            AvgProducts=("UniqueProducts", "mean"),
            AvgAOV=("AverageOrderValue", "mean"),
            AvgBasket=("AverageBasketSize", "mean"),
            AvgLifetime=("CustomerLifetimeDays", "mean"),
            AvgRFMScore=("RFMTotalScore", "mean"),
            Revenue=("Monetary", "sum"),
        )
        .reset_index()
    )

    profile["CustomerPercentage"] = (
        profile["Customers"] / total_customers * 100
    )

    profile["RevenuePercentage"] = (
        profile["Revenue"] / total_revenue * 100
    )

    if "Segment" in df.columns:
        segment_lookup = (
            df[[cluster_column, "Segment"]]
            .drop_duplicates()
        )

        if segment_lookup[cluster_column].duplicated().any():
            raise ValueError(
                "Each cluster must map to exactly one segment name."
            )

        segment_map = segment_lookup.set_index(
            cluster_column
        )["Segment"].to_dict()

        profile["Segment"] = profile[cluster_column].map(
            segment_map
        )
    else:
        profile["Segment"] = (
            profile[cluster_column]
            .map(SEGMENT_NAMES)
            .fillna("Unclassified Segment")
        )

    profile["MarketingRecommendation"] = (
        profile["Segment"]
        .map(MARKETING_RECOMMENDATIONS)
        .fillna("Review segment characteristics before targeting.")
    )

    return profile


def assign_segment_names(
    df: pd.DataFrame,
    cluster_column: str = "Cluster",
) -> pd.DataFrame:
    """
    Add business segment names and marketing recommendations
    to the customer-level dataset.
    """

    validate_cluster_data(df, cluster_column)

    result = df.copy()

    result["Segment"] = (
        result[cluster_column]
        .map(SEGMENT_NAMES)
        .fillna("Unclassified Segment")
    )

    result["MarketingRecommendation"] = (
        result["Segment"]
        .map(MARKETING_RECOMMENDATIONS)
        .fillna("Review segment characteristics before targeting.")
    )

    return result


def generate_segment_summary(
    df: pd.DataFrame,
    cluster_column: str = "Cluster",
) -> pd.DataFrame:
    """
    Generate a concise summary for dashboards and reports.
    """

    profile = profile_clusters(
        df,
        cluster_column=cluster_column,
    )

    summary_columns = [
        cluster_column,
        "Segment",
        "Customers",
        "CustomerPercentage",
        "Revenue",
        "RevenuePercentage",
        "AvgRecency",
        "AvgFrequency",
        "AvgMonetary",
        "AvgRFMScore",
    ]

    return profile[summary_columns].copy()



def create_dynamic_segment_map(profile: pd.DataFrame) -> dict:
    """
    Assign business segment names using cluster characteristics.

    Designed for the project's four-segment model.
    """
    required = {
        "Cluster", "AvgRFMScore", "AvgMonetary",
        "AvgRecency",
    }

    missing = required - set(profile.columns)
    if missing:
        raise ValueError(f"Missing profile columns: {sorted(missing)}")

    if len(profile) != 4 or profile["Cluster"].nunique() != 4:
        raise ValueError("Exactly four unique clusters are required.")

    remaining = profile.copy()
    segment_map = {}

    # Highest overall RFM score: VIP
    vip = remaining.loc[remaining["AvgRFMScore"].idxmax()]
    segment_map[int(vip["Cluster"])] = "VIP Loyal Customers"
    remaining = remaining.drop(vip.name)

    # Lowest average monetary value: Dormant
    dormant = remaining.loc[remaining["AvgMonetary"].idxmin()]
    segment_map[int(dormant["Cluster"])] = (
        "Dormant Low-Value Customers"
    )
    remaining = remaining.drop(dormant.name)

    # Most recent purchasing activity: Growth
    growth = remaining.loc[remaining["AvgRecency"].idxmin()]
    segment_map[int(growth["Cluster"])] = (
        "Engaged Growth Customers"
    )
    remaining = remaining.drop(growth.name)

    # Remaining cluster: Potential win-back opportunity
    at_risk = remaining.iloc[0]
    segment_map[int(at_risk["Cluster"])] = (
        "At-Risk Valuable Customers"
    )

    return segment_map


def assign_dynamic_segments(
    df: pd.DataFrame,
    segment_map: dict,
) -> pd.DataFrame:
    """
    Assign dynamically determined segment names
    and their marketing recommendations.
    """
    result = df.copy()

    if not set(result["Cluster"].unique()).issubset(segment_map):
        raise ValueError("Some clusters have no segment mapping.")

    result["Segment"] = result["Cluster"].map(segment_map)

    result["MarketingRecommendation"] = (
        result["Segment"].map(MARKETING_RECOMMENDATIONS)
    )

    return result
