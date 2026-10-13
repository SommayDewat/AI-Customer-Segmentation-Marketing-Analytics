"""
Customer-level feature engineering for the
AI Customer Segmentation & Marketing Analytics project.
"""

import pandas as pd


def create_customer_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create customer-level behavioral features.

    Parameters
    ----------
    df : pd.DataFrame
        Clean transaction-level dataset.

    Returns
    -------
    pd.DataFrame
        Customer-level feature table.
    """

    data = df.copy()

    # ---------------------------------------------------------
    # Validate required columns
    # ---------------------------------------------------------

    required_columns = {
        "Customer ID",
        "Invoice",
        "InvoiceDate",
        "Quantity",
        "StockCode",
        "TotalAmount",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # ---------------------------------------------------------
    # Reference date
    # ---------------------------------------------------------

    reference_date = (
        data["InvoiceDate"].max()
        + pd.Timedelta(days=1)
    )

    # ---------------------------------------------------------
    # Customer-level aggregation
    # ---------------------------------------------------------

    customer_features = (
        data.groupby("Customer ID")
        .agg(
            Recency=(
                "InvoiceDate",
                lambda x: (
                    reference_date - x.max()
                ).days,
            ),

            Frequency=(
                "Invoice",
                "nunique",
            ),

            Monetary=(
                "TotalAmount",
                "sum",
            ),

            TotalQuantity=(
                "Quantity",
                "sum",
            ),

            UniqueProducts=(
                "StockCode",
                "nunique",
            ),

            FirstPurchaseDate=(
                "InvoiceDate",
                "min",
            ),

            LastPurchaseDate=(
                "InvoiceDate",
                "max",
            ),
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # Average Order Value
    # ---------------------------------------------------------

    customer_features["AverageOrderValue"] = (
        customer_features["Monetary"]
        / customer_features["Frequency"]
    )

    # ---------------------------------------------------------
    # Average Basket Size
    # ---------------------------------------------------------

    customer_features["AverageBasketSize"] = (
        customer_features["TotalQuantity"]
        / customer_features["Frequency"]
    )

    # ---------------------------------------------------------
    # Customer lifetime
    # ---------------------------------------------------------

    customer_features["CustomerLifetimeDays"] = (
        (
            customer_features["LastPurchaseDate"]
            - customer_features["FirstPurchaseDate"]
        )
        .dt.days
        .clip(lower=1)
    )

    # ---------------------------------------------------------
    # Purchase Frequency
    # Orders per 30 days
    # ---------------------------------------------------------

    customer_features["PurchaseFrequency"] = (
        customer_features["Frequency"]
        / customer_features["CustomerLifetimeDays"]
        * 30
    )
    customer_features.loc[
        customer_features["Frequency"] <= 1,
        "PurchaseFrequency"
    ] = pd.NA

    # ---------------------------------------------------------
    # Average Purchase Gap
    # ---------------------------------------------------------

    customer_features["AveragePurchaseGap"] = (
        customer_features["CustomerLifetimeDays"]
        / customer_features["Frequency"].clip(lower=1)
    )
    customer_features.loc[
        customer_features["Frequency"] <= 1,
        "AveragePurchaseGap"
    ] = pd.NA

    # ---------------------------------------------------------
    # Revenue Contribution
    # ---------------------------------------------------------

    total_revenue = customer_features["Monetary"].sum()

    customer_features["RevenueContribution"] = (
        customer_features["Monetary"]
        / total_revenue
        * 100
    )

    # ---------------------------------------------------------
    # Round numerical features
    # ---------------------------------------------------------

    numerical_columns = [
        "Monetary",
        "AverageOrderValue",
        "AverageBasketSize",
        "PurchaseFrequency",
        "AveragePurchaseGap",
        "RevenueContribution",
    ]

    customer_features[numerical_columns] = (
        customer_features[numerical_columns]
        .round(2)
    )

    return customer_features

def merge_customer_features(
    customer_features: pd.DataFrame,
    rfm: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge behavioral customer features with RFM metrics.

    Parameters
    ----------
    customer_features : pd.DataFrame
        Customer-level behavioral features.

    rfm : pd.DataFrame
        Customer-level RFM table.

    Returns
    -------
    pd.DataFrame
        Integrated customer intelligence dataset.
    """

    required_customer_columns = {
        "Customer ID",
    }

    required_rfm_columns = {
        "CustomerID",
        "Recency",
        "Frequency",
        "Monetary",
        "R_Score",
        "F_Score",
        "M_Score",
        "RFMScore",
        "RFMTotalScore",
    }

    missing_customer_columns = (
        required_customer_columns
        - set(customer_features.columns)
    )

    missing_rfm_columns = (
        required_rfm_columns
        - set(rfm.columns)
    )

    if missing_customer_columns:
        raise ValueError(
            "customer_features is missing columns: "
            f"{sorted(missing_customer_columns)}"
        )

    if missing_rfm_columns:
        raise ValueError(
            "rfm is missing columns: "
            f"{sorted(missing_rfm_columns)}"
        )

    # Rename customer identifier for consistency
    features = customer_features.rename(
        columns={
            "Customer ID": "CustomerID"
        }
    ).copy()

    # Avoid duplicate RFM metrics from the feature table
    rfm_columns = [
        "CustomerID",
        "R_Score",
        "F_Score",
        "M_Score",
        "RFMScore",
        "RFMTotalScore",
    ]

    rfm_data = rfm[rfm_columns].copy()

    # Merge
    customer_intelligence = features.merge(
        rfm_data,
        on="CustomerID",
        how="left",
        validate="one_to_one",
    )

    return customer_intelligence