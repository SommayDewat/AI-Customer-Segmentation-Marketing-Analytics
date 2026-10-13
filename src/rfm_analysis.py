"""
RFM analysis utilities for the
AI Customer Segmentation & Marketing Analytics project.
"""

import pandas as pd


def create_rfm(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create customer-level RFM metrics.

    RFM:
        Recency  -> Days since last purchase
        Frequency -> Number of unique orders
        Monetary -> Total customer revenue

    Parameters
    ----------
    df : pd.DataFrame
        Clean transaction-level dataset.

    Returns
    -------
    pd.DataFrame
        Customer-level RFM table.
    """

    required_columns = {
        "Customer ID",
        "Invoice",
        "InvoiceDate",
        "TotalAmount",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    data = df.copy()

    # ---------------------------------------------------------
    # Reference date
    # ---------------------------------------------------------

    reference_date = (
        data["InvoiceDate"].max()
        + pd.Timedelta(days=1)
    )

    # ---------------------------------------------------------
    # Calculate RFM metrics
    # ---------------------------------------------------------

    rfm = (
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
        )
        .reset_index()
    )

    rfm = rfm.rename(
        columns={
            "Customer ID": "CustomerID"
        }
    )

    # ---------------------------------------------------------
    # RFM scoring
    # ---------------------------------------------------------

    # Recency:
    # Lower is better → higher score
    rfm["R_Score"] = pd.qcut(
        rfm["Recency"],
        q=5,
        labels=[5, 4, 3, 2, 1],
        duplicates="drop",
    )

    # Frequency:
    # Higher is better
    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5],
    )

    # Monetary:
    # Higher is better
    rfm["M_Score"] = pd.qcut(
        rfm["Monetary"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5],
    )

    # ---------------------------------------------------------
    # Convert scores to integers
    # ---------------------------------------------------------

    rfm[
        ["R_Score", "F_Score", "M_Score"]
    ] = rfm[
        ["R_Score", "F_Score", "M_Score"]
    ].astype(int)

    # ---------------------------------------------------------
    # Combined RFM score
    # ---------------------------------------------------------

    rfm["RFMScore"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )

    # ---------------------------------------------------------
    # Overall RFM score
    # ---------------------------------------------------------

    rfm["RFMTotalScore"] = (
        rfm["R_Score"]
        + rfm["F_Score"]
        + rfm["M_Score"]
    )

    return rfm