"""
Data loading utilities for the
AI Customer Segmentation & Marketing Analytics project.
"""

from pathlib import Path
from typing import Optional

import pandas as pd


# Expected schema for the Online Retail dataset
REQUIRED_COLUMNS = {
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
}


def load_data(
    file_path: str,
    sheet_name: str = "Year 2010-2011",
) -> pd.DataFrame:
    """
    Load the Online Retail dataset from an Excel file.

    Parameters
    ----------
    file_path : str
        Path to the Excel dataset.
    sheet_name : str, default="Year 2010-2011"
        Name of the worksheet to load.

    Returns
    -------
    pd.DataFrame
        Loaded and validated dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.
    ValueError
        If the file extension is unsupported or required columns
        are missing.
    """

    path = Path(file_path)

    # ---------------------------------------------------------
    # 1. Validate file existence
    # ---------------------------------------------------------
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    # ---------------------------------------------------------
    # 2. Validate file type
    # ---------------------------------------------------------
    if path.suffix.lower() not in {".xlsx", ".xls"}:
        raise ValueError(
            "Unsupported file format. "
            "Please provide an Excel (.xlsx or .xls) file."
        )

    # ---------------------------------------------------------
    # 3. Load Excel worksheet
    # ---------------------------------------------------------
    try:
        df = pd.read_excel(
            path,
            sheet_name=sheet_name,
        )
    except ValueError as exc:
        raise ValueError(
            f"Unable to load worksheet '{sheet_name}'. "
            "Please verify the sheet name."
        ) from exc

    # ---------------------------------------------------------
    # 4. Validate required columns
    # ---------------------------------------------------------
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # ---------------------------------------------------------
    # 5. Standardize column names
    # ---------------------------------------------------------
    df.columns = (
        df.columns
        .str.strip()
    )

    # ---------------------------------------------------------
    # 6. Convert InvoiceDate to datetime
    # ---------------------------------------------------------
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce",
    )

    # ---------------------------------------------------------
    # 7. Validate critical datetime values
    # ---------------------------------------------------------
    invalid_dates = df["InvoiceDate"].isna().sum()

    if invalid_dates > 0:
        raise ValueError(
            f"Found {invalid_dates:,} invalid InvoiceDate values."
        )

    return df


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Generate basic metadata for a loaded dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Returns
    -------
    dict
        Dataset summary statistics.
    """

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
        "unique_invoices": int(df["Invoice"].nunique()),
        "unique_customers": int(df["Customer ID"].nunique()),
        "unique_products": int(df["StockCode"].nunique()),
        "unique_countries": int(df["Country"].nunique()),
        "date_min": df["InvoiceDate"].min(),
        "date_max": df["InvoiceDate"].max(),
    }