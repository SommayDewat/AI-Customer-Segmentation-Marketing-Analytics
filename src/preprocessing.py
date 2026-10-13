
"""
Data preprocessing and quality utilities for the
AI Customer Segmentation & Marketing Analytics project.
"""

from pathlib import Path

import pandas as pd


def generate_data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a column-level data quality report."""
    return pd.DataFrame({
        "column": df.columns,
        "data_type": df.dtypes.astype(str).values,
        "missing_count": df.isna().sum().values,
        "missing_percentage": (df.isna().mean().values * 100).round(2),
        "unique_values": [
            df[column].nunique(dropna=True)
            for column in df.columns
        ],
    })


def preprocess_data(
    df: pd.DataFrame,
    audit_path: str | None = "reports/cancellation_audit.csv",
) -> tuple[pd.DataFrame, dict]:
    """
    Preserve source rows and conservatively reconcile cancellations.

    A cancellation is applied only when the eligible earlier sale
    lines have enough combined remaining quantity and the allocation
    is unambiguous under the matching rules.

    Exact duplicate rows are retained as separate source lines.
    """
    data = df.copy()
    data.columns = data.columns.str.strip()

    required = {
        "Invoice", "StockCode", "Quantity", "InvoiceDate",
        "Price", "Customer ID",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    initial_rows = len(data)
    duplicate_rows = int(data.duplicated().sum())
    missing_customer_ids = int(data["Customer ID"].isna().sum())

    # Preserve a unique identifier for every raw source row.
    data["_SourceRowID"] = range(len(data))
    data["InvoiceDate"] = pd.to_datetime(
        data["InvoiceDate"], errors="coerce"
    )
    data["InvoiceText"] = data["Invoice"].astype(str).str.strip()
    data["Customer ID"] = pd.to_numeric(
        data["Customer ID"], errors="coerce"
    )

    is_cancel = data["InvoiceText"].str.upper().str.startswith("C")

    sales = data.loc[
        ~is_cancel
        & data["Quantity"].gt(0)
        & data["Price"].gt(0)
        & data["InvoiceDate"].notna()
        & data["Customer ID"].notna()
    ].copy()

    cancellations = data.loc[
        is_cancel & data["Quantity"].lt(0)
    ].copy()

    sales = sales.sort_values(
        ["InvoiceDate", "_SourceRowID"], kind="stable"
    ).reset_index(drop=True)

    sales["SaleLineID"] = sales["_SourceRowID"]
    sales["RemainingQuantity"] = sales["Quantity"].astype(int)
    sales["OriginalInvoice"] = sales["InvoiceText"]

    cancellations = cancellations.sort_values(
        ["InvoiceDate", "_SourceRowID"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    audit = []
    cancelled_quantity = 0
    matched_cancellation_rows = 0

    for _, cancel in cancellations.iterrows():
        requested_qty = int(-cancel["Quantity"])
        customer_id = cancel["Customer ID"]
        cancel_date = cancel["InvoiceDate"]

        record = {
            "SourceRowID": int(cancel["_SourceRowID"]),
            "CancellationInvoice": cancel["InvoiceText"],
            "CancellationDate": cancel["InvoiceDate"],
            "CustomerID": customer_id,
            "StockCode": cancel["StockCode"],
            "CancelledQuantityRequested": requested_qty,
            "Price": cancel["Price"],
            "Status": "",
            "MatchedSaleInvoices": "",
            "MatchedSaleLineIDs": "",
            "QuantityApplied": 0,
            "Notes": "",
        }

        if (
            pd.isna(customer_id)
            or pd.isna(cancel_date)
            or pd.isna(cancel["StockCode"])
            or pd.isna(cancel["Price"])
        ):
            record["Status"] = "unmatched"
            record["Notes"] = (
                "Missing customer, date, product, or price"
            )
            audit.append(record)
            continue

        prior_mask = (
            sales["Customer ID"].eq(customer_id)
            & sales["StockCode"].eq(cancel["StockCode"])
            & sales["Price"].eq(cancel["Price"])
            & sales["InvoiceDate"].lt(cancel_date)
        )
        prior = sales.loc[prior_mask]
        eligible = prior.loc[prior["RemainingQuantity"].gt(0)]

        available_qty = int(eligible["RemainingQuantity"].sum())

        if available_qty < requested_qty:
            record["Status"] = "unmatched"
            record["Notes"] = (
                f"Insufficient earlier quantity: {available_qty} "
                f"available, {requested_qty} requested"
                if not prior.empty
                else "No earlier sale matches customer/product/price"
            )
            audit.append(record)
            continue

        # If several sale lines qualify, use a deterministic
        # earliest-sale-first allocation. Record every consumed line.
        # This is an allocation rule, not proof of original linkage.
        remaining_to_cancel = requested_qty
        allocations = []

        for sale_idx in eligible.index:
            if remaining_to_cancel == 0:
                break

            available = int(sales.at[sale_idx, "RemainingQuantity"])
            take = min(available, remaining_to_cancel)

            if take > 0:
                allocations.append({
                    "sale_idx": sale_idx,
                    "quantity": take,
                    "invoice": sales.at[sale_idx, "OriginalInvoice"],
                    "line_id": int(sales.at[sale_idx, "SaleLineID"]),
                })
                remaining_to_cancel -= take

        if remaining_to_cancel != 0:
            record["Status"] = "unmatched"
            record["Notes"] = (
                "Allocation failed validation; no quantity applied"
            )
            audit.append(record)
            continue

        # Apply only after the full allocation has been planned.
        for allocation in allocations:
            idx = allocation["sale_idx"]
            sales.at[idx, "RemainingQuantity"] -= allocation["quantity"]

        record["Status"] = "matched"
        record["MatchedSaleInvoices"] = ";".join(
            str(x["invoice"]) for x in allocations
        )
        record["MatchedSaleLineIDs"] = ";".join(
            str(x["line_id"]) for x in allocations
        )
        record["QuantityApplied"] = requested_qty
        record["Notes"] = (
            "Allocated to earlier matching sale line(s); "
            "earliest eligible sales consumed first"
        )
        audit.append(record)

        matched_cancellation_rows += 1
        cancelled_quantity += requested_qty

    sales["Quantity"] = sales["RemainingQuantity"]
    sales = sales.loc[sales["Quantity"].gt(0)].copy()

    internal_columns = [
        "_SourceRowID",
        "InvoiceText",
        "SaleLineID",
        "RemainingQuantity",
        "OriginalInvoice",
    ]
    sales = sales.drop(
        columns=[c for c in internal_columns if c in sales.columns]
    )

    sales["TotalAmount"] = (
        sales["Quantity"] * sales["Price"]
    ).round(2)
    sales = sales.reset_index(drop=True)

    audit_df = pd.DataFrame(audit)

    if audit_path is not None:
        path = Path(audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        audit_df.to_csv(path, index=False)

    quality_metrics = {
        "initial_rows": initial_rows,
        "final_rows": len(sales),
        "rows_removed": initial_rows - len(sales),
        "missing_customer_ids": missing_customer_ids,
        "duplicate_rows_preserved": duplicate_rows,
        "cancellation_rows": int(is_cancel.sum()),
        "cancellation_lines_audited": len(cancellations),
        "matched_cancellation_lines": matched_cancellation_rows,
        "unmatched_cancellation_lines": int(
            audit_df["Status"].eq("unmatched").sum()
        ) if not audit_df.empty else 0,
        "ambiguous_cancellation_lines": int(
            audit_df["Status"].eq("ambiguous").sum()
        ) if not audit_df.empty else 0,
        "cancelled_quantity_applied": cancelled_quantity,
        "final_customers": int(sales["Customer ID"].nunique()),
        "final_invoices": int(sales["Invoice"].nunique()),
        "final_products": int(sales["StockCode"].nunique()),
        "final_countries": int(sales["Country"].nunique())
        if "Country" in sales.columns else None,
        "total_revenue": round(float(sales["TotalAmount"].sum()), 2),
    }

    return sales, quality_metrics
