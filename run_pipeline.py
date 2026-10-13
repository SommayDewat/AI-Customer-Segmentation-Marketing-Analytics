
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import create_customer_features
from src.rfm_analysis import create_rfm
from src.insights import (
    assign_dynamic_segments,
    profile_clusters,
    generate_segment_summary,
)


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "raw" / "online_retail_II.xlsx"
OUTPUT_DIR = PROJECT_DIR / "reports" / "pipeline_test"

EXPECTED_SEGMENTS = {
    "VIP Loyal Customers",
    "Engaged Growth Customers",
    "At-Risk Valuable Customers",
    "Dormant Low-Value Customers",
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load the original dataset.
    print("1. Loading original dataset...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    raw = load_data(
        str(DATA_FILE),
        sheet_name="Year 2010-2011",
    )

    # 2. Clean transactions and reconcile cancellations.
    print("2. Reconciling cancellations...")

    cleaned, quality_metrics = preprocess_data(
        raw,
        audit_path=str(OUTPUT_DIR / "cancellation_audit.csv"),
    )

    cleaned.to_csv(
        OUTPUT_DIR / "cleaned_transactions.csv",
        index=False,
    )

    # 3. Build customer features and RFM scores.
    print("3. Creating customer features and RFM...")

    features = create_customer_features(cleaned)
    rfm = create_rfm(cleaned)

    features = features.rename(
        columns={"Customer ID": "CustomerID"}
    )

    customers = features.merge(
        rfm[
            [
                "CustomerID",
                "R_Score",
                "F_Score",
                "M_Score",
                "RFMScore",
                "RFMTotalScore",
            ]
        ],
        on="CustomerID",
        how="inner",
        validate="one_to_one",
    )

    if customers.empty:
        raise ValueError("No customer records remain after feature engineering.")

    if customers["CustomerID"].duplicated().any():
        raise ValueError("Duplicate CustomerID values found.")

    # 4. Transform and scale RFM features.
    model_columns = ["Recency", "Frequency", "Monetary"]
    model_data = customers[model_columns].copy()

    if not np.isfinite(model_data.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values found in RFM features.")

    if (model_data < 0).any().any():
        raise ValueError(
            "Negative RFM values found; log1p cannot be applied safely."
        )

    log_data = np.log1p(model_data)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(log_data)

    scaled_df = pd.DataFrame(
        scaled,
        columns=model_columns,
        index=customers.index,
    )

    # 5. Evaluate K-Means cluster counts.
    print("4. Evaluating K-Means cluster counts...")

    evaluation_rows = []

    for k in range(2, 9):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        labels = model.fit_predict(scaled_df)

        evaluation_rows.append(
            {
                "K": k,
                "Inertia": float(model.inertia_),
                "SilhouetteScore": float(
                    silhouette_score(scaled_df, labels)
                ),
                "DaviesBouldinIndex": float(
                    davies_bouldin_score(scaled_df, labels)
                ),
            }
        )

    evaluation = pd.DataFrame(evaluation_rows)

    # 6. Fit the four-cluster model.
    print("5. Fitting the four-cluster model...")

    final_model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10,
    )

    customers["Cluster"] = final_model.fit_predict(scaled_df)

    # Assign business labels based on the observed cluster profiles.
    # These IDs must be revalidated if the model or input data changes.

    segment_map = {
        0: "Engaged Growth Customers",
        1: "At-Risk Valuable Customers",
        2: "VIP Loyal Customers",
        3: "Dormant Low-Value Customers",
    }


    customers = assign_dynamic_segments(
        customers,
        segment_map,
    )

    # Generate profiles and summary exactly once.
    profiles = profile_clusters(
        customers,
        cluster_column="Cluster",
    )

    summary = generate_segment_summary(
        customers,
        cluster_column="Cluster",
    )

    # 7. Validate the outputs before saving reports.
    print("6. Validating results...")

    actual_segments = set(customers["Segment"].dropna().unique())

    if actual_segments != EXPECTED_SEGMENTS:
        raise ValueError(
            "Unexpected segment labels. "
            f"Expected {EXPECTED_SEGMENTS}; found {actual_segments}"
        )

    if customers["CustomerID"].isna().any():
        raise ValueError("Missing CustomerID values found.")

    if not customers["CustomerID"].is_unique:
        raise ValueError("CustomerID values are not unique.")

    expected_cancellations = int(
        raw["Invoice"]
        .astype(str)
        .str.startswith("C", na=False)
        .sum()
    )

    audit_path = OUTPUT_DIR / "cancellation_audit.csv"
    audit = pd.read_csv(audit_path)

    if len(audit) != expected_cancellations:
        raise ValueError(
            "Cancellation audit count does not match the raw dataset: "
            f"{len(audit)} audited versus {expected_cancellations} expected."
        )

    if "Status" not in audit.columns:
        raise ValueError("Cancellation audit is missing the Status column.")

    # 8. Write reports to the test directory only.
    print("7. Writing test reports...")

    customers.to_csv(
        OUTPUT_DIR / "customer_segments.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "segment_summary.csv",
        index=False,
    )

    profiles.to_csv(
        OUTPUT_DIR / "cluster_profiles.csv",
        index=False,
    )

    evaluation.to_csv(
        OUTPUT_DIR / "cluster_evaluation.csv",
        index=False,
    )

    (OUTPUT_DIR / "quality_metrics.json").write_text(
        json.dumps(quality_metrics, indent=2, default=str),
        encoding="utf-8",
    )

    # 9. Print a concise validation report.
    print("\nPIPELINE TEST COMPLETE")
    print("Output directory:", OUTPUT_DIR)
    print("Customers:", len(customers))
    print("Customer revenue:", round(customers["Monetary"].sum(), 2))
    print(
        "Cleaned transaction revenue:",
        quality_metrics.get("total_revenue", "Not available"),
    )

    print("\nCluster-to-segment mapping:")
    print(
        customers.groupby(["Cluster", "Segment"])
        .size()
        .to_string()
    )

    print("\nSegment counts:")
    print(customers["Segment"].value_counts().to_string())

    print("\nCancellation audit")
    print("Audit rows:", len(audit))
    print("Expected cancellation rows:", expected_cancellations)
    print("Matched:", int(audit["Status"].eq("matched").sum()))
    print("Unmatched:", int(audit["Status"].eq("unmatched").sum()))

    print("\nCluster evaluation:")
    print(evaluation.round(4).to_string(index=False))

    print("\nActive dashboard reports have not been overwritten.")


if __name__ == "__main__":
    main()
