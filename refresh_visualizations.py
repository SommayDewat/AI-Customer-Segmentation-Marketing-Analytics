
from pathlib import Path

import pandas as pd

from src.visualization import (
    plot_cluster_distribution,
    plot_revenue_contribution,
    plot_rfm_comparison,
    plot_customer_value,
    plot_recency_vs_monetary,
    plot_cluster_evaluation,
)

REPORTS = Path("reports")

profiles_path = REPORTS / "cluster_profiles.csv"
customers_path = REPORTS / "customer_segments.csv"
evaluation_path = REPORTS / "cluster_evaluation.csv"

# Load the latest pipeline reports.
profiles = pd.read_csv(profiles_path)
customers = pd.read_csv(customers_path)
evaluation = pd.read_csv(evaluation_path)

# Validate the inputs before replacing any charts.
assert profiles["Segment"].nunique() == 4, "Expected four segments."
assert profiles["Customers"].sum() == 4327, "Unexpected customer total."
assert customers["CustomerID"].nunique() == 4327, "Unexpected customer count."

assert (
    profiles["Customers"].sum()
    == customers.groupby("Segment")["CustomerID"].nunique().sum()
), "Segment counts do not reconcile."

# Regenerate the five charts using current report data.
plot_cluster_distribution(profiles)
plot_revenue_contribution(profiles)
plot_rfm_comparison(profiles)
plot_customer_value(profiles)
plot_recency_vs_monetary(customers)

# Regenerate clustering evaluation charts from current metrics.
plot_cluster_evaluation(evaluation)

print("\nVisualization refresh completed.")
print("Customers:", customers["CustomerID"].nunique())
print("Revenue by segment:")
print(
    profiles[["Segment", "Customers", "Revenue", "RevenuePercentage"]]
    .sort_values("Revenue", ascending=False)
    .to_string(index=False)
)
