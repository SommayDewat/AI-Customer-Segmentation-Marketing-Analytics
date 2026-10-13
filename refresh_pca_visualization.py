
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.visualization import plot_pca_clusters

REPORTS = Path("reports")

customers_path = REPORTS / "customer_segments.csv"
output_path = REPORTS / "pca_customer_segments.png"

customers = pd.read_csv(customers_path)

required = {
    "CustomerID",
    "Recency",
    "Frequency",
    "Monetary",
    "Cluster",
    "Segment",
}

missing = required - set(customers.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

if customers["CustomerID"].duplicated().any():
    raise ValueError("CustomerID values must be unique.")

# Recreate the pipeline's exact model feature transformation.
model_columns = ["Recency", "Frequency", "Monetary"]
model_data = customers[model_columns].copy()

if not np.isfinite(model_data.to_numpy(dtype=float)).all():
    raise ValueError("RFM data contains non-finite values.")

if (model_data < 0).any().any():
    raise ValueError("RFM data contains negative values.")

log_data = np.log1p(model_data)
scaled = StandardScaler().fit_transform(log_data)

# Verify that recreated clusters match the saved report.
model = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10,
)
recreated_clusters = model.fit_predict(scaled)

if not np.array_equal(
    recreated_clusters,
    customers["Cluster"].to_numpy(dtype=int),
):
    raise ValueError(
        "Recreated cluster IDs do not match the saved report. "
        "PCA was not generated."
    )

scaled_df = pd.DataFrame(
    scaled,
    columns=model_columns,
    index=customers.index,
)

# Use the existing PCA plotting utility.
plot_pca_clusters(
    scaled_df,
    customers,
    output_path=str(output_path),
)

print("\nPCA refresh completed.")
print("Customers visualized:", len(customers))
print("Features used: Recency, Frequency, Monetary")
print("Output:", output_path)
