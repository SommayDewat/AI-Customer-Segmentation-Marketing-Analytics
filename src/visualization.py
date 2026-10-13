"""
Visualization utilities for the
AI Customer Segmentation & Marketing Analytics project.
"""

from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.dashboard_theme import SEGMENT_COLORS, SEGMENT_ORDER


# Keep saved Matplotlib/Seaborn charts visually aligned with the Streamlit app.
DARK_BACKGROUND = "#0B1120"
AXES_BACKGROUND = "#111827"
TEXT_COLOR = "#E5E7EB"
GRID_COLOR = "#334155"
METRIC_COLORS = {
    "AvgRecency": "#FBBF24",
    "AvgFrequency": "#60A5FA",
    "AvgMonetary": "#2DD4BF",
    "AvgAOV": "#60A5FA",
    "AvgMonetary_value": "#2DD4BF",
}

sns.set_theme(
    style="darkgrid",
    rc={
        "figure.facecolor": DARK_BACKGROUND,
        "axes.facecolor": AXES_BACKGROUND,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.titlecolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.45,
        "savefig.facecolor": DARK_BACKGROUND,
        "savefig.edgecolor": DARK_BACKGROUND,
        "legend.facecolor": AXES_BACKGROUND,
        "legend.edgecolor": GRID_COLOR,
    },
)


def _style_axes(ax) -> None:
    """Apply consistent dark-theme finishing touches to a Matplotlib axes."""
    ax.set_facecolor(AXES_BACKGROUND)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, alpha=0.45)


def plot_cluster_distribution(
    profile: pd.DataFrame,
    output_path: str = "reports/cluster_distribution.png",
) -> None:
    """
    Plot the number of customers in each segment.
    """

    required_columns = {
        "Segment",
        "Customers",
    }

    missing = required_columns - set(profile.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    plot_data = profile.sort_values(
        "Customers",
        ascending=False,
    )

    plt.figure(figsize=(10, 6))

    ax = sns.barplot(
        data=plot_data,
        x="Segment",
        y="Customers",
        hue="Segment",
        hue_order=SEGMENT_ORDER,
        palette=SEGMENT_COLORS,
        dodge=False,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    _style_axes(ax)

    plt.title("Customer Distribution by Segment")
    plt.xlabel("Customer Segment")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_revenue_contribution(
    profile: pd.DataFrame,
    output_path: str = "reports/revenue_contribution.png",
) -> None:
    """
    Plot revenue contribution by customer segment.
    """

    required_columns = {
        "Segment",
        "Revenue",
    }

    missing = required_columns - set(profile.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    plot_data = profile.sort_values(
        "Revenue",
        ascending=False,
    )

    plt.figure(figsize=(10, 6))

    ax = sns.barplot(
        data=plot_data,
        x="Segment",
        y="Revenue",
        hue="Segment",
        hue_order=SEGMENT_ORDER,
        palette=SEGMENT_COLORS,
        dodge=False,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    _style_axes(ax)

    plt.title("Revenue Contribution by Segment")
    plt.xlabel("Customer Segment")
    plt.ylabel("Revenue")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_rfm_comparison(
    profile: pd.DataFrame,
    output_path: str = "reports/rfm_comparison.png",
) -> None:
    """
    Compare average RFM-related metrics across segments.
    """

    required_columns = {
        "Segment",
        "AvgRecency",
        "AvgFrequency",
        "AvgMonetary",
    }

    missing = required_columns - set(profile.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    plot_data = profile[
        [
            "Segment",
            "AvgRecency",
            "AvgFrequency",
            "AvgMonetary",
        ]
    ].copy()

    normalized_data = plot_data.copy()

    for column in [
        "AvgRecency",
        "AvgFrequency",
        "AvgMonetary",
    ]:
        min_value = normalized_data[column].min()
        max_value = normalized_data[column].max()

        if max_value == min_value:
            normalized_data[column] = 0
        else:
            normalized_data[column] = (
                normalized_data[column] - min_value
            ) / (max_value - min_value)

    melted = normalized_data.melt(
        id_vars="Segment",
        var_name="Metric",
        value_name="NormalizedValue",
    )

    plt.figure(figsize=(12, 7))

    ax = sns.barplot(
        data=melted,
        x="Segment",
        y="NormalizedValue",
        hue="Metric",
        palette={
            "AvgRecency": METRIC_COLORS["AvgRecency"],
            "AvgFrequency": METRIC_COLORS["AvgFrequency"],
            "AvgMonetary": METRIC_COLORS["AvgMonetary"],
        },
        hue_order=["AvgRecency", "AvgFrequency", "AvgMonetary"],
    )
    _style_axes(ax)

    plt.title("Normalized RFM Comparison by Segment")
    plt.xlabel("Customer Segment")
    plt.ylabel("Normalized Score")
    plt.xticks(rotation=20, ha="right")
    legend = plt.legend(title="Metric")
    if legend is not None:
        legend.get_frame().set_facecolor(AXES_BACKGROUND)
        legend.get_frame().set_edgecolor(GRID_COLOR)
        for label in legend.get_texts():
            label.set_color(TEXT_COLOR)
        legend.get_title().set_color(TEXT_COLOR)
    plt.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_customer_value(
    profile: pd.DataFrame,
    output_path: str = "reports/customer_value_comparison.png",
) -> None:
    """
    Compare average monetary value and average order value.
    """

    required_columns = {
        "Segment",
        "AvgMonetary",
        "AvgAOV",
    }

    missing = required_columns - set(profile.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    plot_data = profile[
        [
            "Segment",
            "AvgMonetary",
            "AvgAOV",
        ]
    ].melt(
        id_vars="Segment",
        var_name="Metric",
        value_name="Value",
    )

    plt.figure(figsize=(12, 7))

    ax = sns.barplot(
        data=plot_data,
        x="Segment",
        y="Value",
        hue="Metric",
        palette={
            "AvgMonetary": METRIC_COLORS["AvgMonetary"],
            "AvgAOV": METRIC_COLORS["AvgAOV"],
        },
        hue_order=["AvgMonetary", "AvgAOV"],
    )
    _style_axes(ax)

    plt.title("Customer Value Comparison")
    plt.xlabel("Customer Segment")
    plt.ylabel("Average Value")
    plt.xticks(rotation=20, ha="right")
    legend = plt.legend(title="Metric")
    if legend is not None:
        legend.get_frame().set_facecolor(AXES_BACKGROUND)
        legend.get_frame().set_edgecolor(GRID_COLOR)
        for label in legend.get_texts():
            label.set_color(TEXT_COLOR)
        legend.get_title().set_color(TEXT_COLOR)
    plt.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_recency_vs_monetary(
    customer_data: pd.DataFrame,
    output_path: str = "reports/recency_vs_monetary.png",
) -> None:
    """
    Plot customer recency against monetary value.
    """

    required_columns = {
        "Recency",
        "Monetary",
        "Segment",
    }

    missing = required_columns - set(customer_data.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    plt.figure(figsize=(10, 7))

    ax = sns.scatterplot(
        data=customer_data,
        x="Recency",
        y="Monetary",
        hue="Segment",
        hue_order=SEGMENT_ORDER,
        palette=SEGMENT_COLORS,
        alpha=0.65,
        s=38,
    )
    _style_axes(ax)

    plt.title("Customer Recency vs Monetary Value")
    plt.xlabel("Recency (Days)")
    plt.ylabel("Monetary Value")
    plt.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_cluster_evaluation(
    evaluation_results: pd.DataFrame,
    output_directory: str = "reports",
) -> None:
    """
    Generate separate evaluation plots for K-Means clustering.

    Creates:
    - kmeans_inertia.png
    - silhouette_score.png
    - davies_bouldin_index.png
    """

    required_columns = {
        "K",
        "Inertia",
        "SilhouetteScore",
        "DaviesBouldinIndex",
    }

    missing = required_columns - set(evaluation_results.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 1. K-Means Inertia
    # -----------------------------
    plt.figure(figsize=(8, 5))

    ax = plt.gca()
    ax.plot(
        evaluation_results["K"],
        evaluation_results["Inertia"],
        marker="o",
        color="#2DD4BF",
        linewidth=2.5,
        markersize=7,
    )
    _style_axes(ax)

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("K-Means Inertia by Number of Clusters")
    plt.xticks(evaluation_results["K"])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = output_dir / "kmeans_inertia.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # 2. Silhouette Score
    # -----------------------------
    plt.figure(figsize=(8, 5))

    ax = plt.gca()
    ax.plot(
        evaluation_results["K"],
        evaluation_results["SilhouetteScore"],
        marker="o",
        color="#60A5FA",
        linewidth=2.5,
        markersize=7,
    )
    _style_axes(ax)

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score by Number of Clusters")
    plt.xticks(evaluation_results["K"])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = output_dir / "silhouette_score.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # 3. Davies-Bouldin Index
    # -----------------------------
    plt.figure(figsize=(8, 5))

    ax = plt.gca()
    ax.plot(
        evaluation_results["K"],
        evaluation_results["DaviesBouldinIndex"],
        marker="o",
        color="#FBBF24",
        linewidth=2.5,
        markersize=7,
    )
    _style_axes(ax)

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Davies-Bouldin Index")
    plt.title("Davies-Bouldin Index by Number of Clusters")
    plt.xticks(evaluation_results["K"])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = output_dir / "davies_bouldin_index.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Cluster evaluation plots saved successfully:")
    print(f" - {output_dir / 'kmeans_inertia.png'}")
    print(f" - {output_dir / 'silhouette_score.png'}")
    print(f" - {output_dir / 'davies_bouldin_index.png'}")


def plot_pca_clusters(
    scaled_data: pd.DataFrame,
    customer_data: pd.DataFrame,
    output_path: str = "reports/pca_customer_segments.png",
) -> None:
    """
    Visualize customer segments using 2D PCA.
    PCA is used for visualization, not model training.
    """

    if not {"Cluster", "Segment"}.issubset(customer_data.columns):
        raise ValueError(
            "customer_data must contain Cluster and Segment."
        )

    if len(scaled_data) != len(customer_data):
        raise ValueError(
            "Both datasets must have the same number of rows."
        )

    if scaled_data.shape[1] < 2:
        raise ValueError("PCA requires at least two features.")

    if scaled_data.isna().any().any():
        raise ValueError("Scaled features contain missing values.")

    # Reduce the 8-dimensional feature space to 2 dimensions
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled_data)

    pca_df = pd.DataFrame(
        components,
        columns=["PC1", "PC2"],
    )

    # Customer rows must match the scaled feature row order
    pca_df["Segment"] = customer_data["Segment"].to_numpy()

    variance = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(11, 7), facecolor=DARK_BACKGROUND)
    ax.set_facecolor(AXES_BACKGROUND)

    for segment in SEGMENT_ORDER:
        group = pca_df[pca_df["Segment"] == segment]
        if group.empty:
            continue
        ax.scatter(
            group["PC1"],
            group["PC2"],
            label=segment,
            color=SEGMENT_COLORS.get(segment, "#CBD5E1"),
            alpha=0.65,
            s=22,
            edgecolors="none",
        )

    ax.set_title("Customer Segmentation - PCA Analysis")
    ax.set_xlabel(f"PC1 ({variance[0]:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({variance[1]:.2f}% variance)")
    ax.legend(
        title="Customer Segment",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(color=GRID_COLOR, alpha=0.45)
    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_facecolor(AXES_BACKGROUND)
        legend.get_frame().set_edgecolor(GRID_COLOR)
        for label in legend.get_texts():
            label.set_color(TEXT_COLOR)
        legend.get_title().set_color(TEXT_COLOR)

    fig.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"PCA visualization saved: {path}")
    print(
        f"Explained variance: "
        f"PC1={variance[0]:.2f}%, "
        f"PC2={variance[1]:.2f}%"
    )
