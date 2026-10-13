from src.dashboard_theme import SEGMENT_COLORS, SEGMENT_ORDER
from pathlib import Path
import plotly.express as px
import pandas as pd
import streamlit as st

# --------------------------------------------------
# Configuration
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_DIR / "reports"

CUSTOMER_FILE = REPORTS_DIR / "customer_segments.csv"
SUMMARY_FILE = REPORTS_DIR / "segment_summary.csv"
PROFILE_FILE = REPORTS_DIR / "cluster_profiles.csv"

SEGMENT_ORDER = [
    "VIP Loyal Customers",
    "Engaged Growth Customers",
    "At-Risk Valuable Customers",
    "Dormant Low-Value Customers",
]

st.set_page_config(
    page_title="AI Customer Segmentation",
    page_icon="ðŸ“Š",
    layout="wide",
)


# --------------------------------------------------
# Data loading
# --------------------------------------------------
@st.cache_data
def load_reports():
    required_files = [
        CUSTOMER_FILE,
        SUMMARY_FILE,
        PROFILE_FILE,
    ]

    missing_files = [
        str(path.name)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing report files: " + ", ".join(missing_files)
        )

    customers = pd.read_csv(CUSTOMER_FILE)
    summary = pd.read_csv(SUMMARY_FILE)
    profiles = pd.read_csv(PROFILE_FILE)

    required_customer_columns = {
        "CustomerID",
        "Recency",
        "Frequency",
        "Monetary",
        "Segment",
        "MarketingRecommendation",
    }

    missing_columns = (
        required_customer_columns - set(customers.columns)
    )

    if missing_columns:
        raise ValueError(
            "Customer report is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    customers["CustomerID"] = (
        customers["CustomerID"].astype("Int64").astype(str)
    )

    customers["Monetary"] = pd.to_numeric(
        customers["Monetary"], errors="coerce"
    )

    customers["Recency"] = pd.to_numeric(
        customers["Recency"], errors="coerce"
    )

    customers["Frequency"] = pd.to_numeric(
        customers["Frequency"], errors="coerce"
    )

    return customers, summary, profiles


# --------------------------------------------------
# Load data and handle errors
# --------------------------------------------------
st.title("AI Customer Segmentation & Marketing Analytics")
st.caption(
    "Understand customer value, identify engagement opportunities, "
    "and support data-driven marketing decisions."
)

try:
    customer_data, segment_summary, cluster_profiles = load_reports()
except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Unable to load dashboard data: {error}")
    st.info(
        "Check that the CSV reports exist in the project's reports folder."
    )
    st.stop()


# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------
st.sidebar.header("Dashboard Filters")

available_segments = [
    segment
    for segment in SEGMENT_ORDER
    if segment in customer_data["Segment"].dropna().unique()
]

other_segments = sorted(
    set(customer_data["Segment"].dropna().unique())
    - set(available_segments)
)

available_segments.extend(other_segments)

selected_segments = st.sidebar.multiselect(
    "Customer segments",
    options=available_segments,
    default=available_segments,
)

min_recency = int(customer_data["Recency"].min())
max_recency = int(customer_data["Recency"].max())

recency_range = st.sidebar.slider(
    "Recency (days since last purchase)",
    min_value=min_recency,
    max_value=max_recency,
    value=(min_recency, max_recency),
)

filtered_data = customer_data[
    customer_data["Segment"].isin(selected_segments)
    & customer_data["Recency"].between(
        recency_range[0], recency_range[1]
    )
].copy()


# --------------------------------------------------
# KPI cards
# --------------------------------------------------
st.subheader("Business Overview")

total_customers = len(filtered_data)
total_revenue = filtered_data["Monetary"].sum()
average_value = (
    filtered_data["Monetary"].mean()
    if total_customers
    else 0
)
average_recency = (
    filtered_data["Recency"].mean()
    if total_customers
    else 0
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric("Customers", f"{total_customers:,}")
kpi2.metric("Historical Revenue", f"Â£{total_revenue:,.2f}")
kpi3.metric("Avg. Customer Revenue", f"Â£{average_value:,.2f}")
kpi4.metric("Avg. Recency", f"{average_recency:.1f} days")

st.caption(
    "Revenue is calculated from the filtered historical customer data. "
    "Currency is shown in GBP because the source retail dataset uses "
    "prices from a UK retailer."
)



# --------------------------------------------------
# Segment distribution and revenue
# --------------------------------------------------
st.divider()
st.subheader("Customer Segment Performance")

if filtered_data.empty:
    st.warning("No customers match the selected filters.")
else:
    segment_metrics = (
        filtered_data.groupby("Segment", as_index=False)
        .agg(
            Customers=("CustomerID", "nunique"),
            Revenue=("Monetary", "sum"),
            AverageRecency=("Recency", "mean"),
            AverageFrequency=("Frequency", "mean"),
        )
    )

    # Sort charts so the largest segment appears at the top
    customer_chart_data = segment_metrics.sort_values(
        "Customers", ascending=True
    )

    revenue_chart_data = segment_metrics.sort_values(
        "Revenue", ascending=True
    )

    left, right = st.columns(2)

    with left:
        st.markdown("**Customer Count by Segment**")

        fig_customers = px.bar(
            customer_chart_data,
            x="Customers",
            y="Segment",
            orientation="h",
            text="Customers",
            color="Segment",
            color_discrete_map=SEGMENT_COLORS, category_orders={"Segment": SEGMENT_ORDER},
            template="plotly_dark",
            labels={
                "Customers": "Number of Customers",
                "Segment": "",
            },
        )

        fig_customers.update_traces(
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Customers: %{x:,}"
                "<extra></extra>"
            ),
        )

        fig_customers.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            height=360,
            margin=dict(l=10, r=35, t=20, b=20),
            yaxis=dict(categoryorder="total ascending"),
        )

        st.plotly_chart(fig_customers, width="stretch")

    with right:
        st.markdown("**Historical Revenue by Segment**")

        fig_revenue = px.bar(
            revenue_chart_data,
            x="Revenue",
            y="Segment",
            orientation="h",
            text="Revenue",
            color="Segment", color_discrete_map=SEGMENT_COLORS, category_orders={"Segment": SEGMENT_ORDER},
            template="plotly_dark",
            labels={
                "Revenue": "Historical Revenue (Â£)",
                "Segment": "",
            },
        )

        fig_revenue.update_traces(
            texttemplate="Â£%{x:,.0f}",
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Historical revenue: Â£%{x:,.2f}"
                "<extra></extra>"
            ),
        )

        fig_revenue.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            height=360,
            margin=dict(l=10, r=75, t=20, b=20),
            yaxis=dict(categoryorder="total ascending"),
        )

        st.plotly_chart(
            fig_revenue,
            width="stretch",
        )

    # Clean, readable summary table
    display_metrics = segment_metrics.copy()

    display_metrics["Revenue"] = display_metrics["Revenue"].map(
        lambda value: f"Â£{value:,.2f}"
    )

    display_metrics["AverageRecency"] = (
        display_metrics["AverageRecency"].round(1)
    )

    display_metrics["AverageFrequency"] = (
        display_metrics["AverageFrequency"].round(2)
    )

    display_metrics = display_metrics.rename(
        columns={
            "AverageRecency": "Avg. Recency (days)",
            "AverageFrequency": "Avg. Orders",
        }
    )

    st.dataframe(
        display_metrics,
        width="stretch",
        hide_index=True,
    )



# --------------------------------------------------
# Key Business Insights
# --------------------------------------------------
st.divider()
st.subheader("Key Business Insights")

if not filtered_data.empty:
    total_revenue_all = filtered_data["Monetary"].sum()

    segment_insights = (
        filtered_data.groupby("Segment")
        .agg(
            Customers=("CustomerID", "nunique"),
            Revenue=("Monetary", "sum"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
        )
    )

    segment_insights["RevenueShare"] = (
        segment_insights["Revenue"] / total_revenue_all * 100
        if total_revenue_all > 0
        else 0
    )

    top_revenue_segment = segment_insights["Revenue"].idxmax()
    largest_segment = segment_insights["Customers"].idxmax()

    insight1, insight2 = st.columns(2)

    with insight1:
        st.markdown("**Highest Revenue Segment**")
        st.metric(
            "Segment",
            top_revenue_segment,
            f"{segment_insights.loc[top_revenue_segment, 'RevenueShare']:.1f}% of filtered revenue",
        )

    with insight2:
        st.markdown("**Largest Segment by Customer Count**")
        st.metric(
            "Segment",
            largest_segment,
            f"{int(segment_insights.loc[largest_segment, 'Customers']):,} customers",
        )

    st.markdown("**Suggested business actions**")

    for segment, row in segment_insights.iterrows():
        if segment == "VIP Loyal Customers":
            action = (
                "Protect retention with loyalty benefits, early access, "
                "and relevant personalized offers."
            )
        elif segment == "Engaged Growth Customers":
            action = (
                "Increase purchase frequency through bundles, "
                "cross-selling, and product recommendations."
            )
        elif segment == "At-Risk Valuable Customers":
            action = (
                "Test targeted win-back campaigns and monitor "
                "whether engagement improves."
            )
        else:
            action = (
                "Use low-cost reactivation campaigns and measure "
                "response before increasing campaign spend."
            )

        st.write(
            f"**{segment}** â€” {action}"
        )
else:
    st.info("Select at least one customer to see business insights.")


# --------------------------------------------------
# Marketing recommendations
# --------------------------------------------------
st.divider()
st.subheader("Marketing Strategy")

if not filtered_data.empty:
    recommendation_data = (
        filtered_data.groupby(
            ["Segment", "MarketingRecommendation"],
            dropna=False,
        )
        .size()
        .reset_index(name="Customers")
    )

    for segment in SEGMENT_ORDER:
        segment_rows = recommendation_data[
            recommendation_data["Segment"] == segment
        ]

        if segment_rows.empty:
            continue

        with st.expander(segment):
            st.write(
                segment_rows["MarketingRecommendation"]
                .dropna()
                .iloc[0]
                if not segment_rows["MarketingRecommendation"]
                .dropna()
                .empty
                else "No recommendation available."
            )

            st.caption(
                f"Customers matching current filters: "
                f"{int(segment_rows['Customers'].sum()):,}"
            )



# --------------------------------------------------
# Customer Explorer
# --------------------------------------------------
st.divider()
st.subheader("Customer Explorer")
st.caption(
    "Search customers, review purchasing behaviour, "
    "and explore segment-specific marketing actions."
)

search_id = st.text_input(
    "Search by Customer ID",
    placeholder="Enter a customer ID, e.g. 12347",
    key="customer_search",
)

explorer_data = filtered_data.copy()

if search_id.strip():
    explorer_data = explorer_data[
        explorer_data["CustomerID"].str.contains(
            search_id.strip(),
            regex=False,
            na=False,
        )
    ]

if explorer_data.empty:
    st.warning("No customers match your search and current filters.")
else:
    # Summary table: keep it compact and readable
    visible_columns = [
        column
        for column in [
            "CustomerID",
            "Segment",
            "Recency",
            "Frequency",
            "Monetary",
            "RFMTotalScore",
        ]
        if column in explorer_data.columns
    ]

    table_data = explorer_data[visible_columns].copy()

    table_data = table_data.rename(
        columns={
            "CustomerID": "Customer ID",
            "Recency": "Recency (days)",
            "Frequency": "Orders",
            "Monetary": "Historical Revenue (Â£)",
            "RFMTotalScore": "RFM Score",
        }
    )

    if "Historical Revenue (Â£)" in table_data.columns:
        table_data["Historical Revenue (Â£)"] = (
            table_data["Historical Revenue (Â£)"]
            .map(lambda value: f"Â£{value:,.2f}")
        )

    st.dataframe(
        table_data,
        width="stretch",
        hide_index=True,
        height=360,
    )

    # Export the current customer search results
    export_data = explorer_data[visible_columns].copy()

    st.download_button(
        label="Download customer results (CSV)",
        data=export_data.to_csv(index=False).encode("utf-8"),
        file_name="customer_search_results.csv",
        mime="text/csv",
        key="download_customer_results",
    )

    # Individual customer profile
    st.markdown("### Customer Profile")

    customer_options = explorer_data["CustomerID"].astype(str).tolist()

    selected_customer = st.selectbox(
        "Select a customer",
        options=customer_options,
        key="selected_customer_profile",
    )

    customer_match = explorer_data[
        explorer_data["CustomerID"].astype(str) == selected_customer
    ]

    if not customer_match.empty:
        customer = customer_match.iloc[0]

        profile1, profile2, profile3, profile4 = st.columns(4)

        profile1.metric(
            "Segment",
            str(customer["Segment"]),
        )

        profile2.metric(
            "Days Since Purchase",
            f"{customer['Recency']:,.0f}",
        )

        profile3.metric(
            "Number of Orders",
            f"{customer['Frequency']:,.0f}",
        )

        profile4.metric(
            "Historical Revenue",
            f"Â£{customer['Monetary']:,.2f}",
        )

        st.markdown("**Marketing Recommendation**")

        recommendation = customer.get(
            "MarketingRecommendation",
            "No recommendation available.",
        )

        st.info(str(recommendation))

        if "RFMScore" in customer.index:
            st.caption(f"RFM score: {customer['RFMScore']}")

        if "RFMTotalScore" in customer.index:
            st.caption(
                f"Combined RFM score: "
                f"{customer['RFMTotalScore']}"
            )



# --------------------------------------------------
# Existing analytical charts
# --------------------------------------------------
st.divider()
st.subheader("Analytical Visualizations")

chart_files = {
    "Customer Segment Distribution": "cluster_distribution.png",
    "Revenue Contribution": "revenue_contribution.png",
    "RFM Comparison": "rfm_comparison.png",
    "Customer Value Comparison": "customer_value_comparison.png",
    "Recency vs Monetary": "recency_vs_monetary.png",
    "K-Means Inertia": "kmeans_inertia.png",
    "Silhouette Score": "silhouette_score.png",
    "Davies-Bouldin Index": "davies_bouldin_index.png",
    "PCA Customer Segments": "pca_customer_segments.png",
}

existing_charts = {
    title: REPORTS_DIR / filename
    for title, filename in chart_files.items()
    if (REPORTS_DIR / filename).exists()
}

if existing_charts:
    selected_chart = st.selectbox(
        "Choose a visualization",
        options=list(existing_charts.keys()),
    )

    st.image(
        str(existing_charts[selected_chart]),
        caption=selected_chart,
        width="stretch",
    )


else:
    st.info("No saved visualization images were found.")

st.divider()
st.caption(
    "Customer segmentation is based on historical purchasing behaviour. "
    "'At-Risk' is a business segment label, not a validated churn prediction."
)



