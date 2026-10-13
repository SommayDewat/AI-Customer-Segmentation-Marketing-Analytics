from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# Configuration
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_DIR / "reports"
CUSTOMER_FILE = REPORTS_DIR / "customer_segments.csv"

SEGMENT_COLORS = {
    "VIP Loyal Customers": "#22C55E",
    "Engaged Growth Customers": "#3B82F6",
    "At-Risk Valuable Customers": "#F59E0B",
    "Dormant Low-Value Customers": "#EF4444",
}


st.set_page_config(
    page_title="RFM & Customer Value",
    page_icon="💰",
    layout="wide",
)


# --------------------------------------------------
# Load customer data
# --------------------------------------------------
@st.cache_data
def load_customer_data():
    if not CUSTOMER_FILE.exists():
        raise FileNotFoundError(
            f"Customer report not found: {CUSTOMER_FILE}"
        )

    df = pd.read_csv(CUSTOMER_FILE)

    required_columns = [
        "CustomerID",
        "Recency",
        "Frequency",
        "Monetary",
        "R_Score",
        "F_Score",
        "M_Score",
        "RFMTotalScore",
        "AverageOrderValue",
        "RevenueContribution",
        "Segment",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    numeric_columns = [
        "Recency",
        "Frequency",
        "Monetary",
        "R_Score",
        "F_Score",
        "M_Score",
        "RFMTotalScore",
        "AverageOrderValue",
        "RevenueContribution",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


try:
    customers = load_customer_data()
except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Unable to load customer data: {error}")
    st.stop()


# --------------------------------------------------
# Page heading
# --------------------------------------------------
st.title("RFM & Customer Value")
st.caption(
    "Analyse customer recency, purchase frequency, monetary value, "
    "and segment-level customer economics."
)


# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------
st.sidebar.header("RFM Filters")

available_segments = sorted(customers["Segment"].dropna().unique())

selected_segments = st.sidebar.multiselect(
    "Customer segments",
    options=available_segments,
    default=available_segments,
    key="rfm_segments_filter",
)

filtered = customers[
    customers["Segment"].isin(selected_segments)
].copy()


if filtered.empty:
    st.warning("No customers match the selected segments.")
    st.stop()


# --------------------------------------------------
# KPI overview
# --------------------------------------------------
total_customers = filtered["CustomerID"].nunique()
total_revenue = filtered["Monetary"].sum()
avg_recency = filtered["Recency"].mean()
avg_frequency = filtered["Frequency"].mean()
avg_customer_value = filtered["Monetary"].mean()

st.subheader("Customer Value Overview")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Customers", f"{total_customers:,}")
k2.metric("Historical Revenue", f"£{total_revenue:,.2f}")
k3.metric("Average Recency", f"{avg_recency:.1f} days")
k4.metric("Average Orders", f"{avg_frequency:.2f}")


st.caption(
    "RFM uses historical purchasing behaviour. Monetary values represent "
    "historical revenue in the source dataset, displayed in GBP."
)


# --------------------------------------------------
# RFM score distribution
# --------------------------------------------------
st.divider()
st.subheader("RFM Score Analysis")

score_left, score_right = st.columns(2)

with score_left:
    score_data = (
        filtered.groupby("RFMTotalScore", as_index=False)
        .agg(Customers=("CustomerID", "nunique"))
        .sort_values("RFMTotalScore")
    )

    fig_score = px.bar(
        score_data,
        x="RFMTotalScore",
        y="Customers",
        color="Customers",
        color_continuous_scale="Blues",
        template="plotly_dark",
        labels={
            "RFMTotalScore": "Combined RFM Score",
            "Customers": "Number of Customers",
        },
        title="Customers by Combined RFM Score",
    )

    fig_score.update_layout(
        coloraxis_showscale=False,
        height=380,
        margin=dict(l=10, r=45, t=55, b=25),
    )

    st.plotly_chart(fig_score, width="stretch")

with score_right:
    rfm_means = (
        filtered.groupby("Segment", as_index=False)
        .agg(
            Recency=("Recency", "mean"),
            Frequency=("Frequency", "mean"),
            Monetary=("Monetary", "mean"),
        )
    )

    metric_columns = st.columns(3)

    metric_config = [
        ("Recency", "Average Recency (days)", "#F59E0B"),
        ("Frequency", "Average Orders", "#3B82F6"),
        ("Monetary", "Average Revenue (£)", "#22C55E"),
    ]

    for column, (metric, chart_title, chart_color) in zip(
        metric_columns, metric_config
    ):
        with column:
            fig_metric = px.bar(
                rfm_means.sort_values(metric, ascending=True),
                x=metric,
                y="Segment",
                orientation="h",
                text=metric,
                color_discrete_sequence=[chart_color],
                template="plotly_dark",
                title=chart_title,
            )

            fig_metric.update_traces(
                texttemplate=(
                    "%{x:,.1f}" if metric != "Frequency"
                    else "%{x:,.2f}"
                ),
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    + chart_title
                    + ": %{x:,.2f}<extra></extra>"
                ),
            )

            fig_metric.update_layout(
                showlegend=False,
                height=400,
                margin=dict(l=10, r=45, t=55, b=25),
                yaxis=dict(title="", automargin=True, tickfont=dict(size=10),),
                xaxis=dict(title="", separatethousands=True),
            )

            st.plotly_chart(
                fig_metric,
                width="stretch",
                key=f"rfm_metric_{metric.lower()}",
            )


# --------------------------------------------------
# Customer value distribution
# --------------------------------------------------
st.divider()
st.subheader("Customer Monetary Value")

value_left, value_right = st.columns(2)

with value_left:
    fig_value = px.histogram(
        filtered,
        x="Monetary",
        nbins=40,
        color_discrete_sequence=["#6366F1"],
        template="plotly_dark",
        title="Distribution of Historical Customer Revenue",
        labels={
            "Monetary": "Historical Revenue (£)",
            "count": "Number of Customers",
        },
    )

    fig_value.update_layout(
        height=380,
        margin=dict(l=10, r=45, t=55, b=25),
    )

    st.plotly_chart(fig_value, width="stretch")

with value_right:
    fig_aov = px.box(
        filtered,
        x="Segment",
        y="AverageOrderValue",
        color="Segment",
        color_discrete_map=SEGMENT_COLORS,
        points=False,
        template="plotly_dark",
        title="Average Order Value by Segment",
        labels={
            "Segment": "",
            "AverageOrderValue": "Average Order Value (£)",
        },
    )

    fig_aov.update_layout(
        showlegend=False,
        height=380,
        margin=dict(l=10, r=45, t=55, b=25),
    )

    fig_aov.update_xaxes(tickangle=-25)

    st.plotly_chart(fig_aov, width="stretch")


# --------------------------------------------------
# Customer value table
# --------------------------------------------------
st.divider()
st.subheader("Highest Historical Customer Value")

top_n = st.slider(
    "Number of customers to display",
    min_value=5,
    max_value=50,
    value=10,
    step=5,
    key="rfm_top_n",
)

top_customers = (
    filtered.sort_values("Monetary", ascending=False)
    .head(top_n)
    .copy()
)

display_columns = [
    "CustomerID",
    "Segment",
    "Recency",
    "Frequency",
    "Monetary",
    "AverageOrderValue",
    "RFMTotalScore",
]

display_columns = [
    column for column in display_columns if column in top_customers.columns
]

display_table = top_customers[display_columns].rename(
    columns={
        "CustomerID": "Customer ID",
        "Recency": "Recency (days)",
        "Frequency": "Orders",
        "Monetary": "Historical Revenue (£)",
        "AverageOrderValue": "Average Order Value (£)",
        "RFMTotalScore": "Combined RFM Score",
    }
)

if "Historical Revenue (£)" in display_table.columns:
    display_table["Historical Revenue (£)"] = (
        display_table["Historical Revenue (£)"]
        .map(lambda value: f"£{value:,.2f}")
    )

if "Average Order Value (£)" in display_table.columns:
    display_table["Average Order Value (£)"] = (
        display_table["Average Order Value (£)"]
        .map(lambda value: f"£{value:,.2f}")
    )

st.dataframe(
    display_table,
    width="stretch",
    hide_index=True,
)

st.download_button(
    "Download customer value table (CSV)",
    data=top_customers[display_columns].to_csv(index=False).encode("utf-8"),
    file_name="top_customer_value.csv",
    mime="text/csv",
    key="download_rfm_customer_value",
)


# --------------------------------------------------
# Revenue contribution
# --------------------------------------------------
st.divider()
st.subheader("Segment Revenue Contribution")

contribution = (
    filtered.groupby("Segment", as_index=False)
    .agg(
        Revenue=("Monetary", "sum"),
        Customers=("CustomerID", "nunique"),
    )
)

revenue_total = contribution["Revenue"].sum()

contribution["RevenueSharePct"] = (
    contribution["Revenue"] / revenue_total * 100
    if revenue_total > 0
    else 0
)

fig_contribution = px.bar(
    contribution.sort_values("Revenue"),
    x="Revenue",
    y="Segment",
    orientation="h",
    text="RevenueSharePct",
    color="Segment",
    color_discrete_map=SEGMENT_COLORS,
    template="plotly_dark",
    labels={
        "Revenue": "Historical Revenue (£)",
        "Segment": "",
        "RevenueSharePct": "Revenue Share (%)",
    },
    title="Historical Revenue and Share by Segment",
)

fig_contribution.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside",
    cliponaxis=False,
    hovertemplate=(
        "<b>%{y}</b><br>"
        "Revenue: £%{x:,.2f}<br>"
        "Revenue share: %{text:.1f}%"
        "<extra></extra>"
    ),
)

fig_contribution.update_layout(
    showlegend=False,
    height=400,
    margin=dict(l=10, r=90, t=60, b=40),
)

st.plotly_chart(fig_contribution, width="stretch")


# --------------------------------------------------
# Interpretation notes
# --------------------------------------------------
st.divider()
st.subheader("How to Interpret These Metrics")

st.markdown(
    """
    - **Recency:** Days since the customer's last purchase. Lower values
      generally indicate more recent activity.
    - **Frequency:** Number of purchases or orders represented by the
      dataset's customer-level feature.
    - **Monetary:** Historical customer revenue, not a forecast of future spend.
    - **RFM scores:** Relative scoring features derived during preprocessing;
      higher combined scores should be interpreted according to the project's
      scoring convention.
    - **Revenue contribution:** A descriptive share of the selected customers'
      historical revenue. It does not establish the causal effect of a
      marketing campaign.
    """
)

st.caption(
    "This page describes historical behaviour; it is not a churn model or "
    "a forecast of future customer value."
)