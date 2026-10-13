from src.dashboard_theme import SEGMENT_COLORS, SEGMENT_ORDER
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_DIR / "reports"
CUSTOMER_FILE = REPORTS_DIR / "customer_segments.csv"

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide",
)

st.title("Customer Segmentation")
st.caption(
    "Explore customer groups, purchasing behaviour, "
    "and historical revenue contribution."
)


# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_customer_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Customer report not found: {path}"
        )

    data = pd.read_csv(path)

    required_columns = {
        "CustomerID",
        "Segment",
        "Monetary",
        "Recency",
        "Frequency",
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    for column in ["Monetary", "Recency", "Frequency"]:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data["CustomerID"] = data["CustomerID"].astype("Int64").astype(str)

    return data


try:
    customers = load_customer_data(CUSTOMER_FILE)
except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
    st.error(str(error))
    st.stop()


# --------------------------------------------------
# Filters
# --------------------------------------------------
st.sidebar.header("Segmentation Filters")

segments = sorted(customers["Segment"].dropna().unique().tolist())

selected_segments = st.sidebar.multiselect(
    "Select segments",
    options=segments,
    default=segments,
)

filtered = customers[
    customers["Segment"].isin(selected_segments)
].copy()


# --------------------------------------------------
# KPIs
# --------------------------------------------------
kpi1, kpi2, kpi3 = st.columns(3)

kpi1.metric("Customers", f"{filtered['CustomerID'].nunique():,}")
kpi2.metric("Historical Revenue", f"£{filtered['Monetary'].sum():,.2f}")

avg_frequency = (
    filtered["Frequency"].mean()
    if not filtered.empty
    else 0
)

kpi3.metric("Average Orders per Customer", f"{avg_frequency:.2f}")


# --------------------------------------------------
# Segment charts
# --------------------------------------------------
st.divider()
st.subheader("Segment Comparison")

if filtered.empty:
    st.info("Select at least one segment to view the analysis.")
else:
    profile = (
        filtered.groupby("Segment", as_index=False)
        .agg(
            Customers=("CustomerID", "nunique"),
            Revenue=("Monetary", "sum"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
        )
    )

    left, right = st.columns(2)

    with left:
        st.markdown("**Customers by Segment**")

        fig_customers = px.bar(
            profile.sort_values("Customers"),
            x="Customers",
            y="Segment",
            orientation="h",
            text="Customers",
            template="plotly_dark",
            labels={"Customers": "Customers", "Segment": ""},
        )

        fig_customers.update_traces(
            texttemplate="%{x:,.0f}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Customers: %{x:,.0f}"
                "<extra></extra>"
            ),
        )

        fig_customers.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            height=400,
            margin=dict(l=10, r=90, t=25, b=35),
            xaxis=dict(
                title="Customers",
                separatethousands=True,
            ),
            yaxis=dict(
                title="",
                categoryorder="total ascending",
            ),
        )

        st.plotly_chart(fig_customers, width="stretch")

    with right:
        st.markdown("**Revenue by Segment**")

        fig_revenue = px.bar(
            profile.sort_values("Revenue"),
            x="Revenue",
            y="Segment",
            orientation="h",
            text="Revenue",
            template="plotly_dark",
            labels={"Revenue": "Historical Revenue (£)", "Segment": ""},
        )

        fig_revenue.update_traces(
            texttemplate="£%{x:,.0f}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Historical revenue: £%{x:,.2f}"
                "<extra></extra>"
            ),
        )

        fig_revenue.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            height=400,
            margin=dict(l=10, r=125, t=25, b=35),
            xaxis=dict(
                title="Historical Revenue (£)",
                separatethousands=True,
            ),
            yaxis=dict(
                title="",
                categoryorder="total ascending",
            ),
        )

        st.plotly_chart(fig_revenue, width="stretch")

    st.markdown("**Segment Profile**")

    display_profile = profile.rename(
        columns={
            "AvgRecency": "Average Recency (days)",
            "AvgFrequency": "Average Orders",
        }
    )

    display_profile["Revenue"] = display_profile["Revenue"].map(
        lambda value: f"£{value:,.2f}"
    )

    display_profile["Average Recency (days)"] = (
        display_profile["Average Recency (days)"].round(1)
    )

    display_profile["Average Orders"] = (
        display_profile["Average Orders"].round(2)
    )

    st.dataframe(
        display_profile,
        width="stretch",
        hide_index=True,
    )


st.caption(
    "Segments are based on historical purchasing behaviour. "
    "The 'At-Risk' label is a business interpretation, not a validated "
    "churn prediction."
)
