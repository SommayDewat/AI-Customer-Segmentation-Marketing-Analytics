import streamlit as st

st.set_page_config(
    page_title="AI Customer Segmentation & Marketing Analytics",
    page_icon="📊",
    layout="wide",
)


st.markdown(
    """
    <style>
    /* Overall dashboard canvas */
    .stApp {
        background:
            radial-gradient(
                circle at 90% 0%,
                rgba(20, 184, 166, 0.09),
                transparent 32%
            ),
            #0b1120;
        color: #e5edf7;
    }

    /* Main content spacing */
    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Main headings */
    h1, h2, h3 {
        color: #f1f5f9 !important;
        letter-spacing: -0.025em;
    }

    /* KPI cards */
    [data-testid="stMetric"] {
        background: linear-gradient(
            145deg,
            rgba(30, 41, 59, 0.96),
            rgba(15, 23, 42, 0.96)
        );
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.14);
        transition: border-color 0.2s ease,
                    transform 0.2s ease;
    }

    [data-testid="stMetric"]:hover {
        border-color: rgba(45, 212, 191, 0.55);
        transform: translateY(-2px);
    }

    [data-testid="stMetricLabel"] {
        color: #aabbd0 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 700;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #101a2c;
        border-right: 1px solid rgba(148, 163, 184, 0.15);
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid rgba(45, 212, 191, 0.45);
        transition: all 0.2s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: #2dd4bf;
        color: #5eead4;
    }

    /* Expanders and bordered content */
    [data-testid="stExpander"] {
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 12px;
        overflow: hidden;
    }

    /* Tables */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 12px;
        overflow: hidden;
    }

    /* Separators */
    hr {
        border-color: rgba(148, 163, 184, 0.18);
    }

    /* Supporting text */
    [data-testid="stCaptionContainer"] {
        color: #94a3b8;
    }

    /* Dashboard accent colors */
    :root {
        --dashboard-bg: #0B1220;
        --dashboard-surface: #111C2E;
        --dashboard-teal: #2DD4BF;
        --dashboard-blue: #60A5FA;
        --dashboard-amber: #FBBF24;
        --dashboard-slate: #94A3B8;
    }

    /* Multiselect tags: consistent dark slate styling */
    [data-baseweb="tag"] {
        background: #26364A !important;
        color: #F8FAFC !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
    }

    [data-baseweb="tag"] span {
        color: #F8FAFC !important;
    }

    [data-baseweb="tag"] svg {
        fill: #CBD5E1 !important;
        color: #CBD5E1 !important;
    }

    /* Keep focus accents consistent */
    [data-baseweb="select"]:focus-within {
        border-color: #2DD4BF !important;
        box-shadow: 0 0 0 1px #2DD4BF !important;
    }


    /* Slider accent: replace bright red with teal */
    div[data-baseweb="slider"] [role="slider"] {
        background-color: #2DD4BF !important;
        border-color: #2DD4BF !important;
    }

    div[data-baseweb="slider"] [data-testid="stTickBar"] {
        background-color: #2DD4BF !important;
    }

    /* Sidebar input surfaces */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        border-color: #334155;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


overview_page = st.Page(
    "dashboard/overview.py",
    title="Executive Overview",
    icon=":material/dashboard:",
    default=True,
)

segmentation_page = st.Page(
    "dashboard/pages/1_Customer_Segmentation.py",
    title="Customer Segmentation",
    icon=":material/groups:",
)

rfm_page = st.Page(
    "dashboard/pages/2_RFM_Customer_Value.py",
    title="RFM & Customer Value",
    icon=":material/payments:",
)

navigation = st.navigation(
    {
        "Overview": [overview_page],
        "Customer Analytics": [
            segmentation_page,
            rfm_page,
        ],
    }
)

navigation.run()