# AI Customer Segmentation & Marketing Analytics

An end-to-end data analytics and machine learning project that segments customers based on purchasing behavior using **RFM Analysis, K-Means Clustering, and an interactive Streamlit dashboard**. The project helps analyze customer value, understand purchasing patterns, and generate actionable marketing insights.

## 📌 Project Overview

Customer segmentation helps businesses understand different groups of customers and develop targeted marketing strategies. This project uses transactional retail data to calculate customer-level RFM metrics, identify behavioral patterns, and group customers into meaningful segments using unsupervised machine learning.

The project combines data preprocessing, feature engineering, clustering, model evaluation, and interactive visualization in one analytics workflow.

## ✨ Key Features

- **RFM Analysis:** Analyze customers using Recency, Frequency, and Monetary value.
- **Customer Segmentation:** Apply K-Means clustering to group customers based on behavioral features.
- **Model Evaluation:** Review clustering performance using evaluation metrics and visualizations.
- **Interactive Dashboard:** Explore customer segments and customer-value insights through Streamlit.
- **Data Processing Pipeline:** Organize data cleaning, feature engineering, clustering, and analysis workflows.
- **Visual Analytics:** Explore cluster distributions, segment comparisons, customer value, and other analytical charts.
- **Saved Models:** Store the trained clustering model and feature scaler for reuse.

## 🛠️ Technology Stack

- **Programming:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, K-Means Clustering
- **Visualization:** Matplotlib, Plotly
- **Dashboard:** Streamlit
- **Development:** Jupyter Notebook, Git, GitHub

## 📂 Project Structure

```text
AI-Customer-Segmentation-Marketing-Analytics/
│
├── app.py
├── run_pipeline.py
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── dashboard/
│   ├── overview.py
│   └── pages/
│       ├── 1_Customer_Segmentation.py
│       └── 2_RFM_Customer_Value.py
│
├── data/
│   └── raw/                    # Local source data; not uploaded
│
├── models/
│   ├── customer_segmentation_scaler.joblib
│   ├── kmeans_customer_segmentation.joblib
│   └── segment_mapping.json
│
├── notebooks/
│   └── customer_segmentation_analysis.ipynb
│
├── reports/
│   ├── cluster_distribution.png
│   ├── cluster_evaluation.csv
│   ├── cluster_profiles.csv
│   ├── customer_value_comparison.png
│   ├── pca_customer_segments.png
│   ├── rfm_comparison.png
│   └── segment_summary.csv
│
└── src/
    ├── clustering.py
    ├── dashboard_theme.py
    ├── data_loader.py
    ├── feature_engineering.py
    ├── insights.py
    ├── model_preparation.py
    ├── preprocessing.py
    ├── rfm_analysis.py
    └── visualization.py
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/SommayDewat/AI-Customer-Segmentation-Marketing-Analytics.git
cd AI-Customer-Segmentation-Marketing-Analytics
```

### 2. Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## 📊 Dataset

The raw transaction dataset is not included in this repository. You must obtain the appropriate dataset separately and place it at the file path expected by `src/data_loader.py` before running the data pipeline.

The project uses transaction-level retail data to derive customer-level purchasing features. Check the data-loading and preprocessing scripts for the required columns and input format.

## 🚀 How to Run

### Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to explore the dashboard.

### Run the Data Pipeline

```bash
python run_pipeline.py
```

Ensure the source dataset is available at the expected path before running the pipeline.

## 📈 Analytical Workflow

1. **Data Preparation:** Load and clean retail transaction data.
2. **Feature Engineering:** Calculate customer-level purchasing metrics and RFM features.
3. **Customer Segmentation:** Apply K-Means clustering to group customers.
4. **Model Evaluation:** Examine clustering evaluation metrics and segment profiles.
5. **Visualization:** Generate charts to explore cluster distribution and customer behavior.
6. **Dashboard Exploration:** Present segmentation and customer-value insights interactively.

## 🎯 Business Value

This project demonstrates how customer analytics and unsupervised machine learning can support:

- Customer behavior analysis
- Identification of high-value customer groups
- Data-driven marketing segmentation
- Customer-value comparison
- More informed marketing and retention strategies

These are potential business applications of the analysis, not guaranteed business outcomes.

## 👨‍💻 Author

**Sommay Dewat**

- **GitHub:** [SommayDewat](https://github.com/SommayDewat)
- **LinkedIn:** [Connect on LinkedIn](https://www.linkedin.com/in/sommay-dewat/)

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
