# Real Estate Buyer Segmentation & Investment Profiling
### Unsupervised Machine Learning Solution for Parcl Co. Limited

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=Happybro2005/EstateML&branch=main&mainModule=app.py)

This project implements an end-to-end unsupervised machine learning pipeline to identify hidden buyer segments in the real estate market. Using client transactional and demographic history, the system groups customers into distinct investment cohorts (First-Time Buyers, Global Investors, Corporate Buyers, Luxury Investors) using clustering algorithms. A premium multi-page Streamlit dashboard allows brokers and executives to interactively explore findings and profile new leads.

---

## 🚀 Key Features

1. **Robust Cleaning & Preprocessing**: Resolves duplication, imputes missing values, engineers age vectors, and applies customized nominal/binary/ordinal encoders.
2. **Automated Validation Suite**: Evaluates cluster separation using Elbow Inertia, Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score.
3. **Dimensionality Reduction**: Projects features to 2D and 3D spaces using PCA for interactive web rendering.
4. **Centroid-Based Profiling**: Profiles cluster centroids dynamically based on features before applying labels.
5. **Interactive Dashboard**: Feature-rich 9-page Streamlit dashboard with custom glassmorphic styling, Plotly charts, and an automated report exporter.
6. **Assign New Buyer (Lead Router)**: Input lead parameters to instantly route them to an existing buyer cohort and retrieve recommended marketing campaigns.

---

## 📁 Folder Structure

```
buyer_segmentation/
│
├── data/
│   ├── clients.csv              # Raw demographic data
│   ├── properties.csv           # Raw property transaction details
│   └── processed_clients.csv    # Final labeled data with cluster assignments
│
├── notebooks/
│   └── buyer_segmentation.ipynb # Interactive Jupyter analysis notebook
│
├── src/
│   ├── config.py                # Global configuration and constants
│   ├── utils.py                 # Logging configurations and helpers
│   ├── preprocessing.py         # Data cleaning, parsing, and pipeline encoders
│   ├── modeling.py              # Clustering models, metrics, PCA, and profiling
│   ├── visualization.py         # Matplotlib, Seaborn, and Plotly charts
│   └── run_pipeline.py          # Orchestrates training, validation, and reporting
│
├── models/
│   ├── kmeans_model.pkl         # Trained KMeans model
│   ├── scaler.pkl               # Standardized feature scaler
│   └── encoder.pkl              # Fitted categorical One-Hot encoder
│
├── reports/
│   ├── report.md                # Automated executive market report
│   ├── pipeline.log             # Standardized operational pipeline logs
│   └── figures/                 # Saved static reports charts
│
├── app.py                       # Multi-page Streamlit application
├── requirements.txt             # Project dependencies list
├── README.md                    # Project overview and installation guide
└── tests/
    └── test_pipeline.py         # Automated unit testing suite
```

---

## 🛠️ Installation

### 1. Prerequisites
Ensure you have Python 3.8+ installed (tested on Python 3.14.3).

### 2. Clone and Setup
Open your terminal in the `buyer_segmentation` directory and install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline and Generate Models
Execute the pipeline script to generate the synthetic datasets, evaluate clustering, fit the model, and save the reports:
```bash
python src/run_pipeline.py
```

### 4. Run Unit Tests
To verify pipeline correctness and encoding shapes:
```bash
python -m unittest tests/test_pipeline.py
```

---

## 💻 How to Run the Dashboard

To start the interactive market intelligence dashboard, run:
```bash
streamlit run app.py
```
Open the local URL displayed in the terminal (usually `http://localhost:8501`).

---

## 📊 Dashboard Pages

- **🏠 Home**: Executive overview, KPI cards (Total volume, Avg age, Avg satisfaction).
- **📊 EDA**: Interactive feature distribution viewer and correlation matrices.
- **🤖 Buyer Segmentation**: Silhouette/Davies-Bouldin metrics, and 2D/3D PCA cluster scatter plots.
- **🌍 Geographic Analysis**: Country origin counts and local regional target trends.
- **💰 Investment Profiling**: Visualizing loan usage, property sizes, and price structures.
- **📈 Cluster Insights**: Centroid stats and specific marketing/sales suggestions.
- **🔍 Assign New Buyer**: Dynamic calculator forms to classify new leads.
- **📄 Download Report**: Preview and download the generated executive `report.md`.
- **ℹ Model Information**: Explains the technical model hyperparameters and metrics.

---

## 🔮 Future Improvements

- **Supervised Re-fitting**: Re-train classification models (e.g., Random Forests) on the labeled clusters to fast-track new lead classification.
- **Dynamic SQL Connection**: Link dataset loading to an active CRM database.
- **Natural Language Query**: Add a LLM-based query interface in Streamlit to retrieve insights via text.
