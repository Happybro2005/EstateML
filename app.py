import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Add src to sys.path so we can import config/utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
import config
from utils import logger

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Parcl Co. | Real Estate Market Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Brand (Rendered at top)
st.sidebar.markdown("""
    <div style='text-align: center; padding-top: 1rem;'>
        <h2 style='font-family: Outfit; font-weight: 800; color: #0072FF; margin-bottom: 0px;'>Parcl Co.</h2>
        <p style='color: #8A9AAD; font-size: 0.8rem; text-transform: uppercase;'>Market Intelligence</p>
    </div>
    <hr style='margin-top: 0.5rem; margin-bottom: 1rem;'/>
""", unsafe_allow_html=True)

# Sidebar Theme Toggle Switch
dark_mode = st.sidebar.toggle("Dark Mode Active", value=True)
st.sidebar.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 1rem;'/>", unsafe_allow_html=True)

# Custom Premium Styling (Conditional Theme Injection)
if dark_mode:
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
        
        /* Make entire page body, app canvas, and sidebar consistently dark */
        html, body, [class*="css"], .stApp, [data-testid="stSidebar"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: #0F172A !important;
            color: #F8FAFC !important;
        }
        
        /* Style headers and markdown text for high contrast */
        .stMarkdown, p, li, span, label, div {
            color: #E2E8F0 !important;
        }
        h1, h2, h3, h4, h5, h6, [data-testid="stHeader"] {
            color: #FFFFFF !important;
        }
        
        .main-title {
            font-family: 'Outfit', sans-serif;
            font-weight: 800;
            font-size: 2.8rem;
            background: linear-gradient(135deg, #00C6FF, #0072FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            font-size: 1.2rem;
            color: #94A3B8;
            margin-bottom: 2rem;
        }
        
        .section-header {
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            font-size: 1.8rem;
            color: #F1F5F9;
            border-bottom: 2px solid #334155;
            padding-bottom: 0.4rem;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        
        .kpi-container {
            display: flex;
            gap: 1.5rem;
            margin-bottom: 2rem;
            flex-wrap: wrap;
        }
        
        .kpi-card {
            background: rgba(30, 41, 59, 0.75);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1.5rem;
            flex: 1;
            min-width: 220px;
            box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.3);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .kpi-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 16px 50px 0 rgba(0, 72, 255, 0.2);
            border-color: #0072FF;
        }
        
        .kpi-val {
            font-size: 2rem;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 0.2rem;
        }
        
        .kpi-label {
            font-size: 0.9rem;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .rec-box {
            background: linear-gradient(135deg, rgba(0, 114, 255, 0.1), rgba(0, 198, 255, 0.1));
            border-left: 5px solid #0072FF;
            border-radius: 8px;
            padding: 1.5rem;
            margin-top: 1rem;
            margin-bottom: 1.5rem;
        }
        
        .rec-title {
            font-weight: 700;
            color: #00C6FF;
            margin-bottom: 0.5rem;
        }
        
        /* Ensure Streamlit form / selectors look clean and legible on dark backgrounds */
        div[data-baseweb="select"] > div {
            background-color: #1E293B !important;
            color: #F8FAFC !important;
            border-color: #334155 !important;
        }
        input, button, select, textarea {
            background-color: #1E293B !important;
            color: #F8FAFC !important;
        }
        
        </style>
    """, unsafe_allow_html=True)
    plotly_template = "plotly_dark"
else:
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
        
        /* Make entire page body, app canvas, and sidebar consistently light */
        html, body, [class*="css"], .stApp, [data-testid="stSidebar"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: #F8FAFC !important;
            color: #0F172A !important;
        }
        
        /* Style headers and markdown text for high contrast */
        .stMarkdown, p, li, span, label, div {
            color: #334155 !important;
        }
        h1, h2, h3, h4, h5, h6, [data-testid="stHeader"] {
            color: #0F172A !important;
        }
        
        .main-title {
            font-family: 'Outfit', sans-serif;
            font-weight: 800;
            font-size: 2.8rem;
            background: linear-gradient(135deg, #00C6FF, #0072FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            font-size: 1.2rem;
            color: #64748B;
            margin-bottom: 2rem;
        }
        
        .section-header {
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            font-size: 1.8rem;
            color: #1E293B;
            border-bottom: 2px solid #E2E8F0;
            padding-bottom: 0.4rem;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        
        .kpi-container {
            display: flex;
            gap: 1.5rem;
            margin-bottom: 2rem;
            flex-wrap: wrap;
        }
        
        .kpi-card {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(226, 232, 240, 0.8);
            border-radius: 16px;
            padding: 1.5rem;
            flex: 1;
            min-width: 220px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.04);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .kpi-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.08);
            border-color: #0072FF;
        }
        
        .kpi-val {
            font-size: 2rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 0.2rem;
        }
        
        .kpi-label {
            font-size: 0.9rem;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .rec-box {
            background: linear-gradient(135deg, rgba(0, 114, 255, 0.05), rgba(0, 198, 255, 0.05));
            border-left: 5px solid #0072FF;
            border-radius: 8px;
            padding: 1.5rem;
            margin-top: 1rem;
            margin-bottom: 1.5rem;
        }
        
        .rec-title {
            font-weight: 700;
            color: #0072FF;
            margin-bottom: 0.5rem;
        }
        
        </style>
    """, unsafe_allow_html=True)
    plotly_template = "plotly_white"

# Apply Plotly Template globally
import plotly.io as pio
pio.templates.default = plotly_template

# Helper function to load cache-safe resources
@st.cache_data
def load_processed_data():
    if os.path.exists(config.PROCESSED_CLIENTS_PATH):
        df = pd.read_csv(config.PROCESSED_CLIENTS_PATH)
        return df
    return None

@st.cache_resource
def load_ml_models():
    kmeans = None
    scaler = None
    encoder = None
    if os.path.exists(config.KMEANS_MODEL_PATH):
        kmeans = joblib.load(config.KMEANS_MODEL_PATH)
    if os.path.exists(config.SCALER_PATH):
        scaler = joblib.load(config.SCALER_PATH)
    if os.path.exists(config.ENCODER_PATH):
        encoder = joblib.load(config.ENCODER_PATH)
    return kmeans, scaler, encoder

df_data = load_processed_data()
kmeans, scaler, encoder = load_ml_models()

# Brand rendered at top of sidebar

# Navigation Radio
page = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home",
        "📊 EDA",
        "🤖 Buyer Segmentation",
        "🌍 Geographic Analysis",
        "💰 Investment Profiling",
        "📈 Cluster Insights",
        "🔍 Assign New Buyer",
        "📄 Download Report",
        "ℹ Model Information"
    ]
)

# ----------------- SIDEBAR FILTERS -----------------
# We apply filters only to data-exploration pages
apply_filters = page in ["🏠 Home", "📊 EDA", "🌍 Geographic Analysis", "💰 Investment Profiling", "📈 Cluster Insights"]
df_filtered = df_data

if df_data is not None and apply_filters:
    st.sidebar.markdown("### Global Dashboard Filters")
    
    # 1. Country Filter
    countries = ["All"] + sorted(df_data[config.COL_COUNTRY].dropna().unique().tolist())
    selected_country = st.sidebar.selectbox("Country", countries)
    
    # 2. Region Filter
    regions = ["All"] + sorted(df_data[config.COL_REGION].dropna().unique().tolist())
    selected_region = st.sidebar.selectbox("Region", regions)
    
    # 3. Client Type Filter
    client_types = ["All"] + sorted(df_data[config.COL_CLIENT_TYPE].dropna().unique().tolist())
    selected_client_type = st.sidebar.selectbox("Client Type", client_types)
    
    # 4. Acquisition Purpose Filter
    purposes = ["All"] + sorted(df_data[config.COL_ACQUISITION_PURPOSE].dropna().unique().tolist())
    selected_purpose = st.sidebar.selectbox("Acquisition Purpose", purposes)
    
    # 5. Loan Applied Filter
    loans = ["All", "Yes", "No"]
    selected_loan = st.sidebar.selectbox("Loan Applied", loans)
    
    # Apply filter logic
    if selected_country != "All":
        df_filtered = df_filtered[df_filtered[config.COL_COUNTRY] == selected_country]
    if selected_region != "All":
        df_filtered = df_filtered[df_filtered[config.COL_REGION] == selected_region]
    if selected_client_type != "All":
        df_filtered = df_filtered[df_filtered[config.COL_CLIENT_TYPE] == selected_client_type]
    if selected_purpose != "All":
        df_filtered = df_filtered[df_filtered[config.COL_ACQUISITION_PURPOSE] == selected_purpose]
    if selected_loan != "All":
        loan_binary = 1 if selected_loan == "Yes" else 0
        df_filtered = df_filtered[df_filtered[config.COL_LOAN_APPLIED] == loan_binary]

# ----------------- HOME PAGE -----------------
if page == "🏠 Home":
    st.markdown("<h1 class='main-title'>Real Estate Buyer Segmentation</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence</p>", unsafe_allow_html=True)
    
    if df_filtered is not None:
        # Display Glassmorphic KPI Cards
        total_c = len(df_filtered)
        avg_age_val = df_filtered[config.COL_AGE].mean()
        avg_sat_val = df_filtered[config.COL_SATISFACTION].mean()
        total_spend_val = df_filtered[config.COL_PURCHASE_PRICE].sum()
        
        st.markdown(f"""
            <div class='kpi-container'>
                <div class='kpi-card'>
                    <div class='kpi-val'>{total_c:,}</div>
                    <div class='kpi-label'>Total Active Clients</div>
                </div>
                <div class='kpi-card'>
                    <div class='kpi-val'>{avg_age_val:.1f} yrs</div>
                    <div class='kpi-label'>Average Buyer Age</div>
                </div>
                <div class='kpi-card'>
                    <div class='kpi-val'>{avg_sat_val:.2f} / 5.0</div>
                    <div class='kpi-label'>Avg Satisfaction Score</div>
                </div>
                <div class='kpi-card'>
                    <div class='kpi-val'>${total_spend_val/1e6:.1f}M</div>
                    <div class='kpi-label'>Total Transaction Volume</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h3 class='section-header'>Project Executive Summary</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.write("""
                **Parcl Co. Limited** is a modern property services firm that utilizes cutting-edge unsupervised 
                machine learning algorithms to segment real estate buyers based on structural transaction matrices. 
                Instead of simple manual heuristics, this platform automatically groups customers into clusters 
                that share buying intents, geographic preferences, demographic footprints, and investment sizes.
                
                This dashboard helps the management:
                - Identify hidden buying intents (e.g. First-time home acquisition, commercial developer expansion, high-value asset shelters).
                - Target marketing campaign channels appropriately (VIP portals, commercial broker partnerships, online ads).
                - Allocate budget toward local and international lead sources.
            """)
        with col2:
            st.image("https://images.unsplash.com/photo-1560518883-ce09059eeffa?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80", caption="Real Estate Portfolio Management", use_column_width=True)
            
        st.markdown("<h3 class='section-header'>Real Estate Buyer Cohorts</h3>", unsafe_allow_html=True)
        cohort_cols = st.columns(4)
        cohort_details = [
            ("First-Time Buyers", "Younger clients buying starter apartments or townhouses with high loan rates, referred online."),
            ("Global Investors", "Overseas individual and corporate buyers seeking rental apartments/villas for buy-to-let income, fully self-funded."),
            ("Corporate Buyers", "Corporate entities acquiring high-value commercial properties or institutional portfolios, mixed funding."),
            ("Luxury Investors", "High-Net-Worth individuals acquiring premium villas or penthouses in prime/coastal areas, self-funded.")
        ]
        for i, (cohort, desc) in enumerate(cohort_details):
            with cohort_cols[i]:
                st.markdown(f"""
                    <div style='background: #F8FAFC; border: 1px solid #E2E8F0; padding: 1.2rem; border-radius: 12px; height: 100%;'>
                        <h4 style='color: #0F172A; font-family: Outfit; font-weight: 700; margin-bottom: 0.5rem;'>{cohort}</h4>
                        <p style='color: #475569; font-size: 0.9rem;'>{desc}</p>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Data not found. Please run the machine learning pipeline first to generate `processed_clients.csv`.")

# ----------------- EDA PAGE -----------------
elif page == "📊 EDA":
    st.markdown("<h1 class='main-title'>Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    
    if df_filtered is not None:
        st.write("Explore raw demographic and transaction distributions of Parcl Co. clients.")
        
        tab1, tab2 = st.tabs(["📊 Feature Distributions", "🔗 Numeric Correlation Heatmap"])
        
        with tab1:
            col_select = st.selectbox(
                "Select Feature to Visualize:",
                [
                    "gender", "client_type", "country", "region", 
                    "acquisition_purpose", "loan_applied", "referral_channel", 
                    "satisfaction_score", "age", "purchase_price_usd", "size_sqft"
                ]
            )
            
            fig = None
            obs = ""
            
            if col_select in ["age", "purchase_price_usd", "size_sqft"]:
                fig = px.histogram(df_filtered, x=col_select, kde=True, title=f"Distribution of {col_select.replace('_', ' ').title()}", color_discrete_sequence=["#0072FF"])
                if col_select == "age":
                    obs = "The age distribution shows a multi-modal pattern, reflecting distinct age groups for first-time buyers (twenties/early thirties) and seasoned or luxury investors (forties and above)."
                elif col_select == "purchase_price_usd":
                    obs = "Transaction values span from $120k up to $5M. The density is heavily right-skewed, showing a high concentration of starter properties, alongside a small volume of premium, high-ticket transactions."
                elif col_select == "size_sqft":
                    obs = "Property sizes correspond to price structures, ranging from small 500 sqft apartments up to massive 8,000 sqft commercial/luxury spaces."
            else:
                counts = df_filtered[col_select].value_counts().reset_index()
                counts.columns = [col_select, "Count"]
                fig = px.bar(counts, x=col_select, y="Count", color=col_select, title=f"Distribution of {col_select.replace('_', ' ').title()}", color_discrete_sequence=px.colors.qualitative.Set1)
                
                if col_select == "gender":
                    obs = "The client gender column is fairly balanced across individuals, with a dedicated 'Corporate Entity' segment representing institutional transactions."
                elif col_select == "client_type":
                    obs = "Individual buyers dominate the count, while Corporate clients make up a smaller but highly capitalized portion of the customer base."
                elif col_select == "country":
                    obs = "Domestic UK buyers are the largest customer group, but overseas buyers (Germany, USA, UAE, Switzerland) represent a crucial cross-border segment."
                elif col_select == "region":
                    obs = "Geographic regions reveal local concentrations (London Central, North West, Wales) vs International Hubs targeted by global funds."
                elif col_select == "acquisition_purpose":
                    obs = "Residential home occupancy is the most common reason for purchase, followed by buy-to-let rental portfolio development and luxury vacations."
                elif col_select == "loan_applied":
                    obs = "A substantial portion of buyers utilize mortgage financing (1 = Yes), while high-end buyers and international investors purchase using cash (0 = No)."
                elif col_select == "referral_channel":
                    obs = "Online searches and social media are the primary source for residential individual buyers, while VIP networks and agent channels bring in luxury and international clients."
                elif col_select == "satisfaction_score":
                    obs = "The satisfaction distribution shows high average ratings (4.0+), indicating strong overall buyer happiness, with occasional lower scores among corporate/value buyers."
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f"<div class='rec-box'><div class='rec-title'>EDA Observation:</div><p style='margin-bottom:0;'>{obs}</p></div>", unsafe_allow_html=True)
                
        with tab2:
            st.write("Pearson correlation matrix between numeric columns and loan applied status.")
            num_cols = config.NUMERICAL_COLS + [config.COL_LOAN_APPLIED]
            corr = df_filtered[num_cols].corr()
            
            fig_heat = px.imshow(
                corr, text_auto=".2f",
                labels=dict(color="Correlation"),
                x=corr.columns, y=corr.index,
                color_continuous_scale="RdBu", color_continuous_midpoint=0
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            st.markdown("""
                **Key Correlation Insights:**
                - **Purchase Price vs Property Size**: Strong positive correlation (+0.95+), confirming property dimensions dictate transaction ticket value.
                - **Purchase Price vs Loan Applied**: Moderately negative correlation, indicating higher-priced transactions are more frequently funded without external mortgages (cash buyers/investors).
                - **Age vs Purchase Price**: Positive correlation, suggesting older buyers command larger financial resources to acquire bigger property portfolios.
            """)
    else:
        st.warning("Data not loaded. Run pipeline first.")

# ----------------- BUYER SEGMENTATION PAGE -----------------
elif page == "🤖 Buyer Segmentation":
    st.markdown("<h1 class='main-title'>Buyer Segmentation Models</h1>", unsafe_allow_html=True)
    
    if df_data is not None:
        st.write("Dimensionality reduction and cluster visual mappings using Principal Component Analysis (PCA).")
        
        tab_pca, tab_metrics = st.tabs(["🌌 PCA Space Visualizer", "📈 Model Selection Metrics"])
        
        with tab_pca:
            # Let's project PCA components
            # We can calculate PCA dynamically or reload from features.
            # To keep app fast, we run a quick PCA on the data
            kmeans_model, scaler_obj, encoder_obj = load_ml_models()
            
            if kmeans_model is not None:
                # Separate numerical and categorical
                numeric_cols = config.NUMERICAL_COLS
                nominal_cols = config.NOMINAL_COLS
                binary_cols = [config.COL_LOAN_APPLIED]
                
                X_num = scaler_obj.transform(df_data[numeric_cols])
                X_cat = encoder_obj.transform(df_data[nominal_cols])
                X_bin = df_data[binary_cols].values
                X_processed = np.hstack((X_num, X_cat, X_bin))
                
                # Apply 3-component PCA
                from sklearn.decomposition import PCA
                pca = PCA(n_components=3, random_state=config.RANDOM_STATE)
                X_pca = pca.fit_transform(X_processed)
                
                df_pca = pd.DataFrame({
                    "PC1": X_pca[:, 0],
                    "PC2": X_pca[:, 1],
                    "PC3": X_pca[:, 2],
                    "Segment": df_data["cluster_name"]
                })
                
                col_left, col_right = st.columns([3, 1])
                with col_left:
                    view_3d = st.checkbox("Toggle 3D PCA Space View", value=True)
                    if view_3d:
                        fig_pca = px.scatter_3d(
                            df_pca, x="PC1", y="PC2", z="PC3", color="Segment",
                            color_discrete_sequence=px.colors.qualitative.Set1,
                            title="Interactive 3D PCA Cluster Space Projection",
                            opacity=0.75, height=700
                        )
                        fig_pca.update_layout(margin=dict(l=0, r=0, b=0, t=50))
                    else:
                        fig_pca = px.scatter(
                            df_pca, x="PC1", y="PC2", color="Segment",
                            color_discrete_sequence=px.colors.qualitative.Set1,
                            title="Interactive 2D PCA Cluster Space Projection",
                            opacity=0.8, height=600
                        )
                    st.plotly_chart(fig_pca, use_container_width=True)
                    
                with col_right:
                    st.markdown("#### PCA Details:")
                    st.write(f"**Explained Variance Ratio:**")
                    for i, ratio in enumerate(pca.explained_variance_ratio_):
                        st.write(f"- PC{i+1}: **{ratio*100:.1f}%**")
                    st.write(f"**Total Captured Variance:** **{sum(pca.explained_variance_ratio_)*100:.1f}%**")
                    
                    st.markdown("#### Cluster Shares:")
                    sizes = df_data["cluster_name"].value_counts().reset_index()
                    sizes.columns = ["Segment", "Count"]
                    fig_sizes = px.pie(sizes, values="Count", names="Segment", color="Segment", color_discrete_sequence=px.colors.qualitative.Set1, height=280)
                    fig_sizes.update_layout(showlegend=False, margin=dict(l=0, r=0, b=0, t=0))
                    st.plotly_chart(fig_sizes, use_container_width=True)
            else:
                st.error("Saved KMeans model not found under models/")
                
        with tab_metrics:
            st.write("Clustering algorithm evaluation curves for selecting optimal cluster size (K).")
            # Display pre-saved figures for metrics or render metrics if possible.
            # We will show the metrics curves.
            
            metrics_file = os.path.join(config.FIGURES_DIR, "clustering_metrics.png")
            elbow_file = os.path.join(config.FIGURES_DIR, "clustering_elbow_curve.png")
            
            col_el, col_me = st.columns(2)
            with col_el:
                st.markdown("#### Elbow Method (Inertia)")
                if os.path.exists(elbow_file):
                    st.image(elbow_file, use_column_width=True)
                else:
                    st.info("Run `run_pipeline.py` to generate model comparison metrics.")
            with col_me:
                st.markdown("#### Validation Metrics (Silhouette, DB, CH)")
                if os.path.exists(metrics_file):
                    st.image(metrics_file, use_column_width=True)
                else:
                    st.info("Run `run_pipeline.py` to generate model comparison metrics.")
    else:
        st.warning("Data not loaded. Run pipeline first.")

# ----------------- GEOGRAPHIC ANALYSIS PAGE -----------------
elif page == "🌍 Geographic Analysis":
    st.markdown("<h1 class='main-title'>Geographic Market Analysis</h1>", unsafe_allow_html=True)
    
    if df_filtered is not None:
        st.write("Understand buyer geographic footfall across country origins and local target regions.")
        
        tab_geo1, tab_geo2 = st.tabs(["🌐 Country Origins", "📍 Target Regional Distributions"])
        
        with tab_geo1:
            country_sum = df_filtered.groupby([config.COL_COUNTRY, "cluster_name"]).size().reset_index(name="Count")
            fig_country = px.bar(
                country_sum, x=config.COL_COUNTRY, y="Count", color="cluster_name",
                title="Buyer Segments grouped by Country of Origin",
                color_discrete_sequence=px.colors.qualitative.Set1,
                barmode="stack"
            )
            st.plotly_chart(fig_country, use_container_width=True)
            
            # Country Average purchase price
            avg_price_c = df_filtered.groupby(config.COL_COUNTRY)[config.COL_PURCHASE_PRICE].mean().reset_index()
            fig_price_c = px.bar(
                avg_price_c, x=config.COL_COUNTRY, y=config.COL_PURCHASE_PRICE,
                title="Average Property Investment Size by Country (USD)",
                color_discrete_sequence=["#2E8B57"]
            )
            st.plotly_chart(fig_price_c, use_container_width=True)
            
        with tab_geo2:
            region_sum = df_filtered.groupby([config.COL_REGION, "cluster_name"]).size().reset_index(name="Count")
            fig_region = px.bar(
                region_sum, x=config.COL_REGION, y="Count", color="cluster_name",
                title="Buyer Segments grouped by Target Real Estate Region",
                color_discrete_sequence=px.colors.qualitative.Set1,
                barmode="group"
            )
            st.plotly_chart(fig_region, use_container_width=True)
    else:
        st.warning("Data not loaded. Run pipeline first.")

# ----------------- INVESTMENT PROFILING PAGE -----------------
elif page == "💰 Investment Profiling":
    st.markdown("<h1 class='main-title'>Investment and Financing Profiling</h1>", unsafe_allow_html=True)
    
    if df_filtered is not None:
        st.write("Evaluate financing methods, buying purpose, property specifications, and spend structures per segment.")
        
        col1, col2 = st.columns(2)
        with col1:
            # Price Boxplot by Segment
            fig_price = px.box(
                df_filtered, x="cluster_name", y=config.COL_PURCHASE_PRICE,
                color="cluster_name", color_discrete_sequence=px.colors.qualitative.Set1,
                title="Property Acquisition Price by Buyer Segment"
            )
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Property size distribution by Segment
            fig_size = px.box(
                df_filtered, x="cluster_name", y=config.COL_SIZE,
                color="cluster_name", color_discrete_sequence=px.colors.qualitative.Set1,
                title="Property Dimensions (Size in Sqft) by Buyer Segment"
            )
            st.plotly_chart(fig_size, use_container_width=True)
            
        with col2:
            # Loan Applied by segment
            loan_counts = df_filtered.groupby(["cluster_name", config.COL_LOAN_APPLIED]).size().reset_index(name="Count")
            loan_counts["Loan Applied"] = loan_counts[config.COL_LOAN_APPLIED].map({1: "Yes", 0: "No"})
            fig_loan = px.bar(
                loan_counts, x="cluster_name", y="Count", color="Loan Applied",
                color_discrete_sequence=["#FF7F0E", "#1F77B4"],
                title="Mortgage Loan Applications by Buyer Segment",
                barmode="stack"
            )
            st.plotly_chart(fig_loan, use_container_width=True)
            
            # Acquisition purpose by Segment
            purpose_counts = df_filtered.groupby(["cluster_name", config.COL_ACQUISITION_PURPOSE]).size().reset_index(name="Count")
            fig_purpose = px.bar(
                purpose_counts, x="cluster_name", y="Count", color=config.COL_ACQUISITION_PURPOSE,
                title="Acquisition Purpose distribution by Buyer Segment",
                barmode="stack"
            )
            st.plotly_chart(fig_purpose, use_container_width=True)
    else:
        st.warning("Data not loaded. Run pipeline first.")

# ----------------- CLUSTER INSIGHTS PAGE -----------------
elif page == "📈 Cluster Insights":
    st.markdown("<h1 class='main-title'>Buyer Segment Profiles & Insights</h1>", unsafe_allow_html=True)
    
    if df_filtered is not None:
        st.write("Understand average features, cluster center characteristics, and business recommendations.")
        
        # 1. Descriptive stats table
        st.markdown("### Segment Descriptive Centroids")
        
        # Group by and aggregate
        cluster_summary = df_filtered.groupby("cluster_name").agg({
            config.COL_AGE: "mean",
            config.COL_PURCHASE_PRICE: "mean",
            config.COL_SIZE: "mean",
            config.COL_SATISFACTION: "mean",
            config.COL_LOAN_APPLIED: lambda x: f"{(x == 1).mean()*100:.1f}%"
        }).rename(columns={
            config.COL_AGE: "Average Age",
            config.COL_PURCHASE_PRICE: "Average Price ($)",
            config.COL_SIZE: "Average Size (sqft)",
            config.COL_SATISFACTION: "Average Satisfaction",
            config.COL_LOAN_APPLIED: "Loan Applied %"
        })
        
        # Format price
        cluster_summary["Average Price ($)"] = cluster_summary["Average Price ($)"].apply(lambda x: f"${x:,.2f}")
        cluster_summary["Average Age"] = cluster_summary["Average Age"].apply(lambda x: f"{x:.1f} yrs")
        cluster_summary["Average Satisfaction"] = cluster_summary["Average Satisfaction"].apply(lambda x: f"{x:.2f} / 5.0")
        
        st.dataframe(cluster_summary, use_container_width=True)
        
        # Select Segment for recommendations
        st.markdown("### Actionable Business Recommendations")
        selected_seg = st.selectbox("Select Buyer Segment to Analyze:", sorted(df_filtered["cluster_name"].unique().tolist()))
        
        # Display recommendations
        if selected_seg == "First-Time Buyers":
            st.markdown(f"""
                <div class='rec-box'>
                    <div class='rec-title'>Targeting Strategy for **First-Time Buyers**:</div>
                    <ul>
                        <li><b>Financing Partnerships</b>: Over 85% of this segment requires mortgage options. Partner with retail banks to launch pre-approved low-interest home loans.</li>
                        <li><b>Social Channels</b>: Target Facebook, Instagram, and Google Search campaigns for starter homes ($150k - $350k).</li>
                        <li><b>Value Proposition</b>: Focus advertising copies on "affordable mortgage alternatives to rent", "low-maintenance living", and "energy efficiency".</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
        elif selected_seg == "Global Investors":
            st.markdown(f"""
                <div class='rec-box'>
                    <div class='rec-title'>Targeting Strategy for **Global Investors**:</div>
                    <ul>
                        <li><b>Passive Yield Campaigns</b>: Focus advertising on net rental yields, tenant occupancy rates, and localized buy-to-let regulations.</li>
                        <li><b>Turnkey Property Management</b>: Bundle property management and leasing services to assure cross-border buyers of hassle-free rental returns.</li>
                        <li><b>International Broker Nets</b>: Establish strategic commission schemes with overseas brokers in Germany, USA, UAE, and China.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
        elif selected_seg == "Corporate Buyers":
            st.markdown(f"""
                <div class='rec-box'>
                    <div class='rec-title'>Targeting Strategy for **Corporate Buyers**:</div>
                    <ul>
                        <li><b>Commercial Portfolios</b>: Proactively offer multi-family portfolios, retail outlets, and commercial warehouse units.</li>
                        <li><b>Tax & Corporate Advising</b>: Offer standard corporate acquisition advisory, capital gains structuring help, and commercial leasing contract guidance.</li>
                        <li><b>Direct Sales Network</b>: Cultivate relationships directly with investment fund managers, corporate trust officers, and commercial developers.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
        elif selected_seg == "Luxury Investors":
            st.markdown(f"""
                <div class='rec-box'>
                    <div class='rec-title'>Targeting Strategy for **Luxury Investors**:</div>
                    <ul>
                        <li><b>Invite-Only Previews</b>: Launch private portfolio viewing events for ultra-high-end properties (Villas, Penthouses over $1.8M).</li>
                        <li><b>Premium Relationship Management</b>: Task senior brokerage directors to personally handle customer relationships and search mandates.</li>
                        <li><b>VIP Referrals</b>: Invest in VIP client referral incentives to leverage high-society networking pools.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
        # Segment Satisfaction Boxplot
        fig_sat = px.box(
            df_filtered, x="cluster_name", y=config.COL_SATISFACTION,
            color="cluster_name", color_discrete_sequence=px.colors.qualitative.Set1,
            title="Customer Satisfaction Score by Buyer Segment"
        )
        st.plotly_chart(fig_sat, use_container_width=True)
    else:
        st.warning("Data not loaded. Run pipeline first.")

# ----------------- ASSIGN NEW BUYER PAGE -----------------
elif page == "🔍 Assign New Buyer":
    st.markdown("<h1 class='main-title'>Assign New Buyer to Existing Cluster</h1>", unsafe_allow_html=True)
    
    if kmeans is not None and scaler is not None and encoder is not None:
        st.write("Input characteristics of a new real estate lead to dynamically assign them to one of our mapped buyer segments.")
        
        # Load unique options from dataset dynamically
        client_type_opts = sorted(df_data[config.COL_CLIENT_TYPE].dropna().unique().tolist()) if df_data is not None else ["Individual", "Company"]
        gender_opts = sorted(df_data[config.COL_GENDER].dropna().unique().tolist()) if df_data is not None else ["F", "M"]
        country_opts = sorted(df_data[config.COL_COUNTRY].dropna().unique().tolist()) if df_data is not None else ["USA", "UK"]
        region_opts = sorted(df_data[config.COL_REGION].dropna().unique().tolist()) if df_data is not None else ["California", "England"]
        purpose_opts = sorted(df_data[config.COL_ACQUISITION_PURPOSE].dropna().unique().tolist()) if df_data is not None else ["Home", "Investment"]
        referral_opts = sorted(df_data[config.COL_REFERRAL_CHANNEL].dropna().unique().tolist()) if df_data is not None else ["Website", "Agency"]
        property_type_opts = sorted(df_data[config.COL_PROPERTY_TYPE].dropna().unique().tolist()) if df_data is not None else ["Apartment", "Office"]
        
        with st.form("new_buyer_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.slider("Age of Buyer", min_value=18, max_value=100, value=45)
                client_type = st.selectbox("Client Type", client_type_opts)
                gender = st.selectbox("Gender", gender_opts)
                loan_applied = st.selectbox("Loan Applied", ["Yes", "No"])
                
            with col2:
                purchase_price = st.number_input("Target Purchase Price (USD)", min_value=10000, max_value=10000000, value=300000, step=10000)
                size_sqft = st.number_input("Target Property Size (Sqft)", min_value=100, max_value=20000, value=1100, step=50)
                country = st.selectbox("Country of Origin", country_opts)
                region = st.selectbox("Target Region", region_opts)
                
            with col3:
                acquisition_purpose = st.selectbox("Acquisition Purpose", purpose_opts)
                referral_channel = st.selectbox("Referral Channel", referral_opts)
                property_type = st.selectbox("Property Type (Unit Category)", property_type_opts)
                satisfaction = st.slider("Expected Satisfaction Score (or Initial Rating)", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
            
            submit_btn = st.form_submit_form_button = st.form_submit_button("Compute Assigned Segment")
            
        if submit_btn:
            # 1. Format input DataFrame matching config specifications
            loan_val = 1 if loan_applied == "Yes" else 0
            
            input_df = pd.DataFrame([{
                config.COL_CLIENT_TYPE: client_type,
                config.COL_GENDER: gender,
                config.COL_COUNTRY: country,
                config.COL_REGION: region,
                config.COL_REFERRAL_CHANNEL: referral_channel,
                config.COL_ACQUISITION_PURPOSE: acquisition_purpose,
                config.COL_LOAN_APPLIED: loan_val,
                config.COL_AGE: float(age),
                config.COL_PURCHASE_PRICE: float(purchase_price),
                config.COL_SIZE: float(size_sqft),
                config.COL_SATISFACTION: float(satisfaction),
                config.COL_PROPERTY_TYPE: property_type
            }])
            
            # Apply preprocessing
            try:
                X_num = scaler.transform(input_df[config.NUMERICAL_COLS])
                X_cat = encoder.transform(input_df[config.NOMINAL_COLS])
                X_bin = input_df[[config.COL_LOAN_APPLIED]].values
                X_processed = np.hstack((X_num, X_cat, X_bin))
                
                # Predict
                predicted_cluster = kmeans.predict(X_processed)[0]
                
                # Load profile mapping dynamically
                if df_data is not None:
                    # Retrieve the cluster names from existing dataset
                    mapping = dict(df_data.groupby("cluster_id")["cluster_name"].first())
                    assigned_name = mapping.get(predicted_cluster, f"Cluster {predicted_cluster}")
                else:
                    assigned_name = f"Cluster {predicted_cluster}"
                    
                st.markdown("### Mapped Assignment Results")
                st.balloons()
                
                # Color code result
                bg_colors = {
                    "First-Time Buyers": "#E1EFFE",
                    "Global Investors": "#DEF7EC",
                    "Corporate Buyers": "#FDE8E8",
                    "Luxury Investors": "#FEF08A"
                }
                text_colors = {
                    "First-Time Buyers": "#1E40AF",
                    "Global Investors": "#03543F",
                    "Corporate Buyers": "#9B1C1C",
                    "Luxury Investors": "#713F12"
                }
                
                col_res1, col_res2 = st.columns([1, 2])
                with col_res1:
                    st.markdown(f"""
                        <div style='background-color: {bg_colors.get(assigned_name, "#F1F5F9")}; 
                                    color: {text_colors.get(assigned_name, "#1E293B")}; 
                                    padding: 2rem; border-radius: 12px; border: 2px solid; text-align: center;'>
                            <h4 style='margin-bottom: 0.2rem; text-transform: uppercase; font-size: 0.8rem; letter-spacing: 0.1em;'>Lead Assigned To</h4>
                            <h2 style='margin: 0px; font-family: Outfit; font-weight: 800;'>{assigned_name}</h2>
                            <p style='margin-top: 0.5rem; font-size: 0.9rem;'>Cluster ID: {predicted_cluster}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                with col_res2:
                    st.markdown("#### Recommended Campaign Theme:")
                    if assigned_name == "First-Time Buyers":
                        st.write("Offer **'Mortgage Guarantee Campaign'** or **'Starter Space Showcases'**. The buyer matches profile: young demographic with high probability of borrowing.")
                    elif assigned_name == "Global Investors":
                        st.write("Offer **'High Yield Rentals Portfolio'** and **'Managed Passive Income Pack'**. The buyer matches profile: cross-border buyer looking for stable residential buy-to-let options.")
                    elif assigned_name == "Corporate Buyers":
                        st.write("Offer **'Commercial Core Assets'** or **'Bulk Office/Warehouse portfolios'**. The buyer matches profile: Corporate buyer seeking yield or expansion space.")
                    elif assigned_name == "Luxury Investors":
                        st.write("Offer **'Elite Villa Portfolio'** and **'Off-market Penthouse Preview'**. The buyer matches profile: high-ticket value, cash-only buyer seeking premium real estate assets.")
            except Exception as e:
                st.error(f"Error preprocessing input for predictions: {e}")
    else:
        st.warning("Saved model components not found. Run pipeline first.")

# ----------------- DOWNLOAD REPORT PAGE -----------------
elif page == "📄 Download Report":
    st.markdown("<h1 class='main-title'>Download Market Segment Report</h1>", unsafe_allow_html=True)
    
    if os.path.exists(config.REPORT_PATH):
        st.write("Preview of the generated Real Estate Buyer Segmentation executive report:")
        
        with open(config.REPORT_PATH, "r", encoding="utf-8") as f:
            report_md = f.read()
            
        st.download_button(
            label="Download Markdown Executive Report",
            data=report_md,
            file_name="real_estate_buyer_segmentation_report.md",
            mime="text/markdown"
        )
        
        st.markdown("---")
        st.markdown(report_md)
    else:
        st.warning("The executive report file `report.md` has not been generated yet. Please run `run_pipeline.py` first.")

# ----------------- MODEL INFORMATION PAGE -----------------
elif page == "ℹ Model Information":
    st.markdown("<h1 class='main-title'>Model and Algorithm Architecture</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### KMeans Clustering Details")
        st.write("""
            **Algorithm:** KMeans Clustering is a centroid-based unsupervised partition algorithm. It groups 
            objects in $K$ clusters such that the within-cluster sum of squares (inertia) is minimized.
            
            **Scaler Used:** `StandardScaler` from Scikit-learn. Standardizes features by removing the mean 
            and scaling to unit variance. Essential for KMeans as it calculates Euclidean distances.
            
            **Encoder Used:** `OneHotEncoder(handle_unknown='ignore')`. Converts nominal values into binary arrays, 
            preventing algorithms from assuming numerical ordinal relationships where none exist.
            
            **Dimensionality Reduction:** Principal Component Analysis (PCA). Reduces feature size to 3 main components 
            capturing maximum variance, enabling direct, high-fidelity human visualization.
        """)
        
    with col2:
        st.markdown("### Selected Validation Scores")
        kmeans_model, scaler_obj, encoder_obj = load_ml_models()
        if kmeans_model is not None:
            st.write(f"- Fitted Clusters (Optimal K): **{kmeans_model.n_clusters}**")
            st.write(f"- Random State Seed: **{config.RANDOM_STATE}**")
            st.write("- Feature Matrix columns count: **Scaled numeric columns + One-hot categories + Loan status**")
        else:
            st.write("Model components not currently loaded. Run modeling pipeline.")
            
        metrics_file = os.path.join(config.FIGURES_DIR, "clustering_metrics.png")
        if os.path.exists(metrics_file):
            st.image(metrics_file, caption="Cluster Evaluation Metrics Suite", use_column_width=True)
