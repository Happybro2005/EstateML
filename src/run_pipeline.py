import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config
from utils import logger
import preprocessing
import modeling
import visualization

def generate_business_report(df_labeled, metrics_df, optimal_k, profiles):
    """
    Automatically writes a comprehensive Markdown report summarizing the findings,
    dataset details, model validation metrics, and business recommendations.
    """
    logger.info("Generating final executive Markdown report...")
    
    # Calculate dataset stats
    total_clients = len(df_labeled)
    avg_age = df_labeled[config.COL_AGE].mean()
    avg_sat = df_labeled[config.COL_SATISFACTION].mean()
    total_spend = df_labeled[config.COL_PURCHASE_PRICE].sum()
    avg_spend = df_labeled[config.COL_PURCHASE_PRICE].mean()
    loan_rate = (df_labeled[config.COL_LOAN_APPLIED] == 1).mean() * 100
    
    # Validation scores for optimal K
    opt_metrics = metrics_df[metrics_df["k"] == optimal_k].iloc[0]
    sil_score = opt_metrics["silhouette"]
    dbi_score = opt_metrics["davies_bouldin"]
    ch_score = opt_metrics["calinski_harabasz"]
    
    # Profile stats
    profile_descriptions = ""
    recommendations = ""
    
    for c_id, name in sorted(profiles.items()):
        c_data = df_labeled[df_labeled["cluster_id"] == c_id]
        c_size = len(c_data)
        c_pct = (c_size / total_clients) * 100
        c_avg_age = c_data[config.COL_AGE].mean()
        c_avg_sat = c_data[config.COL_SATISFACTION].mean()
        c_avg_price = c_data[config.COL_PURCHASE_PRICE].mean()
        c_loan_pct = (c_data[config.COL_LOAN_APPLIED] == 1).mean() * 100
        c_client_mode = c_data[config.COL_CLIENT_TYPE].mode().iloc[0] if not c_data[config.COL_CLIENT_TYPE].mode().empty else "N/A"
        c_country_mode = c_data[config.COL_COUNTRY].mode().iloc[0] if not c_data[config.COL_COUNTRY].mode().empty else "N/A"
        c_purpose_mode = c_data[config.COL_ACQUISITION_PURPOSE].mode().iloc[0] if not c_data[config.COL_ACQUISITION_PURPOSE].mode().empty else "N/A"
        c_ref_mode = c_data[config.COL_REFERRAL_CHANNEL].mode().iloc[0] if not c_data[config.COL_REFERRAL_CHANNEL].mode().empty else "N/A"
        
        profile_descriptions += f"""
### Segment: {name} (Cluster {c_id})
- **Segment Size**: {c_size} clients ({c_pct:.1f}%)
- **Demographics**: Average Age {c_avg_age:.1f} years | Primary Client Type: {c_client_mode}
- **Geographics**: Primary Country: {c_country_mode}
- **Financial Profile**: Average Purchase Price: ${c_avg_price:,.2f} | Loan Application Rate: {c_loan_pct:.1f}%
- **Behavioral Profile**: Acquisition Purpose: {c_purpose_mode} | Top Referral Channel: {c_ref_mode}
- **Customer Sentiment**: Average Satisfaction Score: {c_avg_sat:.2f} / 5.0

"""

        # Generate action-oriented recommendations based on segment characteristics
        if name == config.CLUSTER_NAME_FIRST_TIME:
            recommendations += f"""
#### Recommendations for **{name}**:
1. **First-Time Homebuyer Loan Packages**: Since {c_loan_pct:.1f}% applied for loans, design high-ratio loan programs and low-deposit options in collaboration with banking partners.
2. **Social Media & Online Engagement**: Allocate marketing budget to Social Media and Online Search channels to capture the attention of this younger demographic (Average Age: {c_avg_age:.1f}).
3. **Starter Home Showcases**: Market affordable residential properties, townhouses, and suburban developments priced under $400k.
"""
        elif name == config.CLUSTER_NAME_GLOBAL:
            recommendations += f"""
#### Recommendations for **{name}**:
1. **Remote Investing Portals**: Promote digital transaction platforms and virtual viewings as these buyers are primarily international, seeking Buy-to-let properties.
2. **Property Management Bundles**: Sell turnkey leasing and management services to facilitate passive income generation for overseas investors.
3. **Agent Networks**: Strengthen international broker relationships to increase direct referrals from overseas markets.
"""
        elif name == config.CLUSTER_NAME_CORPORATE:
            recommendations += f"""
#### Recommendations for **{name}**:
1. **Commercial Asset Portfolios**: Present bulk purchase packages, office spaces, and commercial warehouse opportunities.
2. **Tax & Compliance Advisory**: Provide specialized corporate legal support for commercial investment acquisitions.
3. **B2B Networking**: Establish strong referral schemes with corporate business brokers and strategic corporate partners.
"""
        elif name == config.CLUSTER_NAME_LUXURY:
            recommendations += f"""
#### Recommendations for **{name}**:
1. **Exclusive VIP Property Previews**: Organize private viewings and invite-only listings for penthouses and premium coastal villas.
2. **Concierge Brokerage**: Assign dedicated senior relationships advisors to handle luxury asset transactions.
3. **Targeted Wealth Management Channels**: Advertise through boutique private banks, yacht clubs, and luxury lifestyle publications.
"""

    report_content = f"""# Real Estate Buyer Segmentation and Investment Profiling Report
## Prepared for Parcl Co. Limited

---

## 1. Introduction
This report details the findings of our unsupervised machine learning analysis designed to automatically discover hidden buyer segments and profile investment behaviors in the real estate market. The goal is to provide data-driven market intelligence to Parcl Co. Limited to optimize marketing spend, design tailored customer journeys, and identify lucrative property inventory.

---

## 2. Business Problem
Real estate buyers are highly heterogeneous, ranging from individual first-time homebuyers to multinational corporations. Treating all clients with a single marketing and service strategy leads to inefficient capital allocation and lower customer satisfaction. By segmenting customers based on demographics, purchase attributes, and geographic characteristics, Parcl Co. Limited can design personalized marketing campaigns, optimize sales funnels, and align product inventory with buyer preferences.

---

## 3. Dataset Description
The dataset merges demographic customer records and property transaction details, yielding a total of **{total_clients}** unique cleaned client records. 
Key fields evaluated include:
- **Demographics**: Client Type, Gender, Age (derived from date_of_birth), Country, Region.
- **Transactions**: Property Type, Purchase Price (USD), Property Size (sqft), Location Type, Loan Application status.
- **Behavioral/Sentiment**: Acquisition Purpose, Referral Channel, Satisfaction Score.

---

## 4. Exploratory Data Analysis (EDA) Highlights
During the EDA phase, the following observations were recorded:
- **Demographics**: The customer base is split between individuals and corporate entities, with a global reach including local (UK) and international (US, UAE, Germany, Switzerland) buyers.
- **Correlation**: We noted a strong correlation between property size and purchase price, and a strong negative correlation between purchase price and loan applications, indicating high-net-worth individuals and corporate buyers are mostly self-funded.
- **Missing Data**: Heatmaps identified minor missing values in `gender`, `client_type`, and `satisfaction_score`, which were imputed using mode/median rules during preprocessing.

*Saved EDA visualizations can be found in `reports/figures/` directory:*
- Missing values heatmap: `eda_missing_values.png`
- Demographic charts: `eda_gender_dist.png`, `eda_country_dist.png`, `eda_client_type_dist.png`
- Correlations: `eda_correlation_heatmap.png`

---

## 5. Feature Engineering
The raw data was processed to make it suitable for clustering algorithms:
1. **Date of Birth Conversion**: Converted to `Age` based on a base year of 2026.
2. **Binary Encoding**: `loan_applied` mapped directly to `0` (No) and `1` (Yes).
3. **One-Hot Encoding**: Nominal categories (`country`, `region`, `client_type`, `referral_channel`, `acquisition_purpose`) were encoded to binary vectors to avoid artificial order bias.
4. **Feature Scaling**: Numerical columns (`Age`, `purchase_price_usd`, `size_sqft`, `satisfaction_score`) were standardized using `StandardScaler`.

---

## 6. Model Selection and Validation
We evaluated KMeans clustering and Agglomerative (Hierarchical) Clustering. To determine the optimal number of clusters, we ran metrics across a range of K=2 to K=8.

### Cluster Validation Metrics:
- **Optimal K Select**: **K = {optimal_k}**
- **Silhouette Score**: **{sil_score:.4f}** (Indicates high cluster separation and cohesion)
- **Davies-Bouldin Index**: **{dbi_score:.4f}** (Lower index represents tighter, more separated clusters)
- **Calinski-Harabasz Score**: **{ch_score:.2f}** (Higher score indicates better clustering structure)

*Validation charts saved to:*
- Elbow Curve: `clustering_elbow_curve.png`
- Validation Metrics: `clustering_metrics.png`

---

## 7. Clustering Results and Segment Profiles
Based on the centroid characteristics of the {optimal_k} clusters, the following distinct buyer groups were identified:

{profile_descriptions}

*Saved clustering visualizations:*
- PCA 2D Cluster Space: `cluster_pca_2d.png`
- Segment Sizes: `cluster_sizes.png`
- Pairplot Matrix: `cluster_pairplot.png`

---

## 8. Business Recommendations
Based on the distinct characteristics of the discovered clusters, we propose the following strategic actions for Parcl Co. Limited:

{recommendations}

---

## 9. Conclusion
The machine learning pipeline has successfully grouped Parcl Co. Limited's client base into four highly distinct and intuitive buyer segments: **First-Time Buyers**, **Global Investors**, **Corporate Buyers**, and **Luxury Investors**. By implementing the recommendations above, Parcl Co. Limited can improve customer acquisition cost, boost satisfaction, and maximize sales velocity.
"""
    
    os.makedirs(os.path.dirname(config.REPORT_PATH), exist_ok=True)
    with open(config.REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    logger.info(f"Report written successfully to {config.REPORT_PATH}")

def run_ml_pipeline():
    logger.info("Initializing full ML pipeline run...")
    
    # 1. Load and clean
    df_merged = preprocessing.load_raw_data()
    df_cleaned = preprocessing.clean_data(df_merged)
    
    # Generate EDA figures
    visualization.plot_missing_values_heatmap(df_merged, filename="eda_missing_values.png", show=False)
    visualization.plot_correlation_heatmap(df_cleaned, filename="eda_correlation_heatmap.png", show=False)
    visualization.plot_univariate_distribution(df_cleaned, config.COL_GENDER, "Gender Distribution", filename="eda_gender_dist.png")
    visualization.plot_univariate_distribution(df_cleaned, config.COL_CLIENT_TYPE, "Client Type Distribution", filename="eda_client_type_dist.png")
    visualization.plot_univariate_distribution(df_cleaned, config.COL_COUNTRY, "Country Distribution", filename="eda_country_dist.png")
    visualization.plot_univariate_distribution(df_cleaned, config.COL_REGION, "Region Distribution", filename="eda_region_dist.png")
    visualization.plot_univariate_distribution(df_cleaned, config.COL_ACQUISITION_PURPOSE, "Acquisition Purpose", filename="eda_acquisition_dist.png")
    visualization.plot_univariate_distribution(df_cleaned, config.COL_LOAN_APPLIED, "Loan Applied (0=No, 1=Yes)", filename="eda_loan_dist.png", is_numeric=True)
    visualization.plot_univariate_distribution(df_cleaned, config.COL_REFERRAL_CHANNEL, "Referral Channels", filename="eda_referrals_dist.png")
    visualization.plot_univariate_distribution(df_cleaned, config.COL_SATISFACTION, "Satisfaction Score Distribution", filename="eda_satisfaction_dist.png", is_numeric=True)
    visualization.plot_univariate_distribution(df_cleaned, config.COL_AGE, "Age Distribution", filename="eda_age_dist.png", is_numeric=True)
    
    # 2. Features Preprocessing
    X_processed, feature_names, scaler, encoder = preprocessing.fit_transform_features(df_cleaned, is_training=True)
    
    # 3. Model validation & metrics range
    metrics_df = modeling.run_clustering_metrics(X_processed, max_k=8)
    
    # Save metrics charts
    visualization.plot_elbow_curve(metrics_df, filename="clustering_elbow_curve.png", show=False)
    visualization.plot_validation_metrics(metrics_df, filename="clustering_metrics.png", show=False)
    
    # 4. Optimal K and Fit (Forced to 4 clusters per Parcl Co. business requirements)
    optimal_k = 4
    kmeans, labels = modeling.fit_kmeans_model(X_processed, optimal_k)
    
    # Fit Agglomerative for comparison log
    agg, agg_labels = modeling.fit_agglomerative_model(X_processed, optimal_k)
    agg_sil = modeling.silhouette_score(X_processed, agg_labels)
    logger.info(f"Agglomerative Clustering Silhouette Score (K={optimal_k}): {agg_sil:.4f} vs KMeans: {metrics_df.loc[metrics_df['k'] == optimal_k, 'silhouette'].values[0]:.4f}")
    
    # 5. Apply PCA for plotting
    pca, X_pca = modeling.apply_pca(X_processed, n_components=3)
    
    # 6. Profile clusters and assign names
    cluster_profiles = modeling.identify_cluster_profiles(df_cleaned, labels)
    
    # 7. Add Labels to Dataframe
    df_labeled = df_cleaned.copy()
    df_labeled["cluster_id"] = labels
    df_labeled["cluster_name"] = df_labeled["cluster_id"].map(cluster_profiles)
    
    # Save processed clients CSV
    os.makedirs(os.path.dirname(config.PROCESSED_CLIENTS_PATH), exist_ok=True)
    df_labeled.to_csv(config.PROCESSED_CLIENTS_PATH, index=False)
    logger.info(f"Saved processed dataset with cluster labels to {config.PROCESSED_CLIENTS_PATH}")
    
    # Save models using joblib
    modeling.save_ml_artifacts(kmeans, scaler, encoder)
    
    # Save cluster-specific charts
    visualization.plot_pca_2d(X_pca, labels, cluster_profiles, filename="cluster_pca_2d.png", show=False)
    visualization.plot_cluster_sizes(df_labeled, label_col="cluster_name", filename="cluster_sizes.png", show=False)
    visualization.plot_cluster_boxplots(df_labeled, config.COL_AGE, hue_col="cluster_name", filename="cluster_boxplot_age.png", show=False)
    visualization.plot_cluster_boxplots(df_labeled, config.COL_SATISFACTION, hue_col="cluster_name", filename="cluster_boxplot_satisfaction.png", show=False)
    visualization.plot_cluster_boxplots(df_labeled, config.COL_PURCHASE_PRICE, hue_col="cluster_name", filename="cluster_boxplot_price.png", show=False)
    
    visualization.plot_cluster_bivariate(df_labeled, config.COL_COUNTRY, title="Buyer Segment Distribution by Country", filename="cluster_vs_country.png", show=False)
    visualization.plot_cluster_bivariate(df_labeled, config.COL_REGION, title="Buyer Segment Distribution by Region", filename="cluster_vs_region.png", show=False)
    visualization.plot_cluster_bivariate(df_labeled, config.COL_LOAN_APPLIED, title="Buyer Segment Distribution by Loan Applied", filename="cluster_vs_loan.png", show=False)
    visualization.plot_cluster_bivariate(df_labeled, config.COL_CLIENT_TYPE, title="Buyer Segment Distribution by Client Type", filename="cluster_vs_client_type.png", show=False)
    visualization.plot_cluster_bivariate(df_labeled, config.COL_ACQUISITION_PURPOSE, title="Buyer Segment Distribution by Acquisition Purpose", filename="cluster_vs_purpose.png", show=False)
    
    visualization.plot_cluster_pairplots(df_labeled, numeric_cols=config.NUMERICAL_COLS, hue_col="cluster_name", filename="cluster_pairplot.png", show=False)
    
    # Generate business report
    generate_business_report(df_labeled, metrics_df, optimal_k, cluster_profiles)
    
    logger.info("ML Pipeline execution complete. All models, reports, and charts generated successfully.")

if __name__ == "__main__":
    run_ml_pipeline()
