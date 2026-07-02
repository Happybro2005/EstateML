import os
import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import config
from utils import logger

def run_clustering_metrics(X_processed, max_k=8):
    """
    Computes clustering quality metrics (Inertia, Silhouette, Davies-Bouldin, Calinski-Harabasz)
    for K values ranging from 2 to max_k.
    """
    logger.info(f"Evaluating clustering metrics for K = 2 to {max_k}...")
    metrics = []
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X_processed)
        
        inertia = kmeans.inertia_
        sil = silhouette_score(X_processed, labels)
        dbi = davies_bouldin_score(X_processed, labels)
        ch = calinski_harabasz_score(X_processed, labels)
        
        metrics.append({
            "k": k,
            "inertia": inertia,
            "silhouette": sil,
            "davies_bouldin": dbi,
            "calinski_harabasz": ch
        })
        logger.info(f"K={k}: Inertia={inertia:.2f}, Silhouette={sil:.4f}, DBI={dbi:.4f}, CH={ch:.2f}")
        
    metrics_df = pd.DataFrame(metrics)
    return metrics_df

def determine_optimal_k(metrics_df):
    """
    Automatically recommends the best K based on the maximum Silhouette score.
    """
    # Max silhouette score indicates the highest separation and cohesion
    optimal_idx = metrics_df["silhouette"].idxmax()
    optimal_k = int(metrics_df.loc[optimal_idx, "k"])
    logger.info(f"Programmatically selected optimal K = {optimal_k} based on Silhouette Score.")
    return optimal_k

def fit_kmeans_model(X_processed, k):
    """
    Fits a final KMeans model with the selected K.
    """
    logger.info(f"Fitting final KMeans model with K={k}...")
    kmeans = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=15)
    labels = kmeans.fit_predict(X_processed)
    return kmeans, labels

def fit_agglomerative_model(X_processed, k):
    """
    Fits Agglomerative Hierarchical Clustering for comparison.
    """
    logger.info(f"Fitting Agglomerative Clustering with K={k}...")
    agg = AgglomerativeClustering(n_clusters=k)
    labels = agg.fit_predict(X_processed)
    return agg, labels

def apply_pca(X_processed, n_components=3):
    """
    Applies Principal Component Analysis (PCA) to reduce dimensionality for plotting.
    """
    logger.info(f"Applying PCA to reduce dimensions to {n_components} components...")
    pca = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
    X_pca = pca.fit_transform(X_processed)
    
    # Calculate explained variance
    explained_var = pca.explained_variance_ratio_
    logger.info(f"PCA explained variance ratio: {explained_var} (Total: {sum(explained_var):.4f})")
    
    return pca, X_pca

def identify_cluster_profiles(df_cleaned, labels):
    """
    Profiles each cluster and dynamically assigns names based on characteristics.
    """
    logger.info("Analyzing cluster characteristics and mapping names...")
    
    df_temp = df_cleaned.copy()
    df_temp["cluster_id"] = labels
    
    # Generate profile names based on cluster centroids / average values
    cluster_profiles = {}
    
    # Group by cluster and get averages/mode characteristics
    for cluster_id in df_temp["cluster_id"].unique():
        cluster_data = df_temp[df_temp["cluster_id"] == cluster_id]
        
        avg_price = cluster_data[config.COL_PURCHASE_PRICE].mean()
        avg_age = cluster_data[config.COL_AGE].mean()
        avg_size = cluster_data[config.COL_SIZE].mean()
        loan_pct = (cluster_data[config.COL_LOAN_APPLIED] == 1).mean()
        
        # Check client type mode
        client_type_mode = cluster_data[config.COL_CLIENT_TYPE].mode().iloc[0] if not cluster_data[config.COL_CLIENT_TYPE].mode().empty else "Individual"
        
        # Check country distributions
        country_mode = cluster_data[config.COL_COUNTRY].mode().iloc[0] if not cluster_data[config.COL_COUNTRY].mode().empty else "United Kingdom"
        
        logger.info(f"Cluster {cluster_id} characteristics: Avg Price=${avg_price:,.2f}, Avg Age={avg_age:.1f}, Avg Size={avg_size:.1f} sqft, Loan Applied={loan_pct*100:.1f}%, Client Type={client_type_mode}, Country={country_mode}")
        
        # Label Assignment logic based on centroids of the real dataset
        avg_sat = cluster_data[config.COL_SATISFACTION].mean()
        if avg_age < 40:
            cluster_name = config.CLUSTER_NAME_FIRST_TIME
        elif avg_price > 400000:
            cluster_name = config.CLUSTER_NAME_LUXURY
        elif avg_sat > 3.5:
            cluster_name = config.CLUSTER_NAME_GLOBAL
        else:
            cluster_name = config.CLUSTER_NAME_CORPORATE
            
        cluster_profiles[cluster_id] = cluster_name
        logger.info(f"Assigned name '{cluster_name}' to Cluster {cluster_id}")
        
    return cluster_profiles

def save_ml_artifacts(kmeans_model, scaler, encoder, output_dir=config.MODELS_DIR):
    """
    Saves models and pipelines to files.
    """
    logger.info(f"Saving models to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(kmeans_model, config.KMEANS_MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    joblib.dump(encoder, config.ENCODER_PATH)
    
    logger.info("Saved all artifacts successfully.")
