import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import config
from utils import logger

# Set matplotlib style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 50})

def save_or_show_plot(fig, filename=None, show=True):
    """Utility to either save the plot to the reports directory or show it."""
    if filename:
        filepath = os.path.join(config.FIGURES_DIR, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {filepath}")
    if show:
        plt.show()
    else:
        plt.close(fig)

# --- EDA PLOTS ---

def plot_missing_values_heatmap(df, filename="eda_missing_values.png", show=False):
    """Generates and saves a heatmap of missing values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis", ax=ax)
    ax.set_title("Heatmap of Missing Values in Raw Dataset", fontsize=14, fontweight="bold")
    save_or_show_plot(fig, filename, show)

def plot_univariate_distribution(df, column, title, filename=None, show=False, is_numeric=False):
    """Plots count distribution for categorical, or histogram/KDE for numeric."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if is_numeric:
        sns.histplot(df[column].dropna(), kde=True, ax=ax, color="#1f77b4")
        ax.set_ylabel("Frequency")
    else:
        # Categorical countplot
        order = df[column].value_counts().index
        sns.countplot(data=df, x=column, order=order, ax=ax, palette="viridis")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(column.replace("_", " ").title())
    save_or_show_plot(fig, filename, show)

def plot_correlation_heatmap(df, filename="eda_correlation_heatmap.png", show=False):
    """Generates correlation heatmap for numeric features."""
    fig, ax = plt.subplots(figsize=(8, 6))
    numeric_df = df[config.NUMERICAL_COLS + [config.COL_LOAN_APPLIED]].select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlation Heatmap (Numeric Features)", fontsize=14, fontweight="bold")
    save_or_show_plot(fig, filename, show)

# --- CLUSTERING VALIDATION PLOTS ---

def plot_elbow_curve(metrics_df, filename="clustering_elbow_curve.png", show=False):
    """Plots K vs Inertia (Elbow Method)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(metrics_df["k"], metrics_df["inertia"], marker="o", linestyle="-", color="purple", linewidth=2)
    ax.set_title("Elbow Method (Inertia)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    ax.set_xticks(metrics_df["k"])
    save_or_show_plot(fig, filename, show)

def plot_validation_metrics(metrics_df, filename="clustering_metrics.png", show=False):
    """Plots Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Silhouette
    axes[0].plot(metrics_df["k"], metrics_df["silhouette"], marker="s", color="green", linewidth=2)
    axes[0].set_title("Silhouette Score (Higher is Better)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Score")
    axes[0].set_xticks(metrics_df["k"])
    
    # 2. Davies-Bouldin
    axes[1].plot(metrics_df["k"], metrics_df["davies_bouldin"], marker="^", color="red", linewidth=2)
    axes[1].set_title("Davies-Bouldin Index (Lower is Better)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Index")
    axes[1].set_xticks(metrics_df["k"])
    
    # 3. Calinski-Harabasz
    axes[2].plot(metrics_df["k"], metrics_df["calinski_harabasz"], marker="o", color="blue", linewidth=2)
    axes[2].set_title("Calinski-Harabasz Score (Higher is Better)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("Score")
    axes[2].set_xticks(metrics_df["k"])
    
    plt.suptitle("Cluster Validation Metrics Suite", fontsize=15, fontweight="bold", y=1.02)
    save_or_show_plot(fig, filename, show)

# --- CLUSTER ANALYSIS & PCA PLOTS ---

def plot_pca_2d(X_pca, labels, cluster_names=None, filename="cluster_pca_2d.png", show=False):
    """Generates 2D PCA scatter plot."""
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # Map numeric labels to names if provided
    hue_data = [cluster_names[l] if cluster_names else f"Cluster {l}" for l in labels]
    
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1], hue=hue_data,
        palette="Set1", style=hue_data, s=60, alpha=0.8, ax=ax
    )
    ax.set_title("PCA 2D Cluster Visualization", fontsize=14, fontweight="bold")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(title="Segments", bbox_to_anchor=(1.05, 1), loc="upper left")
    save_or_show_plot(fig, filename, show)

def plot_pca_3d_plotly(X_pca, labels, cluster_names=None):
    """Generates interactive 3D PCA scatter plot using Plotly."""
    hue_data = [cluster_names[l] if cluster_names else f"Cluster {l}" for l in labels]
    
    df_pca = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "PC3": X_pca[:, 2],
        "Segment": hue_data
    })
    
    fig = px.scatter_3d(
        df_pca, x="PC1", y="PC2", z="PC3",
        color="Segment", title="PCA 3D Cluster Visualization",
        color_discrete_sequence=px.colors.qualitative.Set1,
        opacity=0.8
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    return fig

def plot_cluster_sizes(df_labeled, label_col="cluster_name", filename="cluster_sizes.png", show=False):
    """Plots pie and bar chart of cluster sizes."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    counts = df_labeled[label_col].value_counts()
    
    # Pie chart
    axes[0].pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=140, colors=sns.color_palette("Set1"))
    axes[0].set_title("Buyer Segment Distribution (Share)", fontsize=13, fontweight="bold")
    
    # Bar chart
    sns.barplot(x=counts.index, y=counts.values, ax=axes[1], palette="Set1")
    axes[1].set_title("Buyer Segment Count", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Number of Clients")
    axes[1].set_xlabel("Segments")
    plt.xticks(rotation=15)
    
    save_or_show_plot(fig, filename, show)

def plot_cluster_bivariate(df, x_col, hue_col="cluster_name", title="", filename=None, show=False):
    """Generates crosstab visual comparison between a category and the clusters."""
    fig, ax = plt.subplots(figsize=(10, 6))
    # Relative proportion within each cluster
    cross_tab_prop = pd.crosstab(index=df[hue_col], columns=df[x_col], normalize="index") * 100
    
    cross_tab_prop.plot(kind="bar", stacked=True, colormap="viridis", ax=ax, edgecolor="black")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Segments")
    ax.legend(title=x_col.replace("_", " ").title(), bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=15)
    
    save_or_show_plot(fig, filename, show)

def plot_cluster_boxplots(df, numeric_col, hue_col="cluster_name", filename=None, show=False):
    """Plots boxplot of numeric features vs cluster."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x=hue_col, y=numeric_col, palette="Set1", ax=ax)
    ax.set_title(f"{numeric_col.replace('_', ' ').title()} distribution by Buyer Segment", fontsize=13, fontweight="bold")
    ax.set_xlabel("Segments")
    ax.set_ylabel(numeric_col.replace("_", " ").title())
    plt.xticks(rotation=15)
    
    save_or_show_plot(fig, filename, show)

def plot_cluster_pairplots(df, numeric_cols=config.NUMERICAL_COLS, hue_col="cluster_name", filename="cluster_pairplot.png", show=False):
    """Generates pairplot matrix grouped by clusters."""
    cols_to_plot = numeric_cols + [hue_col]
    plt.figure(figsize=(12, 10))
    g = sns.pairplot(df[cols_to_plot], hue=hue_col, palette="Set1", diag_kind="kde", plot_kws={'alpha': 0.6, 's': 40})
    g.fig.suptitle("Pairwise Relationships of Numeric Features by Buyer Segment", y=1.02, fontsize=16, fontweight="bold")
    
    if filename:
        filepath = os.path.join(config.FIGURES_DIR, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        g.savefig(filepath, dpi=200, bbox_inches="tight")
        logger.info(f"Saved pairplot to {filepath}")
        
    if show:
        plt.show()
    else:
        plt.close()
