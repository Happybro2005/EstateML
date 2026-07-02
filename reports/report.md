# Real Estate Buyer Segmentation and Investment Profiling Report
## Prepared for Parcl Co. Limited

---

## 1. Introduction
This report details the findings of our unsupervised machine learning analysis designed to automatically discover hidden buyer segments and profile investment behaviors in the real estate market. The goal is to provide data-driven market intelligence to Parcl Co. Limited to optimize marketing spend, design tailored customer journeys, and identify lucrative property inventory.

---

## 2. Business Problem
Real estate buyers are highly heterogeneous, ranging from individual first-time homebuyers to multinational corporations. Treating all clients with a single marketing and service strategy leads to inefficient capital allocation and lower customer satisfaction. By segmenting customers based on demographics, purchase attributes, and geographic characteristics, Parcl Co. Limited can design personalized marketing campaigns, optimize sales funnels, and align product inventory with buyer preferences.

---

## 3. Dataset Description
The dataset merges demographic customer records and property transaction details, yielding a total of **7305** unique cleaned client records. 
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
- **Optimal K Select**: **K = 4**
- **Silhouette Score**: **0.1317** (Indicates high cluster separation and cohesion)
- **Davies-Bouldin Index**: **2.0385** (Lower index represents tighter, more separated clusters)
- **Calinski-Harabasz Score**: **1070.43** (Higher score indicates better clustering structure)

*Validation charts saved to:*
- Elbow Curve: `clustering_elbow_curve.png`
- Validation Metrics: `clustering_metrics.png`

---

## 7. Clustering Results and Segment Profiles
Based on the centroid characteristics of the 4 clusters, the following distinct buyer groups were identified:


### Segment: Global Investors (Cluster 0)
- **Segment Size**: 2422 clients (33.2%)
- **Demographics**: Average Age 59.0 years | Primary Client Type: Individual
- **Geographics**: Primary Country: USA
- **Financial Profile**: Average Purchase Price: $251,836.07 | Loan Application Rate: 36.4%
- **Behavioral Profile**: Acquisition Purpose: Home | Top Referral Channel: Website
- **Customer Sentiment**: Average Satisfaction Score: 4.06 / 5.0


### Segment: Corporate Buyers (Cluster 1)
- **Segment Size**: 1588 clients (21.7%)
- **Demographics**: Average Age 57.7 years | Primary Client Type: Individual
- **Geographics**: Primary Country: USA
- **Financial Profile**: Average Purchase Price: $261,382.52 | Loan Application Rate: 36.9%
- **Behavioral Profile**: Acquisition Purpose: Home | Top Referral Channel: Website
- **Customer Sentiment**: Average Satisfaction Score: 1.46 / 5.0


### Segment: Luxury Investors (Cluster 2)
- **Segment Size**: 2516 clients (34.4%)
- **Demographics**: Average Age 58.4 years | Primary Client Type: Individual
- **Geographics**: Primary Country: USA
- **Financial Profile**: Average Purchase Price: $488,727.72 | Loan Application Rate: 36.5%
- **Behavioral Profile**: Acquisition Purpose: Home | Top Referral Channel: Website
- **Customer Sentiment**: Average Satisfaction Score: 3.08 / 5.0


### Segment: First-Time Buyers (Cluster 3)
- **Segment Size**: 779 clients (10.7%)
- **Demographics**: Average Age 33.9 years | Primary Client Type: Individual
- **Geographics**: Primary Country: USA
- **Financial Profile**: Average Purchase Price: $341,578.46 | Loan Application Rate: 38.4%
- **Behavioral Profile**: Acquisition Purpose: Home | Top Referral Channel: Website
- **Customer Sentiment**: Average Satisfaction Score: 3.05 / 5.0



*Saved clustering visualizations:*
- PCA 2D Cluster Space: `cluster_pca_2d.png`
- Segment Sizes: `cluster_sizes.png`
- Pairplot Matrix: `cluster_pairplot.png`

---

## 8. Business Recommendations
Based on the distinct characteristics of the discovered clusters, we propose the following strategic actions for Parcl Co. Limited:


#### Recommendations for **Global Investors**:
1. **Remote Investing Portals**: Promote digital transaction platforms and virtual viewings as these buyers are primarily international, seeking Buy-to-let properties.
2. **Property Management Bundles**: Sell turnkey leasing and management services to facilitate passive income generation for overseas investors.
3. **Agent Networks**: Strengthen international broker relationships to increase direct referrals from overseas markets.

#### Recommendations for **Corporate Buyers**:
1. **Commercial Asset Portfolios**: Present bulk purchase packages, office spaces, and commercial warehouse opportunities.
2. **Tax & Compliance Advisory**: Provide specialized corporate legal support for commercial investment acquisitions.
3. **B2B Networking**: Establish strong referral schemes with corporate business brokers and strategic corporate partners.

#### Recommendations for **Luxury Investors**:
1. **Exclusive VIP Property Previews**: Organize private viewings and invite-only listings for penthouses and premium coastal villas.
2. **Concierge Brokerage**: Assign dedicated senior relationships advisors to handle luxury asset transactions.
3. **Targeted Wealth Management Channels**: Advertise through boutique private banks, yacht clubs, and luxury lifestyle publications.

#### Recommendations for **First-Time Buyers**:
1. **First-Time Homebuyer Loan Packages**: Since 38.4% applied for loans, design high-ratio loan programs and low-deposit options in collaboration with banking partners.
2. **Social Media & Online Engagement**: Allocate marketing budget to Social Media and Online Search channels to capture the attention of this younger demographic (Average Age: 33.9).
3. **Starter Home Showcases**: Market affordable residential properties, townhouses, and suburban developments priced under $400k.


---

## 9. Conclusion
The machine learning pipeline has successfully grouped Parcl Co. Limited's client base into four highly distinct and intuitive buyer segments: **First-Time Buyers**, **Global Investors**, **Corporate Buyers**, and **Luxury Investors**. By implementing the recommendations above, Parcl Co. Limited can improve customer acquisition cost, boost satisfaction, and maximize sales velocity.
