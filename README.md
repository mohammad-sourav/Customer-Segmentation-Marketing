# Customer Segmentation Using Unsupervised Machine Learning for Targeted Marketing Strategy

## Project Overview

This project delivers a complete, end-to-end unsupervised machine learning pipeline for **customer segmentation** and **data-driven marketing strategy formulation**. Using a consumer behavioral dataset containing **2,240 customers** and **29 attributes**, the analysis uncovers hidden patterns in purchasing behavior, demographic profiles, and engagement levels.

The pipeline combines **RFM (Recency, Frequency, Monetary) analysis**, **demographic feature engineering**, **outlier treatment**, **standard scaling**, **K-Means clustering**, and **Principal Component Analysis (PCA)** to produce **four distinct, business-actionable customer segments** where each paired with a tailored marketing strategy to improve targeting, personalization, and retention.

The core objective is: **stop treating all customers the same, and start treating the right customers the right way.**

---

## Repository Architecture

```text
Customer-Segmentation-Marketing/
│
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb          # Data loading, cleaning, imputation, EDA
│   ├── 02_feature_engineering.ipynb       # RFM, Tenure, Age, outlier capping, scaling
│   └── 03_clustering_and_profiling.ipynb  # K-Means, PCA, profiling, export
│
├── data/
│   ├── raw/
│   │   └── Dataset.csv
│   └── processed/
│       ├── cleaned_customer_data.csv
│       ├── processed_customer_data.csv
│       ├── scaled_customer_data.csv
│       └── final_customer_segments.csv
│
├── visualizations/
│   ├── customer_age_income_distribution.png
│   ├── optimal_k_evaluation.png
│   └── customer_segments_pca.png
│
├── requirements.txt
└── .gitignore
```

---

## Methodology & Technical Pipeline

### 1. Data Cleaning & Exploratory Data Analysis (EDA)

- **Dataset inspected:** 2,240 rows × 29 columns.
- **Missing value audit:** Only the `Income` column had missing values (**24 records**). These were imputed using the **median income** to avoid distortion from outliers.
- **Age feature created** from `Year_Birth`.
- **EDA performed** on age and income distributions, revealing right-skewed income patterns with extreme upper outliers (up to 666,666), validating the need for downstream outlier treatment.

### 2. Feature Engineering & Scaling

- **RFM Features Engineered:**
  - **Frequency** = `NumWebPurchases + NumCatalogPurchases + NumStorePurchases + NumDealsPurchases`
  - **Monetary** = `MntWines + MntFruits + MntMeatProducts + MntFishProducts + MntSweetProducts + MntGoldProds`
  - **Recency** (already present in dataset)
- **Customer Tenure:** `Customer_Tenure_Days` calculated as the number of days between each customer's enrollment date (`Dt_Customer`) and the most recent enrollment date in the dataset.
- **Outlier Treatment:** Extreme values in `Monetary`, `Income`, and `Customer_Tenure_Days` were **capped using the IQR method (1.5 × IQR rule)** to prevent distance distortion in clustering.
- **Feature Scaling:** All clustering features were standardized with `StandardScaler` (μ = 0, σ = 1) so that no single feature dominates the Euclidean distance metric used by K-Means.

**Final Clustering Feature Set (6 features):**

`Recency`, `Frequency`, `Monetary`, `Customer_Tenure_Days`, `Age`, `Income`

### 3. Unsupervised Clustering (K-Means)

- Tested **K = 2 through 7**.
- Evaluated using two complementary validation metrics:
  - **Elbow Method** (inertia / within-cluster sum of squares)
  - **Silhouette Score**
- **Selected K = 4** as the optimal balance between cluster compactness and separation.

### 4. Dimensionality Reduction & Visualization

- Applied **PCA** to project the 6-dimensional feature space into **2 principal components**.
- **PC1 explained 42.76%** of variance; **PC2 explained 17.66%** — totaling **60.42% cumulative explained variance**.
- Visualized the 4 clusters in 2D PCA space, confirming reasonable separation of the segments.

### 5. Business Profiling & Persona Generation

- Computed **cluster centroids** across all RFM, tenure, and demographic variables.
- Translated centroids into **business-friendly personas** with clear behavioral descriptions.
- Delivered **segment-specific marketing strategies** for each cluster.

---

## Key Visualizations & Findings

### 1. Optimal Cluster Evaluation (K = 4)

The Elbow curve shows diminishing returns in inertia reduction beyond K = 4, while the Silhouette Score peaks/plateaus around the same region — confirming **K = 4** as the most defensible choice.

**Visual Reference:** `visualizations/optimal_k_evaluation.png`

### 2. PCA Cluster Separation

The 2D PCA scatter plot clearly illustrates the distribution and separation of all four customer cohorts across the first two principal components, with a combined explained variance of **60.42%**.

**Visual Reference:** `visualizations/customer_segments_pca.png`

### 3. Age & Income Distribution

EDA plots highlight the skew of income and the central tendency of age (~57 years), with outliers identified before IQR capping.

**Visual Reference:** `visualizations/customer_age_income_distribution.png`

---

## Summary of Customer Segments

The clustering produced four well-defined segments, each with distinct behavioral and demographic signatures:

| **Cluster** | **Customer Segment** | **Percentage** | **Key Characteristics** | **Recommended Marketing Strategy** |
|:---|:---|:---|:---|:---|
| **0** | **High-Value Loyalists** | 25.49% | Highest frequency, highest monetary spend, longest tenure, high income. | VIP loyalty programs, exclusive offers, premium product early access, referral incentives. |
| **1** | **Budget/Casual Shoppers** | 27.50% | Lowest frequency, lowest monetary spend, low income, very recent purchases. | Discount-driven campaigns, value promotions, and second-purchase incentives to build habit. |
| **2** | **Affluent Potential Loyalists** | 22.14% | Highest income, high spend and frequency but **shortest tenure**. | Personalized onboarding, premium recommendations, and loyalty-building to convert spending into loyalty. |
| **3** | **Low-Engagement At-Risk** | 24.87% | Low spend and frequency, **worst recency**, moderate tenure. | Win-back campaigns, "we miss you" offers, and re-engagement nudges. |

---

## Tech Stack & Dependencies

**Programming Language:** Python 3

**Core Libraries:**

- `pandas` — Data manipulation and analysis
- `numpy` — Numerical computing
- `scikit-learn` — StandardScaler, KMeans, PCA, silhouette_score
- `matplotlib` — Data visualization
- `seaborn` — Statistical visualization

All dependencies are specified in `requirements.txt`.

### Installation

```bash
pip install -r requirements.txt
```

---

## Getting Started & Replication

Follow these steps to reproduce the analysis.

### 1. Clone the Repository

```bash
git clone https://github.com/mohammad-sourav/Customer-Segmentation-Marketing.git

cd Customer-Segmentation-Marketing
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebooks in Order

1. `notebooks/01_eda_and_cleaning.ipynb` — Loads raw data, handles missing values, creates `Age`, performs EDA, and exports `cleaned_customer_data.csv`.
2. `notebooks/02_feature_engineering.ipynb` — Engineers RFM and Tenure features, caps outliers via IQR, applies StandardScaler, and exports `processed_customer_data.csv` and `scaled_customer_data.csv`.
3. `notebooks/03_clustering_and_profiling.ipynb` — Evaluates optimal K, trains K-Means (K = 4), applies PCA, profiles clusters, and exports `final_customer_segments.csv`.

Following this sequence reproduces the full data preparation → feature engineering → clustering → profiling workflow.

---

## Business Impact

This project translates raw transaction data into a **repeatable segmentation engine** that enables:

- **Precision targeting** — Focus marketing spend on the segments most likely to convert.
- **Retention uplift** — Identify at-risk customers before they churn.
- **Personalized offers** — Match messages and channels to behavioral profiles.
- **Loyalty optimization** — Reward the customers who drive the most revenue.
- **Campaign ROI improvement** — Reduce inefficient marketing expenditure through more targeted customer selection.

---

## 👤 Author

**Mohammad Sourav**

*Interdisciplinary Researcher | Consumer Psychology, Artificial Intelligence & Data Science*
