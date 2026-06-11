# Credit Card Customer Segmentation

Unsupervised learning to identify behavioural segments among 8,950 credit card customers, with business recommendations for each segment.

---

## Dataset

[Kaggle — Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)

- 8,950 customers, 18 behavioural features
- Features include balance, purchases, cash advances, payment behaviour, credit limit, transaction frequencies
- No target variable

---

## Preprocessing

- Median imputation for `MINIMUM_PAYMENTS` (3.5% missing) and `CREDIT_LIMIT` (0.01% missing)
- `log1p` transformation applied to 10 skewed monetary and count columns (skewness range: 2.39–13.85)
- Frequency and ratio columns (bounded 0–1) left untransformed
- `TENURE` dropped: 84.7% of customers share identical value of 12
- 3 engineered features: `PURCHASE_TO_LIMIT_RATIO`, `CASH_TO_PURCHASE_RATIO`, `PAYMENT_TO_MINIMUM_RATIO`

---

## Methods

**Scaling:** StandardScaler applied after log transformation (log transform fixes distributional shape; scaling fixes inter-feature scale differences)

**Dimensionality reduction:** PCA reduced 19 features to 7 components (84.4% variance explained). Applied to address multicollinearity (max r=0.92 between PURCHASES and ONEOFF_PURCHASES) and reduce noise before clustering.

**Optimal K selection** via three independent methods:

| Method | Result |
|---|---|
| Elbow (inertia) | K = 5 |
| Silhouette analysis | K = 2 (statistically optimal, not business-useful) |
| Hierarchical dendrogram | 2 major branches, 4–5 sub-groups |

K=5 selected: supported by elbow method, consistent with dendrogram sub-structure, silhouette score (0.260) only marginally below K=2 (0.296).

**Method validation:** K-Means compared against Gaussian Mixture Models (GMM)

| Metric | K-Means | GMM |
|---|---|---|
| Silhouette score | 0.2596 | 0.1690 |
| Cluster size range | 1,416–2,195 | 865–2,513 |
| ARI vs K-Means | — | 0.42 |
| NMI vs K-Means | — | 0.48 |

K-Means selected: higher silhouette score, more balanced segments. ARI=0.42 and NMI=0.48 indicate both methods identified consistent broad structure. GMM average assignment confidence of 95.6% validates segment distinctiveness; 234 customers (2.6%) were ambiguous between segments.

---

## Segments

| Segment | N | Avg Balance | Avg Purchases | Avg Cash Advance | Full Payment Rate |
|---|---|---|---|---|---|
| Cash Borrowers | 2,195 (24.5%) | $2,327 | $17 | $2,087 | 3% |
| Installment Shoppers | 1,930 (21.6%) | $436 | $612 | $33 | 31% |
| Infrequent Big Spenders | 1,625 (18.2%) | $392 | $328 | $139 | 12% |
| High Value Users | 1,784 (19.9%) | $1,528 | $3,038 | $67 | 26% |
| Revolvers | 1,416 (15.8%) | $3,312 | $1,276 | $2,662 | 3% |

**Cash Borrowers** — card used almost exclusively for cash withdrawals. Near-zero purchase frequency. High default risk.

**Installment Shoppers** — regular purchasers, strong preference for installment plans, highest full payment rate. Low-risk, high-engagement.

**Infrequent Big Spenders** — low purchase frequency, moderate spend when active, good payment behaviour. Under-engaged with untapped potential.

**High Value Users** — highest purchases ($3,038/month), highest credit limits, highest frequency. Most valuable segment.

**Revolvers** — high balance and cash advance combined with active purchasing. Near-zero full payment rate. Short-term revenue, elevated default risk.

---

## Business Recommendations

**Cash Borrowers** — cap cash advance limits, offer structured personal loan products, monitor for default risk.

**Installment Shoppers** — increase credit limits, target with retail merchant partnerships and installment-specific offers.

**Infrequent Big Spenders** — re-engage with campaigns in travel, electronics, premium categories.

**High Value Users** — retain with premium loyalty programme, proactive credit limit reviews, churn prevention priority.

**Revolvers** — offer balance transfer deals and structured repayment plans to reduce default risk before balances become unserviceable.

---

## Limitations

- Silhouette scores across K=2–12 ranged 0.23–0.30, reflecting customer behaviour existing on a continuum rather than in discrete clusters. This is typical of real customer data.
- 234 customers (2.6%) were flagged by GMM as ambiguous between segments.
- Dataset covers 6 months. Segments should be re-run periodically as behaviour evolves.
- No demographic data available. Income, age, and geography would improve segment interpretability.

---

## Structure

```
customer_segmentation/
├── data/
│   ├── CC GENERAL.csv           # Raw data (download from Kaggle)
│   ├── cleaned_data.csv
│   └── clustered_data.csv
├── notebooks/
│   ├── 01_setup_and_eda.ipynb
│   └── 02_clustering.ipynb
└── outputs/
    └── customer_segmentation_dashboard.pdf
```

---

## Setup

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn yellowbrick kneed jupyter
```
