# Customer Analytics: User Segmentation & Fraud Detection

An end-to-end Data Science project focusing on e-commerce customer behavior analysis, demographic-based user segmentation, and transaction fraud detection using advanced machine learning pipelines.

## 🚀 Project Overview
* **Objective:** Identify distinct customer personas through advanced clustering and deploy predictive models to flag high-risk fraudulent transactions.
* **Dataset:** 5,000 synthetic e-commerce customer profiles containing transactional, behavioral, and demographic features.

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.x
* **Exploratory Data Analysis (EDA):** Pandas, NumPy, Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (K-Means, StandardScaler, PCA)
* **Advanced Frameworks:** XGBoost Classifier for tabular fraud modeling

## 📊 Key Insights & Results
1. **Customer Segmentation:** Successfully clustered users into distinct actionable groups (e.g., High-Value Loyalists, Impulsive Buyers, Dormant Users) using the Elbow Method and Silhouette Scores.
2. **Fraud Detection Pipeline:** Implemented an optimized XGBoost model handling class imbalance to detect fraudulent activities, achieving high precision and recall metrics.

## 📂 Project Structure
* `Customer_Analytics_Fraud_Detection.ipynb` - Core Jupyter Notebook containing the full pipeline (EDA to Modeling).
* `customer_clusters.png` - Visual representation of the K-Means customer segments.
* `ecommerce_customer_data.csv` - Raw structured source dataset.
* `README.md` - Technical project documentation.
*## Installation

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
jupyter notebook Customer_Analytics_Fraud_Detection.ipynb
```
## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/rkkumar945/NEU_Customer_Analytics_ML.git
```

2. Move into the project directory:
```bash
cd NEU_Customer_Analytics_ML
```

3. Install the required libraries:
```bash
pip install -r requirements.txt
```

4. Open the notebook:
```bash
jupyter notebook Customer_Analytics_Fraud_Detection.ipynb
```
