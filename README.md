# 📊 Auto-Insight: Automatic Data Analytics Engine

Auto-Insight is a powerful, low-code data science tool designed to transform raw CSV data into actionable insights instantly. It automates the entire pipeline from data cleaning and preprocessing to advanced machine learning modeling.



## 🌟 Key Features
- [cite_start]**Automated Preprocessing:** Automatically handles missing values, detects outliers, and performs intelligent feature selection[cite: 8, 30].
- [cite_start]**Multi-Model Engine:** Supports Regression, Clustering, Anomaly Detection, and Time-Series Forecasting in a single run[cite: 10, 14, 17, 20].
- [cite_start]**Data Resilience:** Built-in logic to handle "corrupt" datasets by automatically dropping noisy columns to maintain high model accuracy[cite: 8, 11].
- [cite_start]**Professional Reporting:** Generates a comprehensive PDF audit trail of all preprocessing steps and ML metrics[cite: 1, 22].

## 🧪 Performance & Resilience
During stress testing, the engine demonstrated remarkable stability when comparing "Good" vs. "Corrupt" data:

| Metric | Clean Dataset | Corrupt Dataset |
| :--- | :--- | :--- |
| **Regression ($R^2$)** | [cite_start]**0.9015** [cite: 33] | [cite_start]**0.9056** [cite: 11] |
| **Missing Values Handled** | [cite_start]0 [cite: 30] | [cite_start]8 [cite: 8] |
| **Clustering Stability** | [cite_start]0.4500 Silhouette [cite: 38] | [cite_start]0.4500 Silhouette [cite: 16] |

[cite_start]*The engine successfully "salvaged" the corrupt file by identifying and dropping 6 problematic columns, actually improving the $R^2$ score[cite: 8, 11].*

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone [https://github.com/om3107/auto-insight.git](https://github.com/om3107/auto-insight.git)
cd auto-insight
