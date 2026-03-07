# 🔍 Auto-Insight — Automatic Data Analytics Engine
> Upload any CSV file and get instant ML-powered insights — no coding required!

[Auto-Insight Banner]<img width="1911" height="723" alt="ml-insights png" src="https://github.com/user-attachments/assets/9dcd5e11-8045-4547-bfe8-1e13d0a7663f" />


---

## ✨ What It Does

Auto-Insight is a **no-code analytics platform** that automatically analyzes any CSV file and gives you:

- 📈 **Regression Analysis** — Predicts your target column with 90%+ accuracy
- 🔵 **Clustering** — Groups your data into natural segments
- ⚠️ **Anomaly Detection** — Finds suspicious or unusual records
- 📅 **Time Series Forecasting** — Predicts future values
- 🧹 **Auto Preprocessing** — Cleans your data automatically
- 📄 **PDF Export** — Download a full analysis report

---

## 🖥️ Screenshots

| Data Explorer | Preprocessing |
|---|---|
| !(<img width="1919" height="882" alt="data-explorer png" src="https://github.com/user-attachments/assets/4b1ea412-e07b-406b-8119-660c05ee7793" />) | !(<img width="1916" height="870" alt="preprocessing png" src="https://github.com/user-attachments/assets/d566b0dd-9f31-40cb-9fd8-cd689da39e08" />) |

---

## 📊 Performance Results

| Metric | Value |
|---|---|
| R² Score | **0.9015** (90% accuracy) |
| RMSE | **1876.59** |
| Clustering (Silhouette) | **0.45** |
| Anomaly Detection | **4% anomaly rate** |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React + TypeScript + Vite |
| Backend | Python + Flask |
| ML | Scikit-learn, Pandas, NumPy |
| Charts | Recharts |
| Export | PDF Generation |

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.x
- Node.js

### Step 1 — Clone the repo
```bash
git clone https://github.com/om3107/auto-insight.git
cd auto-insight
```

### Step 2 — Install Python dependencies
```bash
pip install flask pandas numpy scikit-learn scipy
```

### Step 3 — Install Node dependencies
```bash
npm install
```

### Step 4 — Run Backend (Terminal 1)
```bash
python app.py
```

### Step 5 — Run Frontend (Terminal 2)
```bash
npm run dev
```

### Step 6 — Open in Browser
```
http://localhost:3000
```

---

## 📁 Project Structure

```
auto-insight/
├── modules/
│   ├── analysis.py        # ML engine
│   ├── preprocessing.py   # Data cleaning
│   └── ingestion.py       # Data loading
├── src/
│   ├── modules/           # Frontend components
│   └── App.tsx            # Main UI
├── app.py                 # Flask backend
└── README.md
```

---

## 👨‍💻 Author

**om3107** — Built with ❤️ as a student project

⭐ Star this repo if you found it useful!
