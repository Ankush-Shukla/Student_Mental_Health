# Student Mental Health: Early Depression Risk Detection

A **two-component intelligent system** that combines **Association Rule Mining** with **Machine Learning** to detect early signs of depression among university students, paired with a production-ready **Django web application** for real-time survey collection and risk prediction.

---

## 🎯 Project Overview

This project aims to help universities and mental health professionals identify students at risk of depression **early and interpretably**.

### Two Main Components:

1. **Offline ML Research Pipeline**  
   - Discovers meaningful association rules using a **custom Apriori algorithm**  
   - Generates interpretable binary rule-based features  
   - Trains and evaluates Logistic Regression + Random Forest models on enriched features  
   - Produces 14 high-quality visualizations and analysis artifacts

2. **Django Web Application**  
   - Public student survey interface  
   - Admin dashboard with analytics and response management  
   - Real-time depression risk prediction using the trained model  
   - Results are visible only to administrators (students see only a thank-you message)

---

## 🛠 Technology Stack

- **Python** 3.14
- **Django** 6.0.3
- **Machine Learning**: scikit-learn, pandas, numpy, SHAP
- **Visualization**: matplotlib, seaborn, networkx
- **Others**: joblib, shortuuid, SQLite (development)

---

## 📁 Project Structure

```bash
.
├── pipeline.py                          # Main CLI pipeline orchestrator
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── data/
│   └── raw/                             # Place your train.csv here
├── outputs/                             # Generated artifacts (gitignored)
├── src/
│   ├── preprocessing.py
│   └── models/
│       ├── train_models.py
│       └── evaluate.py
├── transaction_encoding/
│   └── apriori_engine.py                # Custom Apriori implementation
├── external_library/
│   └── rule_filter.py
├── construction/
│   └── visualize.py                     # 14 visualization functions
├── StudentMentalHealth/                 # Django Web Application
│   ├── manage.py
│   ├── core/
│   │   ├── models.py
│   │   ├── views.py
│   │   ├── urls.py
│   │   ├── inference.py                 # Bridge between Django & ML model
│   │   ├── templates/
│   │   └── admin.py
│   └── db.sqlite3
├── tests/                               # Unit tests
└── README.md
```
# 🚀 Quick Start
1. Clone & Setup
```bash
git clone https://github.com/Ankush-Shukla/Student_Mental_Health.git
cd Student_Mental_Health
```
# Using uv (recommended) 
uv sync
# or pip
pip install -r requirements.txt
2. Run the ML Pipeline
```bash
python pipeline.py --data data/raw/train.csv --output outputs/
```

Key Flags:
--support 0.05 — Minimum support threshold
--confidence 0.60 — Minimum confidence threshold
--lift 1.10 — Minimum lift threshold
--sample 5000 — Sample rows for faster iteration

3. Run the Django Web App
Bashcd StudentMentalHealth
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
Access at: http://127.0.0.1:8000

Public survey: /surveys/
Admin login → Admin Dashboard


📊 What the Pipeline Generates

association_rules.csv, depression_rules.csv
features_enriched.csv (with Rule_000, Rule_001... features)
Trained models: lr.pkl, rf.pkl
bin_encoders.pkl + feature_template.csv (required for inference)
14 publication-quality visualizations (150 DPI, dark theme)
SHAP summary plot
Model performance metrics (metrics.json)


🔍 Key Features
Interpretability-First Approach

Custom Apriori algorithm (pure Python + NumPy, no mlxtend)
Strong association rules are converted into binary features
Makes the final model more transparent and explainable

Robust Preprocessing

Handles survey data quirks (quote artefacts, "?" sentinels, etc.)
Thoughtful binning strategies for Age, CGPA, Pressure, Sleep, etc.
Consistent feature engineering between pipeline and inference

Production-Ready Inference

core/inference.py mirrors the exact preprocessing used during training
Persisted LabelEncoders ensure correct categorical encoding at inference time
Rule features are computed using exact item-set matching

Beautiful Admin Dashboard

Real-time analytics with Chart.js
Response tracking and risk statistics
Printable reports with conic-gradient visuals


🧪 Testing
Bashpytest tests/
Covers preprocessing, Apriori logic, transaction building, and inference.

⚠️ Known Issues & Notes

student_detail view currently has no authentication guard (publicly accessible by ID)
mlxtend is listed but not used (leftover dependency)
Financial stress is stored as CharField due to "?" values in raw data
Result page uses session-based data (only accessible immediately after submission)


📸 Screenshots
(Add screenshots of pipeline outputs, survey form, admin dashboard, and analytics here later)

🤝 Contributing
Contributions are welcome! Feel free to open issues or pull requests.
📄 License
This project is licensed under the MIT License.

👨‍💻 Author
Ankush Shukla

Built with ❤️ for better student mental health awareness.
