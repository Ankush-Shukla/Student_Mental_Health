# Student Mental Health: Early Depression Risk Detection

A **two-component intelligent system** that combines **Association Rule Mining** with **Machine Learning** to detect early signs of depression among university students, paired with a production-ready **Django web application** for real-time survey collection and risk prediction.

---

## 🎯 Project Overview

This project helps universities and mental health professionals identify students at risk of depression **early and interpretably**.

### Two Main Components:

1. **Offline ML Research Pipeline**

   * Discovers meaningful association rules using a **custom Apriori algorithm**
   * Generates interpretable binary rule-based features
   * Trains and evaluates **Logistic Regression** and **Random Forest** models on enriched features
   * Produces **14 high-quality visualizations** and analysis artifacts

2. **Django Web Application**

   * Public student survey interface
   * Admin dashboard with analytics and response management
   * Real-time depression risk prediction using the trained model
   * Results visible only to administrators (students see a thank-you message)

---

## 🛠 Technology Stack

* **Python** 3.14
* **Django** 6.0.3
* **Machine Learning**: scikit-learn, pandas, numpy, SHAP
* **Visualization**: matplotlib, seaborn, networkx
* **Others**: joblib, shortuuid, SQLite (development)

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

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/Ankush-Shukla/Student_Mental_Health.git
cd Student_Mental_Health

# Using uv (recommended) or pip
uv sync
# or
pip install -r requirements.txt
```

### 2. Run the ML Pipeline

```bash
python pipeline.py --data data/raw/train.csv --output outputs/
```

**Key Flags:**

* `--support 0.05` — Minimum support threshold
* `--confidence 0.60` — Minimum confidence threshold
* `--lift 1.10` — Minimum lift threshold
* `--sample 5000` — Sample rows for faster iteration

### 3. Run the Django Web App

```bash
cd StudentMentalHealth
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Access at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

* Public survey: `/surveys/`
* Admin login → Admin Dashboard

---

## 📊 What the Pipeline Generates

* `association_rules.csv`, `depression_rules.csv`
* `features_enriched.csv` (with Rule_000, Rule_001... features)
* Trained models: `lr.pkl`, `rf.pkl`
* `bin_encoders.pkl` + `feature_template.csv` (required for inference)
* 14 publication-quality visualizations (150 DPI, dark theme)
* SHAP summary plot
* Model performance metrics (`metrics.json`)

---

## 🔍 Key Features

### Interpretability-First Approach

* Custom Apriori algorithm (pure Python + NumPy, no mlxtend)
* Strong association rules converted into binary features
* Enhances model transparency and explainability

### Robust Preprocessing

* Handles survey data quirks (quote artefacts, "?" sentinels, etc.)
* Thoughtful binning strategies for Age, CGPA, Pressure, Sleep, etc.
* Consistent feature engineering between pipeline and inference

### Production-Ready Inference

* `core/inference.py` mirrors preprocessing used during training
* Persisted LabelEncoders ensure correct categorical encoding at inference
* Rule features computed using exact item-set matching

### Beautiful Admin Dashboard

* Real-time analytics with Chart.js
* Response tracking and risk statistics
* Printable reports with conic-gradient visuals

---

## 🧪 Testing

```bash
pytest tests/
```

* Covers preprocessing, Apriori logic, transaction building, and inference.

---

## ⚠️ Known Issues & Notes

* ~~`student_detail` view currently has no authentication guard (publicly accessible by ID)~~
* `mlxtend` is listed but not used (leftover dependency)
* Financial stress stored as `CharField` due to "?" values in raw data
* Result page uses session-based data (only accessible immediately after submission)

---

## 📸 Screenshots

# Outputs generated using pipeline:

<img width="1179" height="728" alt="2026-04-04-033855_hyprshot" src="https://github.com/user-attachments/assets/40f77c3a-468e-4c13-bfb4-abab8f8eddf5" />
<img width="1382" height="847" alt="2026-04-04-033842_hyprshot" src="https://github.com/user-attachments/assets/0cb65de6-1374-4780-bd3f-41ed29078ae9" />
<img width="1569" height="643" alt="2026-04-04-033832_hyprshot" src="https://github.com/user-attachments/assets/db25ff4c-869c-4548-bb50-ab0c311762c3" />
<img width="955" height="850" alt="2026-04-04-033822_hyprshot" src="https://github.com/user-attachments/assets/ca5f5a1d-f84b-4c7a-b6f2-248267b3bfc0" />
<img width="1569" height="642" alt="2026-04-04-033813_hyprshot" src="https://github.com/user-attachments/assets/ffbdead5-71e2-4eee-a600-e107f4603102" />
<img width="1571" height="571" alt="2026-04-04-033759_hyprshot" src="https://github.com/user-attachments/assets/a5770fc4-eaca-47a8-9b68-a90e2832e64e" />
<img width="1060" height="845" alt="2026-04-04-033749_hyprshot" src="https://github.com/user-attachments/assets/2d0fc81b-71a1-4d82-8852-89e2e2ecf10f" />
<img width="871" height="850" alt="2026-04-04-033733_hyprshot" src="https://github.com/user-attachments/assets/15861552-cd3b-423d-abb4-69951b4c8bc2" />
<img width="1219" height="820" alt="2026-04-04-033724_hyprshot" src="https://github.com/user-attachments/assets/b0c3f617-f662-43b4-ab68-9e455a067461" />
<img width="978" height="848" alt="2026-04-04-033704_hyprshot" src="https://github.com/user-attachments/assets/6e218970-624a-4679-8a6e-42a104391404" />
<img width="1115" height="539" alt="2026-04-04-033644_hyprshot" src="https://github.com/user-attachments/assets/6b93381f-69d7-4ac7-b832-1112725f4eb2" />

# Django App UI

<img width="1820" height="878" alt="2026-04-04-034010_hyprshot" src="https://github.com/user-attachments/assets/46c1cadd-3e32-439d-8709-47a8932d4a5c" />
<img width="1825" height="872" alt="2026-04-04-033948_hyprshot" src="https://github.com/user-attachments/assets/3c4956a3-ca6e-46a1-a943-d85e43a6ea4c" />
<img width="1826" height="880" alt="2026-04-04-033924_hyprshot" src="https://github.com/user-attachments/assets/6eb56a4e-21e9-480f-92d4-88452ffd116d" />


---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or pull requests.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Ankush Shukla
Built with ❤️ for better student mental health awareness.
