# Depression Risk Detection Pipeline

Early depression risk detection among university students using survey data,
Apriori-based association rule mining, and rule-enriched feature construction.

---

## Structure

```
depression_pipeline/
├── data/
│   ├── raw/            raw CSV files (place train.csv here)
│   └── processed/      intermediate outputs
├── src/
│   ├── preprocessing.py    data cleaning, feature engineering, transaction encoding
│   ├── apriori_engine.py   Apriori algorithm (pure Python/NumPy, no external library)
│   ├── rule_filter.py      rule filtering, deduplication, feature construction
│   └── visualize.py        all plot generation functions
├── tests/
│   ├── test_preprocessing.py
│   └── test_apriori.py
├── pipeline.py         end-to-end runner (CLI)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the Pipeline

Place the raw CSV in `data/raw/train.csv`, then:

```bash
python pipeline.py --data data/raw/train.csv --output outputs/
```

### Options

| Flag              | Default  | Description                                         |
|-------------------|----------|-----------------------------------------------------|
| `--data`          | required | Path to raw CSV                                     |
| `--output`        | outputs/ | Directory for all outputs                           |
| `--support`       | 0.05     | Minimum Apriori support                             |
| `--confidence`    | 0.60     | Minimum rule confidence                             |
| `--lift`          | 1.10     | Minimum rule lift                                   |
| `--max-len`       | 3        | Maximum itemset length                              |
| `--sample`        | None     | Limit rows for faster dev runs (e.g. `--sample 5000`) |

### Example (fast dev run)

```bash
python pipeline.py \
    --data data/raw/train.csv \
    --output outputs/ \
    --support 0.05 \
    --confidence 0.60 \
    --lift 1.05 \
    --max-len 3 \
    --sample 5000
```

---

## Outputs

All files written to `--output` directory:

| File                        | Description                              |
|-----------------------------|------------------------------------------|
| `association_rules.csv`     | All mined rules (antecedent, consequent, support, confidence, lift, conviction) |
| `depression_rules.csv`      | Filtered rules with Depression_Yes as consequent |
| `frequent_itemsets.csv`     | All frequent itemsets with support       |
| `features_enriched.csv`     | Final feature matrix (model-ready + rule features) |
| `correlation_heatmap.png`   | Feature correlation matrix               |
| `target_distribution.png`   | Depression label distribution            |
| `rule_scatter.png`          | Support vs confidence, colour = lift     |
| `top_rules_lift.png`        | Top-15 rules ranked by lift              |
| `rule_confidence_heatmap.png` | Antecedent x consequent confidence grid |
| `rule_network.png`          | Directed rule graph (NetworkX)           |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Dataset Columns Used

| Column                              | Treatment                                 |
|-------------------------------------|-------------------------------------------|
| Gender                              | Binary item (Gender_Male / Gender_Female) |
| Age                                 | Binned: <=21, 22-25, 26-30, >30          |
| CGPA                                | Binned: Low (<5), Mid (5-7.5), High       |
| Academic Pressure                   | Binned: Low/Moderate/High                 |
| Work Pressure                       | Binned: Low/Moderate/High                 |
| Study Satisfaction / Job Satisfaction | Binned: Low/Moderate/High               |
| Sleep Duration                      | Categorical: <5h, 5-6h, 7-8h, >8h        |
| Work/Study Hours                    | Binned: Low/Moderate/High                 |
| Financial Stress                    | Binned: Low/Moderate/High                 |
| Have you ever had suicidal thoughts?| Binary: 0/1                              |
| Family History of Mental Illness    | Binary: 0/1                              |
| Depression                          | Target: 0/1                              |

---

## Notes

- Apriori is implemented from scratch (stdlib + NumPy only) — no mlxtend dependency.
- All plots use a dark theme and are saved as 150 DPI PNGs.
- Rule features (binary columns indicating antecedent satisfaction) are appended to
  the model matrix and exported in `features_enriched.csv` for downstream model use.
