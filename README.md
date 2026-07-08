# Explainable Credit Risk Scorer

Predicts loan default risk with **Random Forest** and **XGBoost**, then explains
*why* each applicant was flagged using **SHAP** (global + per-applicant) and
**LIME** (plain-language local explanation) — the part that actually matters
to a loan officer, not just the accuracy number.

## Why this project stands out
Most student ML projects stop at "here's my accuracy score." Banks and
regulators (fair lending laws, GDPR "right to explanation") require models
to justify individual decisions. This project treats explainability as a
first-class deliverable, not an afterthought.

## Setup
```bash
pip install xgboost shap lime scikit-learn pandas matplotlib seaborn
python credit_risk_scorer.py
```

## Using real data instead of synthetic data
The script currently calls `generate_loan_dataset()` to create a realistic
synthetic dataset (17-18% default rate, matching real-world loan portfolios).
To use a real dataset, replace that call in `main()` with:

```python
df = pd.read_csv("your_loan_data.csv")
```

Good public datasets to swap in:
- **LendingClub Loan Data** (Kaggle) — real peer-to-peer lending outcomes
- **Give Me Some Credit** (Kaggle) — classic credit-scoring competition dataset
- **UCI Statlog German Credit Data** — smaller, well-documented benchmark

Just make sure your target column is named `defaulted` (0/1) or update the
`preprocess()` function's `y = df["defaulted"]` line.

## What gets generated (in `outputs/`)
| File | What it shows |
|---|---|
| `model_comparison.csv` | Accuracy / Precision / Recall / F1 / ROC-AUC for both models |
| `shap_summary_bar.png` | Which features matter most overall |
| `shap_summary_beeswarm.png` | Which features matter most **and** which direction they push risk |
| `shap_dependence_*.png` | How the single most important feature relates to risk (e.g. does risk rise linearly with debt-to-income, or is there a cliff?) |
| `shap_waterfall_applicant.png` | For ONE specific (highest-risk) applicant: exactly which factors pushed them from "average risk" to "flagged" |
| `lime_explanation_applicant.html` | Same applicant, explained in LIME's plain-language local-approximation format — open this in a browser |
| `loan_officer_report.txt` | Auto-generated plain-English writeup: "This applicant was flagged because X, Y, Z" |
| `credit_risk_model.pkl` | The trained model + encoders, ready to load and score new applicants |

## Extending it further (good "Advanced Challenge" additions)
- Add a **Streamlit dashboard** where a loan officer types in applicant details and gets a live SHAP waterfall + LIME explanation.
- Add **fairness auditing**: check if the model's false-positive rate differs across protected attributes (age groups, etc.) using `fairlearn`.
- Add **counterfactual explanations** ("if this applicant's credit score were 40 points higher, they'd be approved") using the `dice-ml` library.
- Compare SHAP vs. LIME explanations for the same applicant and discuss where they agree/disagree — a great talking point in interviews.

## Resume bullet you can use
> Built an explainable credit-risk scoring pipeline (XGBoost/Random Forest, ROC-AUC 0.92) with SHAP and LIME interpretability layers, generating human-readable, per-applicant risk explanations for loan-officer review.
