"""
Explainable Credit Risk Scorer
==============================
Predicts loan default risk using XGBoost / Random Forest, then explains
*why* each applicant was flagged using SHAP (global + local) and LIME
(local, human-readable). Built so a loan officer — not a data scientist —
can understand each decision.

Run:
    python credit_risk_scorer.py

Outputs (written to ./outputs/):
    - model_comparison.csv          Accuracy/Precision/Recall/F1/ROC-AUC for both models
    - shap_summary_bar.png          Global feature importance (mean |SHAP|)
    - shap_summary_beeswarm.png     Global feature impact + direction
    - shap_dependence_<feature>.png How top feature drives risk
    - shap_waterfall_applicant.png  Why ONE specific applicant was flagged
    - lime_explanation_applicant.html  Human-readable local explanation
    - loan_officer_report.txt       Plain-English explanation for a sample applicant
    - credit_risk_model.pkl         Trained best model (pickled)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier

import shap
from lime.lime_tabular import LimeTabularExplainer

import pickle

warnings.filterwarnings("ignore")
np.random.seed(42)

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. DATA GENERATION
# ---------------------------------------------------------------------------
# Swap this function out for pd.read_csv("your_real_loan_data.csv") once you
# have real data (e.g. LendingClub, Kaggle "Give Me Some Credit", or your
# own bank's dataset). The synthetic generator below encodes realistic,
# learnable relationships so the pipeline below is fully demonstrable.

def generate_loan_dataset(n=6000):
    age = np.random.randint(21, 65, n)
    annual_income = np.random.gamma(shape=6, scale=12000, size=n).clip(15000, 300000)
    credit_score = np.random.normal(650, 90, n).clip(300, 850)
    loan_amount = np.random.gamma(shape=3, scale=6000, size=n).clip(1000, 100000)
    employment_years = np.random.gamma(shape=2, scale=3, size=n).clip(0, 40)
    debt_to_income = np.random.beta(2, 5, n) * 60  # percentage
    num_open_accounts = np.random.poisson(5, n).clip(0, 25)
    num_late_payments_2y = np.random.poisson(0.8, n).clip(0, 15)
    home_ownership = np.random.choice(["RENT", "MORTGAGE", "OWN"], n, p=[0.4, 0.45, 0.15])
    loan_purpose = np.random.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "medical", "small_business", "other"],
        n, p=[0.35, 0.25, 0.12, 0.08, 0.1, 0.1]
    )
    loan_term_months = np.random.choice([36, 60], n, p=[0.65, 0.35])

    interest_proxy = (loan_amount / (annual_income + 1)) * 100  # loan-to-income ratio

    # ---- Ground-truth risk logic (what a real bank would actually see) ----
    risk_score = (
        -0.020 * (credit_score - 650)
        + 0.045 * debt_to_income
        + 0.9 * num_late_payments_2y
        + 0.35 * interest_proxy
        - 0.06 * employment_years
        - 0.00002 * (annual_income - 60000)
        + np.where(loan_purpose == "small_business", 0.8, 0)
        + np.where(home_ownership == "RENT", 0.4, 0)
        + np.random.normal(0, 1.5, n)  # irreducible noise
    )
    default_prob = 1 / (1 + np.exp(-(risk_score - 20) / 3))
    defaulted = np.random.binomial(1, default_prob)

    df = pd.DataFrame({
        "age": age.astype(int),
        "annual_income": annual_income.round(2),
        "credit_score": credit_score.round(0),
        "loan_amount": loan_amount.round(2),
        "employment_years": employment_years.round(1),
        "debt_to_income": debt_to_income.round(2),
        "num_open_accounts": num_open_accounts,
        "num_late_payments_2y": num_late_payments_2y,
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "loan_term_months": loan_term_months,
        "loan_to_income_pct": interest_proxy.round(2),
        "defaulted": defaulted
    })
    return df


# ---------------------------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------------------------

def preprocess(df):
    df = df.copy()
    encoders = {}
    for col in ["home_ownership", "loan_purpose"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=["defaulted"])
    y = df["defaulted"]
    return X, y, encoders


# ---------------------------------------------------------------------------
# 3. MODEL TRAINING + COMPARISON
# ---------------------------------------------------------------------------

def train_and_compare(X_train, X_test, y_train, y_test):
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=15,
            class_weight="balanced", random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            eval_metric="logloss", random_state=42
        )
    }

    results = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, proba)
        })
        fitted[name] = model
        print(f"\n=== {name} ===")
        print(classification_report(y_test, preds, target_names=["No Default", "Default"]))

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    results_df.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)
    print("\nModel comparison:\n", results_df)
    return fitted, results_df


# ---------------------------------------------------------------------------
# 4. SHAP EXPLAINABILITY (global + local)
# ---------------------------------------------------------------------------

def run_shap_analysis(model, X_train, X_test, top_n_dependence=2):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # Handle binary-classifier shape differences between RF and XGBoost
    if len(shap_values.shape) == 3:
        # RandomForest: (n_samples, n_features, n_classes) -> take class 1
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    # --- Global: bar plot of mean |SHAP value| per feature ---
    plt.figure()
    shap.plots.bar(sv, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_summary_bar.png"), dpi=150)
    plt.close()

    # --- Global: beeswarm plot (impact + direction) ---
    plt.figure()
    shap.plots.beeswarm(sv, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_summary_beeswarm.png"), dpi=150)
    plt.close()

    # --- Dependence plots for top features ---
    mean_abs = np.abs(sv.values).mean(axis=0)
    top_features = X_test.columns[np.argsort(mean_abs)[::-1][:top_n_dependence]]
    for feat in top_features:
        plt.figure()
        shap.plots.scatter(sv[:, feat], show=False)
        plt.tight_layout()
        safe_name = feat.replace("/", "_")
        plt.savefig(os.path.join(OUT_DIR, f"shap_dependence_{safe_name}.png"), dpi=150)
        plt.close()

    return explainer, sv


def explain_single_applicant_shap(sv, X_test, model, applicant_idx=0):
    """Waterfall plot: why THIS applicant got THIS risk score."""
    plt.figure()
    shap.plots.waterfall(sv[applicant_idx], show=False, max_display=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_waterfall_applicant.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 5. LIME EXPLAINABILITY (local, plain-language friendly)
# ---------------------------------------------------------------------------

def run_lime_explanation(model, X_train, X_test, applicant_idx=0):
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=["No Default", "Default"],
        mode="classification",
        discretize_continuous=True
    )
    exp = explainer.explain_instance(
        X_test.iloc[applicant_idx].values,
        model.predict_proba,
        num_features=8
    )
    exp.save_to_file(os.path.join(OUT_DIR, "lime_explanation_applicant.html"))
    return exp


# ---------------------------------------------------------------------------
# 6. PLAIN-ENGLISH REPORT FOR A LOAN OFFICER
# ---------------------------------------------------------------------------

def generate_officer_report(sv, X_test, model, applicant_idx=0, top_k=5):
    row = X_test.iloc[applicant_idx]
    prob_default = model.predict_proba(X_test.iloc[[applicant_idx]])[0, 1]
    shap_row = sv[applicant_idx]

    contributions = pd.DataFrame({
        "feature": X_test.columns,
        "value": row.values,
        "shap_value": shap_row.values
    })
    contributions["direction"] = np.where(
        contributions["shap_value"] > 0, "increases risk", "decreases risk"
    )
    contributions["abs_impact"] = contributions["shap_value"].abs()
    top = contributions.sort_values("abs_impact", ascending=False).head(top_k)

    lines = []
    lines.append("LOAN RISK EXPLANATION REPORT")
    lines.append("=" * 40)
    lines.append(f"Applicant index: {applicant_idx}")
    lines.append(f"Predicted default probability: {prob_default:.1%}")
    lines.append(f"Decision: {'HIGH RISK - Flagged for review' if prob_default >= 0.5 else 'LOW RISK - Approved'}")
    lines.append("")
    lines.append(f"Top {top_k} factors driving this decision:")
    for _, r in top.iterrows():
        lines.append(f"  - {r['feature']} = {r['value']:.2f}  -> {r['direction']} "
                      f"(impact score: {r['shap_value']:+.3f})")
    lines.append("")
    lines.append("Plain-language summary:")
    biggest = top.iloc[0]
    lines.append(
        f"  This applicant's risk score is primarily driven by their "
        f"'{biggest['feature']}' value of {biggest['value']:.2f}, which "
        f"{biggest['direction']}. Review the SHAP waterfall plot and LIME "
        f"HTML report for the full breakdown."
    )

    report = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "loan_officer_report.txt"), "w") as f:
        f.write(report)
    print("\n" + report)
    return report


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def main():
    print("Generating dataset...")
    df = generate_loan_dataset(n=6000)
    print(f"Dataset shape: {df.shape}, default rate: {df['defaulted'].mean():.1%}")

    X, y, encoders = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\nTraining models...")
    fitted_models, results_df = train_and_compare(X_train, X_test, y_train, y_test)

    best_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_name]
    print(f"\nBest model by ROC-AUC: {best_name}")

    print("\nRunning SHAP analysis...")
    explainer, sv = run_shap_analysis(best_model, X_train, X_test)

    applicant_idx = int(np.argmax(best_model.predict_proba(X_test)[:, 1]))  # highest-risk applicant
    explain_single_applicant_shap(sv, X_test, best_model, applicant_idx)

    print("\nRunning LIME analysis...")
    run_lime_explanation(best_model, X_train, X_test, applicant_idx)

    print("\nGenerating loan officer report...")
    generate_officer_report(sv, X_test, best_model, applicant_idx)

    with open(os.path.join(OUT_DIR, "credit_risk_model.pkl"), "wb") as f:
        pickle.dump({"model": best_model, "encoders": encoders, "features": X.columns.tolist()}, f)

    print(f"\nAll outputs saved to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
