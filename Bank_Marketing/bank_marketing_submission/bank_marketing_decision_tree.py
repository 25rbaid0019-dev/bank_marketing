"""Bank Marketing Campaign Response Prediction using a Decision Tree.

Dataset: UCI Bank Marketing, bank-full.csv
https://archive.ics.uci.edu/dataset/222/bank+marketing

Usage:
    1. Download bank-full.csv from UCI and place it beside this file.
    2. pip install -r requirements.txt
    3. python bank_marketing_decision_tree.py

The script follows the project report: it excludes call duration to avoid
post-call leakage, handles unknown values, one-hot encodes categorical
features in a pipeline, uses a stratified 80:20 split, class-balanced
Decision Tree training and GridSearchCV, then saves evaluation outputs.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
DATA = BASE / "bank-full.csv"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42


def main():
    if not DATA.exists():
        raise FileNotFoundError(
            "bank-full.csv not found. Download it from the UCI Bank Marketing page "
            "and place it beside this script."
        )

    df = pd.read_csv(DATA, sep=";")
    print(f"Dataset shape: {df.shape}")
    print("Target distribution:\n", df["y"].value_counts())

    # Keep the pre-call feature set used by the report.
    df = df.copy()
    df["y_bin"] = (df["y"].str.lower() == "yes").astype(int)
    df = df.drop(columns=["y"])
    df = df.drop(columns=["duration"])

    # Unknown job/education values are treated as missing and imputed;
    # contact/poutcome remain explicit categories.
    for col in ["job", "education"]:
        df[col] = df[col].replace("unknown", np.nan)

    X = df.drop(columns=["y_bin"])
    y = df["y_bin"]
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])

    model = DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)
    pipe = Pipeline([("prep", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    params = {
        "model__criterion": ["gini", "entropy"],
        "model__max_depth": [6, 8, 10, 12, None],
        "model__min_samples_split": [2, 10, 20],
        "model__min_samples_leaf": [1, 5, 10],
    }
    search = GridSearchCV(pipe, params, scoring="f1", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    pred = best.predict(X_test)
    proba = best.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }
    print("Best parameters:", search.best_params_)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("\nClassification report:\n", classification_report(y_test, pred, digits=4))

    pd.Series(metrics, name="score").to_csv(OUT / "metrics.csv")
    pd.DataFrame(confusion_matrix(y_test, pred), index=["Actual No", "Actual Yes"],
                 columns=["Pred No", "Pred Yes"]).to_csv(OUT / "confusion_matrix.csv")

    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=200)
    plt.close()

    # Feature importance after one-hot encoding.
    prep = best.named_steps["prep"]
    tree = best.named_steps["model"]
    names = prep.get_feature_names_out()
    imp = pd.Series(tree.feature_importances_, index=names).sort_values(ascending=False).head(15)
    imp.to_csv(OUT / "feature_importance.csv")
    plt.figure(figsize=(9, 6))
    imp.sort_values().plot.barh()
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUT / "feature_importance.png", dpi=200)
    plt.close()

    # Simple prediction demo using a real row from the test set.
    sample = X_test.iloc[[0]]
    sample_pred = best.predict(sample)[0]
    sample_prob = best.predict_proba(sample)[0, 1]
    demo = sample.copy()
    demo["prediction"] = "Yes" if sample_pred else "No"
    demo["yes_probability"] = sample_prob
    demo.to_csv(OUT / "sample_prediction.csv", index=False)
    print(f"Sample prediction: {'Yes' if sample_pred else 'No'} (Yes probability={sample_prob:.3f})")

    # Save a small tree visualisation.
    plt.figure(figsize=(20, 10))
    plot_tree(tree, max_depth=3, filled=True, fontsize=7)
    plt.title("Decision Tree (first 3 levels)")
    plt.tight_layout()
    plt.savefig(OUT / "decision_tree.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
