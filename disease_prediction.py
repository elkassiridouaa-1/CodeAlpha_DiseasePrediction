"""
CodeAlpha ML Internship - Task 4: Disease Prediction from Medical Data
Author: El Kassiri Douaa

Predicts whether a breast tumour is malignant or benign from 30 features
computed from digitised cell images (UCI Breast Cancer Wisconsin dataset).

Models compared: Logistic Regression, SVM, Random Forest, XGBoost (if installed).
Run:  python disease_prediction.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save plots to files (works in Colab and terminals)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             RocCurveDisplay)

RANDOM_STATE = 42
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------- 1. Load data
data = load_breast_cancer(as_frame=True)
df = data.frame
# In sklearn: 0 = malignant, 1 = benign. Flip so 1 = disease (malignant).
df["target"] = 1 - df["target"]
print("Dataset shape:", df.shape)
print("Class balance (1 = malignant):\n", df["target"].value_counts(), "\n")

# ---------------------------------------------------------------- 2. Explore
plt.figure(figsize=(10, 8))
top = df.corr()["target"].abs().sort_values(ascending=False).index[:11]
sns.heatmap(df[top].corr(), cmap="coolwarm", center=0, annot=False)
plt.title("Correlation of the 10 features most related to the diagnosis")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/correlation_heatmap.png", dpi=120)
plt.close()

# ---------------------------------------------------------------- 3. Split
X = df.drop(columns="target")
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# ---------------------------------------------------------------- 4. Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "SVM (RBF)": SVC(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
}
try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(n_estimators=300, learning_rate=0.05,
                                      eval_metric="logloss", random_state=RANDOM_STATE)
except ImportError:
    print("xgboost not installed - skipping it (pip install xgboost)\n")

def scores(model, X):
    """Probability of disease if available, otherwise SVM decision score (both work for ROC-AUC)."""
    if hasattr(model, "predict_proba") and hasattr(model[-1], "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


results = []
fitted = {}
for name, model in models.items():
    # Scaling inside a pipeline so it is learned from training data only
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1").mean()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = scores(pipe, X_test)
    results.append({
        "Model": name,
        "CV F1": cv,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    })
    fitted[name] = pipe

results_df = pd.DataFrame(results).sort_values("F1", ascending=False).round(4)
print("Model comparison on the test set:\n", results_df.to_string(index=False), "\n")
results_df.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)

# ---------------------------------------------------------------- 5. Tune the best model (SVM grid as example)
param_grid = {"model__C": [0.1, 1, 10, 100], "model__gamma": ["scale", 0.01, 0.001]}
grid = GridSearchCV(Pipeline([("scaler", StandardScaler()),
                              ("model", SVC(random_state=RANDOM_STATE))]),
                    param_grid, cv=5, scoring="recall")  # recall: don't miss sick patients
grid.fit(X_train, y_train)
best = grid.best_estimator_
print("Best SVM params:", grid.best_params_)
pred = best.predict(X_test)
print("\nTuned SVM - classification report:\n",
      classification_report(y_test, pred, target_names=["benign", "malignant"]))

# ---------------------------------------------------------------- 6. Plots
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["benign", "malignant"], yticklabels=["benign", "malignant"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion matrix - tuned SVM")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=120); plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
for name, pipe in fitted.items():
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
ax.set_title("ROC curves"); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curves.png", dpi=120); plt.close()

rf = fitted["Random Forest"].named_steps["model"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()[-10:]
imp.plot(kind="barh", figsize=(7, 5), title="Top 10 features (Random Forest)")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=120); plt.close()

# ---------------------------------------------------------------- 7. Save model
joblib.dump(best, f"{OUT_DIR}/disease_model.joblib")
print(f"Plots, results and model saved in ./{OUT_DIR}/")
