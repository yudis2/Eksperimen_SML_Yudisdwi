import os
import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# ─────────────────────────────────────────────
# 1. Koneksi ke DagsHub menggunakan Personal Access Token
#    Ganti nilai PAT di bawah dengan token kamu
# ─────────────────────────────────────────────
os.environ["DAGSHUB_USER_TOKEN"] = "bce4740be822ef92eb150033be93f5167eaa3cf0"  # ← ganti dengan PAT kamu

dagshub.init(repo_owner='yudis2', repo_name='sml', mlflow=True)

# ─────────────────────────────────────────────
# 2. Buat / aktifkan eksperimen
# ─────────────────────────────────────────────
mlflow.set_experiment("Dropout Students")

# ─────────────────────────────────────────────
# 3. Load data & split
# ─────────────────────────────────────────────
data = pd.read_csv("clean_data.csv")

X = data.drop("Status", axis=1)
y = data["Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)
input_example = X_train.iloc[0:5]

# ─────────────────────────────────────────────
# 4. Grid hyperparameter
# ─────────────────────────────────────────────
n_estimators_range = np.linspace(10, 1000, 5, dtype=int)
max_depth_range    = np.linspace(1, 50,   5, dtype=int)

best_accuracy = 0
best_params   = {}

# ─────────────────────────────────────────────
# 5. Training loop dengan MANUAL LOGGING
# ─────────────────────────────────────────────
for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        run_name = f"rf_{n_estimators}est_{max_depth}dep"

        with mlflow.start_run(run_name=run_name):

            # ── 5a. Log parameter ──────────────────────────────────────
            mlflow.log_params({
                "n_estimators": int(n_estimators),
                "max_depth":    int(max_depth),
                "random_state": 42,
                "test_size":    0.2,
            })

            # ── 5b. Latih model ────────────────────────────────────────
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
            )
            model.fit(X_train, y_train)
            y_pred      = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)

            # ── 5c. Hitung metrik (manual, tanpa autolog) ──────────────
            acc       = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            recall    = recall_score(y_test, y_pred,    average="weighted", zero_division=0)
            f1        = f1_score(y_test, y_pred,        average="weighted", zero_division=0)

            # ROC-AUC: multi-class → one-vs-rest
            try:
                auc = roc_auc_score(
                    y_test, y_pred_prob,
                    multi_class="ovr", average="weighted"
                )
            except Exception:
                auc = float("nan")

            mlflow.log_metrics({
                "accuracy":           acc,
                "precision_weighted": precision,
                "recall_weighted":    recall,
                "f1_weighted":        f1,
                "roc_auc_weighted":   auc,
            })

            # ── 5d. ARTEFAK 1 – Confusion Matrix (PNG) ─────────────────
            cm   = confusion_matrix(y_test, y_pred, labels=model.classes_)
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            disp.plot(ax=ax1, colorbar=False, cmap="Blues")
            ax1.set_title(f"Confusion Matrix\nn_est={n_estimators}, max_depth={max_depth}")
            plt.tight_layout()
            cm_path = "confusion_matrix.png"
            fig1.savefig(cm_path, dpi=100)
            plt.close(fig1)
            mlflow.log_artifact(cm_path, artifact_path="plots")

            # ── 5e. ARTEFAK 2 – Feature Importance (PNG) ───────────────
            importances = model.feature_importances_
            feat_df = (
                pd.DataFrame({"feature": X_train.columns, "importance": importances})
                .sort_values("importance", ascending=False)
                .head(20)
            )
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            sns.barplot(data=feat_df, x="importance", y="feature", ax=ax2, palette="viridis")
            ax2.set_title(f"Feature Importance (Top-20)\nn_est={n_estimators}, max_depth={max_depth}")
            ax2.set_xlabel("Mean Decrease in Impurity")
            plt.tight_layout()
            fi_path = "feature_importance.png"
            fig2.savefig(fi_path, dpi=100)
            plt.close(fig2)
            mlflow.log_artifact(fi_path, artifact_path="plots")

            # ── 5f. ARTEFAK 3 – Classification Report (TXT) ────────────
            report_str = classification_report(
                y_test, y_pred,
                target_names=[str(c) for c in model.classes_],
                zero_division=0,
            )
            report_path = "classification_report.txt"
            with open(report_path, "w") as f:
                f.write(f"Run: {run_name}\n")
                f.write(f"n_estimators={n_estimators}, max_depth={max_depth}\n\n")
                f.write(report_str)
            mlflow.log_artifact(report_path, artifact_path="reports")

            # ── 5g. Simpan model terbaik ───────────────────────────────
            if acc > best_accuracy:
                best_accuracy = acc
                best_params   = {"n_estimators": int(n_estimators), "max_depth": int(max_depth)}

                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="best_model",
                    input_example=input_example,
                )
                mlflow.set_tag("best_model", "true")

            print(
                f"[{run_name}] acc={acc:.4f} | f1={f1:.4f} | "
                f"auc={auc:.4f} | best_so_far={best_accuracy:.4f}"
            )

# ─────────────────────────────────────────────
# 6. Ringkasan akhir
# ─────────────────────────────────────────────
print("\n=== Best Model ===")
print(f"Accuracy : {best_accuracy:.4f}")
print(f"Params   : {best_params}")