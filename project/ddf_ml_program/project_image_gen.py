# ============================================================
# Program: Graph Plotting
# Purpose: Load data gathered from the classification and regression models and create graphs based on it
# Evaluation Metric: None
# ============================================================

# ----------------------------
# Library Imports
# ----------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# ----------------------------
# Load & Prepare Dataset
# ----------------------------

# Load prediction and features from regression model
results = pd.read_csv("project/ml_data_output/spending_predictions.csv")
importances = pd.read_csv("project/ml_data_output/feature_importance.csv")

# Extract needed columns
y_test = results["Actual"]
predictions = results["Predicted"]

# Group by grade
grade_means = results.groupby("Grade Level")[["Predicted"]].mean()
grade_means = grade_means.sort_values("Predicted")

# Load ROC data
roc_df = pd.read_csv("project/ml_data_output/roc_data.csv")
roc_score = roc_df["AUC"].iloc[0]

#Load cleaned ROC Data
clean_roc_df = pd.read_csv("project/ml_data_output/cleaned_roc_data.csv")
clean_roc_score = clean_roc_df["AUC"].iloc[0] 

# ----------------------------
# Plot data in graphs
# ----------------------------

# ---- Actual vs Predicted ----
plt.figure(figsize=(8,8))
plt.scatter(y_test, predictions)

min_val = min(min(y_test), min(predictions))
max_val = max(max(y_test), max(predictions))
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.title(f"Actual vs Predicted (R² = {r2_score(y_test, predictions):.3f})")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.savefig("project/ml_graph_output/actual_vs_predicted.png")
plt.close()

# ---- Feature Importance ----
top_importances = importances.sort_values(
    by="Importance", ascending=False
).head(15)

plt.figure(figsize=(20,8))
plt.barh(top_importances["Feature"], top_importances["Importance"])
plt.title("Top 15 Important Features")
plt.tight_layout()
plt.savefig("project/ml_graph_output/feature_importance.png")
plt.close()

# ---- Range of Error ----
errors = y_test - predictions

plt.figure(figsize=(8,6))
plt.hist(errors, bins=15)
plt.axvline(0, linestyle="--")
plt.title("Prediction Error Distribution")
plt.xlabel("Error (Actual - Predicted)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("project/ml_graph_output/error_distribution.png")
plt.close()

# ---- Residual ----
plt.figure(figsize=(8,6))
plt.scatter(predictions, errors, alpha=0.7)
plt.axhline(0, linestyle="--")

plt.xlabel("Predicted Spending ($)")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot")

plt.tight_layout()
plt.savefig("project/ml_graph_output/residual_plot.png")
plt.close()

# ---- Avg Spending by Grade ----
plt.figure(figsize=(8,6))

plt.bar(grade_means.index, grade_means["Predicted"])

plt.ylabel("Average Predicted Monthly Spending ($)")
plt.xlabel("Grade Level")
plt.title("Average Predicted Spending by Grade (Test Set)")
plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig("project/ml_graph_output/spending_by_grade.png")
plt.close()

# ---- Uncleaned ROC-AUC Curve ----
plt.figure(figsize=(8,6))

plt.plot(roc_df["FPR"], roc_df["TPR"])
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (AUC = {roc_score:.3f})")

plt.tight_layout()
plt.savefig("project/ml_graph_output/roc_curve.png")
plt.close()

# ---- Cleaned ROC-AUC Curve ----
plt.figure(figsize=(8,6))
plt.plot(clean_roc_df["FPR"], clean_roc_df["TPR"])
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (AUC = {clean_roc_score:.3f})")

plt.tight_layout()
plt.savefig("project/ml_graph_output/clean_roc_curve.png")
plt.close()