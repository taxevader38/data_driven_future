import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Load saved data
results = pd.read_csv("project/ml_graph_output/test_predictions.csv")
importances = pd.read_csv("project/ml_graph_output/feature_importance.csv")

# Extract columns
y_test = results["Actual"]
predictions = results["Predicted"]

# Group by grade
grade_means = results.groupby("Grade Level")[["Predicted"]].mean()
grade_means = grade_means.sort_values("Predicted")

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

plt.figure(figsize=(8,6))

plt.bar(grade_means.index, grade_means["Predicted"])

plt.ylabel("Average Predicted Monthly Spending ($)")
plt.xlabel("Grade Level")
plt.title("Average Predicted Spending by Grade (Test Set)")
plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig("project/ml_graph_output/spending_by_grade.png")
plt.close()