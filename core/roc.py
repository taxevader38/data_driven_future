#---- // Graphing //----
import matplotlib.pyplot as plt
import os

#---- // ML Modules // ----
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

#Create a directory for any outputs
os.makedirs("test_outputs_v2", exist_ok=True)

#Read the CSV file and save into a data frame
df = pd.read_csv('data/StudentPerformanceFactors.csv')

X = pd.get_dummies(df.drop("Exam_Score", axis = 1))
y_bin = (df["Exam_Score"] >= 70).astype(int)
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_probs = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_probs)

print("\nAUROC:", auc)

fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.savefig("test_outputs_v2/roc_curve.png")
plt.close()
