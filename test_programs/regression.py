"""
Docstring for core.regression
"""


#---- // Graphing //----
import matplotlib.pyplot as plt
import os

#---- // ML Modules // ----
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

#Create a directory for any outputs
os.makedirs("test_outputs", exist_ok=True)

#Read the CSV file and save into a data frame
df = pd.read_csv('data/StudentPerformanceFactors.csv')

#Use one-hot encoding to change string values into usable data
X = pd.get_dummies(df.drop("Exam_Score", axis = 1))
y = df["Exam_Score"]

#Get the encoded values
X = pd.get_dummies(X, drop_first= True)

#Create the different testing and training states
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create the ML linear regression model
model = LinearRegression()

#Fit in the X and y value training sets
model.fit(X_train, y_train)

#Predict the exam scores and print the first 10 values
y_pred = model.predict(X_test)
print("\n", y_pred[:10])

#Find how accurate our ML exam score predictions are
mae = mean_absolute_error(y_test, y_pred)

#Find any variation in the data
r2 = r2_score(y_test, y_pred)

#Print our results on a new line
print("\nMAE:", mae)
print("\nR²:", r2)

coefficients = pd.Series(model.coef_, index = X.columns)
coefficients.sort_values(ascending=False)
print("\n", coefficients.head())

#Create and print a data frame to compare the actual and predicted data values
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
print("\n", comparison.head())

#Create graphs based on the data
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Exam Scores")
plt.ylabel("Predicted Exam Score")
plt.title("Actual vs. Predicted Exam Score")
plt.savefig("test_outputs/actual_vs_predicted.png")
plt.close()

errors = y_test - y_pred
plt.hist(errors)
plt.title("Prediction Errors")
plt.xlabel("Error (Actual - Predicted)")
plt.savefig("test_outputs/error_distribution.png")
plt.close()

plt.figure(figsize=(8,12))
coefficients.sort_values().plot(kind="barh")
plt.title("Feature Impact on Exam Score")
plt.tight_layout()
plt.savefig("test_outputs/feature_importance.png")
plt.close()

#Save results to csv files
comparison.to_csv("test_outputs/predictions.csv", index=False)
coefficients.to_csv("test_outputs/coefficients.csv")

