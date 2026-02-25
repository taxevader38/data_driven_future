#---- // Graphing //----
import matplotlib.pyplot as plt
import os

#---- // ML Modules // ----
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

#Make a directory dedicated to storing the graph outputs
os.makedirs("ml_graph_output", exist_ok=True)

#Read the data in the CSV file and save it to a dataframe
df = pd.read_csv("data/ddf.csv")

#Drop the timestamp column to prevent any tree regression errors
if "Timestamp" in df.columns:
    df = df.drop(columns=["Timestamp"])

#Define a variable with our target column
target_col = "How much do you spend monthly on in-game purchases?"

#Map out the median spending values for each category as our output
spending_map = {
    "Nothing": 0,
    "$1 - $10": 5,
    "$15 - $30": 22.5,
    "$45 - $100": 72.5,
    "$100+": 100,
    
}

#List out all the feature columns for input
feature_cols = [ 
    'What is your Grade Level?', 
    'Have you ever spent or been asked to spend real money on a video game?', 
    'How often do you make in-game purchases?', 
    'If you have made in-game purchases, around how much money have you spent in total?', 
    'What do you usually buy?', 
    'What is your main reason for spending money in-game?', 
    'Do you feel in game purchases improve your enjoyment when playing games?', 
    'Do you try to budget yourself? How much monthly?', 
    'Have you ever spent more money than expected?', 
    'Which of the following would encourage you the most to buy?'
]

df[target_col] = df[target_col].map(spending_map)

X = df[feature_cols] #Values inputted
y = df[target_col] #Values outputted

#Encode the input values
X = pd.get_dummies(X, drop_first=True)

#Handle any missing values in the data
X = X.fillna(0)
y = y.fillna(y.mean())

#Create the trianing and testing values for the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)


model = RandomForestRegressor(n_estimators = 500, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state = 20)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nMean Absolute Error:", mean_absolute_error(y_test, predictions))
print("\nR² Score:", r2_score(y_test, predictions))

sample = pd.DataFrame([{
    "What grade are you in?_10th": 1,
    "How often do you make in-game purchases?_Sometimes": 1,
    "Have you ever spent money on in-game items?_Yes": 1,
    "Do you usually budget your spending?_Sometimes": 1,
    "Have you ever spent more money than you planned?_Yes": 1
}])

sample = sample.reindex(columns=X.columns, fill_value=0)

print("\nPredicted monthly spending: $",
      round(model.predict(sample)[0], 2))

importances = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

zero_features = importances[importances == 0].index

X_reduced = X.drop(columns= zero_features)

print("\nOriginal shape:", X.shape)
print("\nReduced shape:", X_reduced.shape)

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y,test_size=0.2,random_state=20)

model.fit(X_train, y_train)

preds = model.predict(X_test)

print("\nNew R²:", r2_score(y_test, preds))
print("\nNew MAE:", mean_absolute_error(y_test, preds))

#Create graphs based on the data
plt.figure(figsize=(8, 8))

plt.scatter(y_test, predictions, alpha=0.7)

# Perfect prediction line
min_val = min(min(y_test), min(predictions))
max_val = max(max(y_test), max(predictions))
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.xlabel("Actual Spending ($)")
plt.ylabel("Predicted Spending ($)")
plt.title(f"Actual vs Predicted Spending\nR² = {r2_score(y_test, predictions):.3f}")

plt.tight_layout()
plt.savefig("ml_graph_output/actual_vs_predicted.png")
plt.close()

errors = y_test - predictions

plt.figure(figsize=(8,6))
plt.hist(errors, bins=15)
plt.axvline(0, linestyle="--")
plt.title("Prediction Error Distribution")
plt.xlabel("Error (Actual - Predicted)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("ml_graph_output/error_distribution.png")
plt.close()

top_importances = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(10,8))
top_importances.sort_values().plot(kind="barh")
plt.title("Top 15 Most Important Features")
plt.xlabel("Importance Score")

plt.tight_layout()
plt.savefig("ml_graph_output/feature_importance.png")
plt.close()

plt.figure(figsize=(8,6))
plt.scatter(predictions, errors, alpha=0.7)
plt.axhline(0, linestyle="--")

plt.xlabel("Predicted Spending ($)")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot")

plt.tight_layout()
plt.savefig("ml_graph_output/residual_plot.png")
plt.close()

grade_col = 'What is your Grade Level?'

df["Predicted"] = model.predict(X_reduced)

grade_means = df.groupby(grade_col)[["Predicted"]].mean()

plt.figure(figsize=(8,6))
grade_means.sort_values("Predicted").plot(kind="bar")
plt.ylabel("Average Predicted Monthly Spending ($)")
plt.title("Average Predicted Spending by Grade")

plt.tight_layout()
plt.savefig("ml_graph_output/spending_by_grade.png")
plt.close()


