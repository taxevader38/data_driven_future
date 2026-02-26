# ============================================================
# Program: Random Forest Regression Model
# Purpose: Predict estimated monthly in-game spending amount
# Evaluation Metric: Mean Absolute Error, R² Varriance
# ============================================================


# ----------------------------
# File & Library Imports
# ----------------------------

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ----------------------------
# Output Directory Setup
# ----------------------------

# Create a folder to store model outputs (if it doesn't already exist)
os.makedirs("project/ml_graph_output", exist_ok=True)
os.makedirs("project/ml_data_output", exist_ok=True)


# ----------------------------
# Load & Prepare Dataset
# ----------------------------

# Load survey data from CSV file
df = pd.read_csv("project/data/ddf.csv")

# Remove timestamp column (not useful for prediction)
if "Timestamp" in df.columns:
    df = df.drop(columns=["Timestamp"])

# Define target column (what we are trying to predict)
target_col = "How much do you spend monthly on in-game purchases?"


# ----------------------------
# Convert Spending Categories to Numeric Values
# ----------------------------

# Map categorical spending ranges to approximate median dollar values
# This allows us to use regression (numeric prediction)
spending_map = {
    "Nothing": 0,
    "$1 - $10": 5,
    "$15 - $30": 22.5,
    "$45 - $100": 72.5,
    "$100+": 100,
}

df[target_col] = df[target_col].map(spending_map)


# ----------------------------
# Define Features (Inputs)
# ----------------------------

# These columns will be used to predict spending
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

# Separate input features (X) and target variable (y)
X = df[feature_cols]
y = df[target_col]


# ----------------------------
# Encode Categorical Features
# ----------------------------

# Convert categorical responses into numeric dummy variables
# drop_first=True avoids redundant columns
X = pd.get_dummies(X, drop_first=True)

# Fill any missing values
X = X.fillna(0)
y = y.fillna(y.mean())

# ----------------------------
# Train/Test Split
# ----------------------------

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=20
)

# ----------------------------
# Build & Train Model
# ----------------------------

# Initialize Random Forest Regressor with tuned parameters
model = RandomForestRegressor(n_estimators=500,max_depth=10,min_samples_split=5,min_samples_leaf=2,random_state=20)

# Train model on training data
model.fit(X_train, y_train)

# Generate predictions on test data
predictions = model.predict(X_test)

# ----------------------------
# Evaluate Initial Model
# ----------------------------

print("\nInitial Model Performance:")
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("R² Score:", r2_score(y_test, predictions))


# ----------------------------
# Example Prediction
# ----------------------------

# Create a sample student response for testing predictions
sample = pd.DataFrame([{
    "What grade are you in?_10th": 1,
    "How often do you make in-game purchases?_Sometimes": 1,
    "Have you ever spent money on in-game items?_Yes": 1,
    "Do you usually budget your spending?_Sometimes": 1,
    "Have you ever spent more money than you planned?_Yes": 1
}])

# Ensure sample matches training feature columns
sample = sample.reindex(columns=X.columns, fill_value=0)

print("\nPredicted monthly spending: $",
      round(model.predict(sample)[0], 2))


# ----------------------------
# Feature Importance Analysis
# ----------------------------

# Extract feature importance scores
importances = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

# Identify features with zero importance
zero_features = importances[importances == 0].index

# Remove zero-importance features
X_reduced = X.drop(columns=zero_features)

print("\nFeature Reduction:")
print("Original feature count:", X.shape[1])
print("Reduced feature count:", X_reduced.shape[1])


# ----------------------------
# Retrain Model After Feature Reduction
# ----------------------------

# Re-split data using reduced feature set
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=20
)

# Retrain model
model.fit(X_train, y_train)

# New predictions
preds = model.predict(X_test)

print("\nModel Performance After Feature Reduction:")
print("New R²:", r2_score(y_test, preds))
print("New MAE:", mean_absolute_error(y_test, preds))


# ----------------------------
# Save Model Outputs for Graphing
# ----------------------------

# Save test predictions (used for graph generation)
results_df = pd.DataFrame({
    "Grade Level": df.loc[X_test.index, 'What is your Grade Level?'],
    "Actual": y_test,
    "Predicted": preds
})

results_df.to_csv(
    "project/ml_data_output/spending_predictions.csv",
    index=False
)

# Save feature importance values
importances_df = pd.DataFrame({
    "Feature": X_reduced.columns,
    "Importance": model.feature_importances_
})

importances_df.to_csv(
    "project/ml_data_output/feature_importance.csv",
    index=False
)