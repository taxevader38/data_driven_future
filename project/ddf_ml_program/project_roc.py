# ============================================================
# Program: Random Forest Classification Model
# Purpose: Predict whether a student spends money (binary)
# Evaluation Metric: ROC-AUC
# ============================================================


# ----------------------------
# Library Imports
# ----------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve


# ----------------------------
# Load & Prepare Dataset
# ----------------------------

# Load survey data from CSV file
df = pd.read_csv("project/data/ddf.csv")

# Remove timestamp column (not useful for prediction)
if "Timestamp" in df.columns:
    df = df.drop(columns=["Timestamp"])

# Define target column (original categorical spending response)
target_col = "How much do you spend monthly on in-game purchases?"


# ----------------------------
# Define Feature Columns
# ----------------------------

# These survey responses will be used as predictive inputs
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

# Separate input features (X)
X = df[feature_cols]

# Convert categorical survey responses into numeric dummy variables
X = pd.get_dummies(X, drop_first=True)

# Fill missing values with 0 to prevent training errors
X = X.fillna(0)


# ----------------------------
# Prepare Binary Target Variable
# ----------------------------

# Original target is categorical spending range.
# Convert to binary:
# 0 = "Nothing"
# 1 = Any spending amount

y = df[target_col]
y_binary = (y != "Nothing").astype(int)


# ----------------------------
# Train/Test Split
# ----------------------------

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=20
)


# ----------------------------
# Build & Train Initial Model
# ----------------------------

# Initialize Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    random_state=20
)

# Train model on training data
model.fit(X_train, y_train)

# Generate predicted probabilities for class 1 (spender)
y_probs = model.predict_proba(X_test)[:, 1]


# ----------------------------
# Evaluate Initial Model
# ----------------------------

# Calculate ROC-AUC score
roc_score = roc_auc_score(y_test, y_probs)

print("Initial ROC-AUC Score:", round(roc_score, 3))


# ----------------------------
# Feature Importance Analysis
# ----------------------------

# Extract feature importance scores
importances = pd.Series(
    model.feature_importances_,
    index=X.columns
)

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
    X_reduced, y_binary, test_size=0.2, random_state=20
)

# Retrain classifier
model.fit(X_train, y_train)

# Generate new probability predictions
new_y_probs = model.predict_proba(X_test)[:, 1]

# Calculate new ROC-AUC score
new_roc_score = roc_auc_score(y_test, new_y_probs)

print("\nROC-AUC Score After Feature Reduction:", round(new_roc_score, 3))


# ----------------------------
# Generate ROC Curve Data
# ----------------------------

# Compute ROC curve values for original model
fpr1, tpr1, _ = roc_curve(y_test, y_probs)

# Compute ROC curve values for reduced-feature model
fpr2, tpr2, _ = roc_curve(y_test, new_y_probs)


# ----------------------------
# Save ROC Data for Graphing
# ----------------------------

# Save original ROC curve data
roc_df = pd.DataFrame({
    "FPR": fpr1,
    "TPR": tpr1,
    "AUC": roc_score
})

roc_df.to_csv(
    "project/ml_data_output/roc_data.csv",
    index=False
)

# Save cleaned (feature-reduced) ROC curve data
cleaned_roc_df = pd.DataFrame({
    "FPR": fpr2,
    "TPR": tpr2,
    "AUC": new_roc_score
})

cleaned_roc_df.to_csv(
    "project/ml_data_output/cleaned_roc_data.csv",
    index=False
)