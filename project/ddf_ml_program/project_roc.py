import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve

# Load dataset
df = pd.read_csv("project/data/ddf.csv")

if "Timestamp" in df.columns:
    df = df.drop(columns=["Timestamp"])

target_col = "How much do you spend monthly on in-game purchases?"

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

X = df[feature_cols]
X = pd.get_dummies(X, drop_first=True)
X = X.fillna(0)

# Keep categorical target
y = df[target_col]

# Convert to binary (simpler for AP explanation)
y_binary = (y != "Nothing").astype(int)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=20
)

# Train classifier
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    random_state=20
)

model.fit(X_train, y_train)

# Get probability of class 1 (spender)
y_probs = model.predict_proba(X_test)[:, 1]

# Calculate ROC-AUC
roc_score = roc_auc_score(y_test, y_probs)

print("ROC-AUC Score:", round(roc_score, 3))

importances = pd.Series(
    model.feature_importances_,
    index=X.columns
)

zero_features = importances[importances == 0].index

X_reduced = X.drop(columns=zero_features)

print("\nOriginal feature count:", X.shape[1])
print("\nReduced feature count:", X_reduced.shape[1])

# Re-split and retrain
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_binary, test_size=0.2, random_state=20)

model.fit(X_train, y_train)

new_y_probs = model.predict_proba(X_test)[:, 1]

new_roc_score = roc_auc_score(y_test, new_y_probs)

print("\nNew ROC-AUC Score:", round(new_roc_score, 3))

fpr1, tpr1, _ = roc_curve(y_test, y_probs)

fpr2, tpr2, _ = roc_curve(y_test, new_y_probs)

roc_df = pd.DataFrame({
    "FPR": fpr1,
    "TPR": tpr1
})

cleaned_roc_df = pd.DataFrame({
    "FPR": fpr2,
    "TPR": tpr2
})

roc_df["AUC"] = roc_score
roc_df.to_csv("project/ml_data_output/roc_data.csv", index=False)

cleaned_roc_df["CLEAN_AUC"] = new_roc_score
cleaned_roc_df.to_csv("project/ml_data_output/cleaned_roc_data.csv", index=False)