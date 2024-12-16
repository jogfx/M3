# Import necessary libraries
import numpy as np
import pandas as pd
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, classification_report, recall_score, confusion_matrix,
    roc_auc_score, precision_score, f1_score, roc_curve, auc
)
from sklearn.preprocessing import OrdinalEncoder
from catboost import CatBoostClassifier, Pool

# Load the data
data_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(data_path)

# Convert TotalCharges to numeric, filling NaN values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'], inplace=True)

# Convert SeniorCitizen to object
df['SeniorCitizen'] = df['SeniorCitizen'].astype(object)

# Replace 'No phone service' and 'No internet service' with 'No' for certain columns
df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')
columns_to_replace = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for column in columns_to_replace:
    df[column] = df[column].replace('No internet service', 'No')

# Convert 'Churn' categorical variable to numeric
df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})

# Create the StratifiedShuffleSplit object
strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=64)

train_index, test_index = next(strat_split.split(df, df["Churn"]))

# Create train and test sets
strat_train_set = df.loc[train_index]
strat_test_set = df.loc[test_index]

X_train = strat_train_set.drop("Churn", axis=1)
y_train = strat_train_set["Churn"].copy()

X_test = strat_test_set.drop("Churn", axis=1)
y_test = strat_test_set["Churn"].copy()

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Define a grid of hyperparameters
param_grid = {
    'iterations': [100, 200],
    'learning_rate': [0.01, 0.1],
    'depth': [4, 6, 8, 10, 12],
    'scale_pos_weight': [1, 3]
}

# Perform hyperparameter tuning
best_model = None
best_score = -np.inf
for iterations in param_grid['iterations']:
    for learning_rate in param_grid['learning_rate']:
        for depth in param_grid['depth']:
            for scale_pos_weight in param_grid['scale_pos_weight']:
                model = CatBoostClassifier(
                    verbose=False,
                    random_state=42,
                    iterations=iterations,
                    learning_rate=learning_rate,
                    depth=depth,
                    scale_pos_weight=scale_pos_weight
                )
                model.fit(X_train, y_train, cat_features=categorical_columns, eval_set=(X_test, y_test))
                y_pred = model.predict(X_test)
                f1 = f1_score(y_test, y_pred)
                if f1 > best_score:
                    best_score = f1
                    best_model = model

# Predict on the test set with the best model
y_pred = best_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Create a DataFrame to store results
model_names = ['CatBoost_Model']
result = pd.DataFrame(
    {'Accuracy': [accuracy], 'Recall': [recall], 'Roc_Auc': [roc_auc], 'Precision': [precision], 'F1 Score': [f1]},
    index=model_names
)

# Print results
print(result)

# Save the best model in the 'model' directory
model_dir = "/Users/olive/Documents/BDS/M3/model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, "catboost_best_model.cbm")
best_model.save_model(model_path)
