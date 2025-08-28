import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

# -----------------------------------------------------
# Step 1: Load dataset from UCI repository
# -----------------------------------------------------
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset (Auto MPG dataset, ID=9)
auto_mpg = fetch_ucirepo(id=9) 
  
# separate features and target
features_df = auto_mpg.data.features
target_series = auto_mpg.data.targets 
  
# metadata and variable info
print(auto_mpg.metadata)  
print(auto_mpg.variables)

# join features and target to check for missing/duplicate values
mpg_data = pd.concat([features_df, target_series], axis=1)
print("Shape of extracted data: ", mpg_data.shape)


# -----------------------------------------------------
# Step 2: Data Cleaning / Preprocessing
# -----------------------------------------------------
# replace '?' with NaN
mpg_data.replace('?', np.nan, inplace=True)

print("\nNumber of NaN/Null values in training data:", mpg_data.isnull().sum().sum())

# drop rows with missing values if any
if mpg_data.isnull().sum().sum() > 0:
    mpg_data.dropna(inplace=True)

# check duplicates
print("Number of duplicated samples in training data: ", mpg_data.duplicated().sum())
if mpg_data.duplicated().sum() > 0:
    mpg_data.drop_duplicates(inplace=True)

print("Shape of data after cleaning: ", mpg_data.shape)


# -----------------------------------------------------
# Step 3: Split into Features and Target
# -----------------------------------------------------
X_features = mpg_data.drop('mpg', axis=1)   # independent variables
y_target = mpg_data['mpg']                  # dependent variable

print("\nX shape:", X_features.shape)
print("y shape:", y_target.shape)


# -----------------------------------------------------
# Step 4: Train-Test Split
# -----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_target, test_size=0.3, random_state=42
)

print("\nX_train size: ", X_train.shape)
print("y_train size: ", y_train.shape)
print("X_test size: ", X_test.shape)
print("y_test size: ", y_test.shape)


# -----------------------------------------------------
# Step 5: Our Custom Decision Tree
# -----------------------------------------------------
my_dt = DecisionTree(criterion="information_gain", max_depth=5)
my_dt.fit(X_train, y_train)

# predictions
y_train_pred_custom = my_dt.predict(X_train)
y_test_pred_custom = my_dt.predict(X_test)

# evaluate performance on training data
train_rmse = rmse(y_train_pred_custom, y_train)
train_mae = mae(y_train_pred_custom, y_train)

print("\nTrain Metrics (Custom):")
print(f"    Root Mean Squared Error: {train_rmse:.4f}")
print(f"    Mean Absolute Error: {train_mae:.4f}")

# evaluate performance on test data
test_rmse = rmse(y_test_pred_custom, y_test)
test_mae = mae(y_test_pred_custom, y_test)

print("\nTest Metrics (Custom):")
print(f"    Root Mean Squared Error: {test_rmse:.4f}")
print(f"    Mean Absolute Error: {test_mae:.4f}")


# -----------------------------------------------------
# Step 6: Scikit-Learn Decision Tree
# -----------------------------------------------------
sklearn_dt = DecisionTreeRegressor(max_depth=5)
sklearn_dt.fit(X_train, y_train)

# predictions
y_train_pred_sklearn = sklearn_dt.predict(X_train)
y_test_pred_sklearn = sklearn_dt.predict(X_test)

# evaluate training performance
train_rmse_sklearn = rmse(pd.Series(y_train_pred_sklearn), y_train)
train_mae_sklearn = mae(pd.Series(y_train_pred_sklearn), y_train)

print("\nTrain Metrics (Sklearn):")
print(f"    Root Mean Squared Error: {train_rmse_sklearn:.4f}")
print(f"    Mean Absolute Error: {train_mae_sklearn:.4f}")

# evaluate testing performance
test_rmse_sklearn = rmse(pd.Series(y_test_pred_sklearn), y_test)
test_mae_sklearn = mae(pd.Series(y_test_pred_sklearn), y_test)

print("\nTest Metrics (Sklearn):")
print(f"    Root Mean Squared Error: {test_rmse_sklearn:.4f}")
print(f"    Mean Absolute Error: {test_mae_sklearn:.4f}")


# -----------------------------------------------------
# Step 7: Final Comparison
# -----------------------------------------------------
print("\nPerformance Comparison:\n")
print(f"Our Decision Tree - Train RMSE: {train_rmse:.4f}")
print(f"Our Decision Tree - Test RMSE: {test_rmse:.4f}")
print(f"Scikit-Learn Decision Tree - Train RMSE: {train_rmse_sklearn:.4f}")
print(f"Scikit-Learn Decision Tree - Test RMSE: {test_rmse_sklearn:.4f}")
