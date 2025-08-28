import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# For plotting dataset
plt.figure(figsize=(7, 6))
plt.title('Scatter plot of the data')
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Q2 a)
print("\nQ2 a)")
# Convert to DataFrame and Series for compatibility with our tree implementation
X = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
y = pd.Series(y)

# 70-30 train-test split (first 70% as train, last 30% as test)
split_index = int(0.7 * X.shape[0])
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

# Training the Decision Tree
dt_model = DecisionTree(criterion='information_gain', max_depth=5)
dt_model.fit(X_train, y_train)

# Predicting the values
y_pred = dt_model.predict(X_test)

print("Criteria :", "Information Gain")
print("Accuracy :", np.round(accuracy(y_test, y_pred),4))
for cls in np.unique(y_test):
    print(f"Precision for class {cls} :", np.round(precision(y_test, y_pred, cls),4))
    print(f"Recall for class {cls} :", np.round(recall(y_test, y_pred, cls),4))


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Q2 b)
print("\nQ2 b)")
# Manual KFold Cross Validation
print("\nKFold Cross Validation")
num_folds = 5
fold_size = X.shape[0] // num_folds
fold_accuracies = []

for fold_idx in range(num_folds):
    start_idx = fold_idx * fold_size
    end_idx = (fold_idx + 1) * fold_size

    X_valid = X.iloc[start_idx:end_idx]
    y_valid = y.iloc[start_idx:end_idx]

    X_train_cv = pd.concat([X.iloc[:start_idx], X.iloc[end_idx:]])
    y_train_cv = pd.concat([y.iloc[:start_idx], y.iloc[end_idx:]])

    dt_cv = DecisionTree(criterion='information_gain', max_depth=5)
    dt_cv.fit(X_train_cv, y_train_cv)

    y_valid_pred = dt_cv.predict(X_valid)
    
    fold_acc = accuracy(y_valid, y_valid_pred)
    print(f"Fold {fold_idx} Accuracy : {np.round(fold_acc, 4)}")
    fold_accuracies.append(fold_acc)

print("\nMean Accuracy :", np.round(np.mean(fold_accuracies), 4))


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Nested Cross Validation
print("\nNested Cross Validation")
# Using sklearn for fold splits
from sklearn.model_selection import KFold
from itertools import product

# Hyperparameter grid
hyperparameters = dict()
hyperparameters['max_depth'] = [1,2,3,4,5,6,7,8,9,10]
hyperparameters['criteria_values'] = ['information_gain', 'gini_index']

num_outer_folds = 5
num_inner_folds = 5

kf_outer = KFold(n_splits=num_outer_folds, shuffle=False)
kf_inner = KFold(n_splits=num_inner_folds, shuffle=False)

results = {}
outer_fold_counter = 0
entry_counter = 0

# Outer loop (model evaluation)
for outer_train_idx, outer_test_idx in kf_outer.split(X):
    X_outer_train, X_outer_test = X.iloc[outer_train_idx], X.iloc[outer_test_idx]
    y_outer_train, y_outer_test = y.iloc[outer_train_idx], y.iloc[outer_test_idx]
    
    inner_fold_counter = 0
    
    # Inner loop (hyperparameter tuning)
    for inner_train_idx, inner_valid_idx in kf_inner.split(X_outer_train):
        print("Outer Fold {}, Inner Fold {}".format(outer_fold_counter+1, inner_fold_counter+1))
        
        X_inner_train, X_inner_valid = X_outer_train.iloc[inner_train_idx], X_outer_train.iloc[inner_valid_idx]
        y_inner_train, y_inner_valid = y_outer_train.iloc[inner_train_idx], y_outer_train.iloc[inner_valid_idx]
        
        # Grid search over hyperparameters
        for depth_val, criterion_val in product(hyperparameters['max_depth'], hyperparameters['criteria_values']):
            
            # Train with current hyperparameters
            dt_tuned = DecisionTree(max_depth=depth_val, criterion=criterion_val)
            dt_tuned.fit(X_inner_train, y_inner_train)
            
            # Evaluate on inner validation set
            y_inner_pred = dt_tuned.predict(X_inner_valid)
            val_acc = accuracy(y_inner_valid, y_inner_pred)
            
            # Store results
            results[entry_counter] = {
                'outer_fold': outer_fold_counter, 
                'inner_fold': inner_fold_counter, 
                'max_depth': depth_val,  
                'criterion': criterion_val, 
                'val_accuracy': val_acc
            }
            entry_counter += 1
        inner_fold_counter += 1
    outer_fold_counter += 1

# Convert to DataFrame for analysis
overall_results = pd.DataFrame(results).T

# Plot accuracy vs. depth for each outer fold
fig, ax = plt.subplots(1, num_outer_folds, figsize=(20, 5))
if num_outer_folds == 1:
    ax = [ax]

for fold_id in range(num_outer_folds):
    outer_fold_results = overall_results[overall_results['outer_fold'] == fold_id]
    for criterion_val in hyperparameters['criteria_values']:
        criterion_results = outer_fold_results[outer_fold_results['criterion'] == criterion_val]
        if criterion_results.empty:
            continue
        mean_accs = criterion_results.groupby('max_depth')['val_accuracy'].mean()
        ax[fold_id].plot(mean_accs, marker='o', label=criterion_val)
        ax[fold_id].set_title(f'Outer Fold {fold_id}')
        ax[fold_id].set_xlabel('Max Depth')
        ax[fold_id].set_ylabel('Accuracy')
        ax[fold_id].legend()

plt.tight_layout()
plt.show()

# Selecting the best depth across folds
best_depths = {c: [] for c in hyperparameters['criteria_values']}

for fold_id in range(num_outer_folds):
    for criterion_val in hyperparameters['criteria_values']:
        outer_df = overall_results.query(f'outer_fold == {fold_id}')
        top_results = outer_df.groupby(['max_depth', 'criterion']).mean()["val_accuracy"].sort_values(ascending=False)
        best_depths[criterion_val].append(int(top_results.idxmax()[0]))

print("Best Depths: ", best_depths)
print("Mean Best Depth: ")
for criterion_val in hyperparameters['criteria_values']:
    print(f"Criterion: {criterion_val}, Mean Best Depth: {np.mean(best_depths[criterion_val])}") 
