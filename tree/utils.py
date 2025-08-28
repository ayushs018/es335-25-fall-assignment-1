"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import numpy as np
import pandas as pd


def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    for col in X.columns:
        # Apply one-hot encoding only for categorical attributes with more than 2 categories
        if (not check_ifreal(X[col])) & (len(X[col].unique()) > 2):
            dummy_vars = pd.get_dummies(X[col], prefix=col)  # Create dummy variables
            X = pd.concat([X, dummy_vars], axis=1)           # Append new dummy columns
            X.drop(col, axis=1, inplace=True)                # Drop original categorical column
    return X



def check_ifreal(y: pd.Series, real_distinct_threshold: int = 15) -> bool:
    """
    Function to check if the given series has real or discrete values

    Returns True if the series has real (continuous) values, False otherwise (discrete).
    """
    # Categorical dtype is discrete
    if pd.api.types.is_categorical_dtype(y):
        return False
    # Boolean dtype is discrete
    if pd.api.types.is_bool_dtype(y):
        return False
    # Float dtype is continuous
    if pd.api.types.is_float_dtype(y):
        return True
    # Integer dtype: check number of distinct values
    if pd.api.types.is_integer_dtype(y):
        return len(y.unique()) > real_distinct_threshold
    # Strings are discrete
    if pd.api.types.is_string_dtype(y):
        return False
    return False



def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy

    entropy = -sum(p_i * log2(p_i))
    """
    counts = Y.value_counts()         # Frequency of each class
    total = Y.size                    # Total samples
    probabilities = counts / total    # Probability of each class
    entropy_val = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Add epsilon to avoid log(0)
    return entropy_val



def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index

    gini_index = 1 - sum(p_i^2)
    """
    counts = Y.value_counts()        # Frequency of each class
    total = Y.size                   # Total samples
    probabilities = counts / total   # Probability of each class
    gini_val = 1 - np.sum(probabilities ** 2)
    return gini_val



def mse(Y: pd.Series) -> float:
    """
    Function to calculate the mean squared error (mse)

    mse = sum((y_i - y_mean)^2) / n
    """
    mean_val = Y.mean()                         # Mean of values
    mse_val = np.sum((Y - mean_val) ** 2) / Y.size
    return mse_val



def check_criteria(Y: pd.Series, criterion: str):
    """
    Function to check if the criterion is valid
    """
    # Choose criterion based on target type
    if criterion == "information_gain":
        if check_ifreal(Y):
            chosen_criterion = 'mse'      # Regression → MSE
        else:
            chosen_criterion = 'entropy'  # Classification → Entropy
    elif criterion == "gini_index":
        chosen_criterion = 'gini_index'
    else:
        raise ValueError("Criterion must be 'information_gain' or 'gini_index'.")
    
    # Map criterion to function
    criterion_map = {
        'entropy': entropy,
        'gini_index': gini_index,
        'mse': mse
    }
    return chosen_criterion, criterion_map[chosen_criterion]



def find_optimal_threshold(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to find the optimal threshold for a real feature

    Returns the threshold value for best split in a given real feature
    """
    chosen_criterion, criterion_func = check_criteria(Y, criterion)

    sorted_values = attr.sort_values()
    # If only one or two values, splitting is trivial
    if sorted_values.size == 1:
        return None
    elif sorted_values.size == 2:
        return (sorted_values.sum()) / 2
    
    # Candidate split points: midpoints between consecutive values
    candidate_splits = (sorted_values[:-1] + sorted_values[1:]) / 2

    best_thresh = None
    best_gain = -np.inf

    for split in candidate_splits:
        Y_left = Y[attr <= split]
        Y_right = Y[attr > split]

        if Y_left.empty or Y_right.empty:
            continue

        weighted_criterion = (Y_left.size / Y.size) * criterion_func(Y_left) + \
                             (Y_right.size / Y.size) * criterion_func(Y_right)
        info_gain = criterion_func(Y) - weighted_criterion

        if info_gain > best_gain:
            best_thresh = split
            best_gain = info_gain

    return best_thresh



def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)

    information_gain = criterion(Y) - sum((Y_i.size / Y.size) * criterion(Y_i))
    """
    chosen_criterion, criterion_func = check_criteria(Y, criterion)

    # Case 1: Continuous attribute
    if check_ifreal(attr):
        threshold = find_optimal_threshold(Y, attr, criterion)
        if threshold is None:
            return 0  # No valid threshold found
        Y_left = Y[attr <= threshold]
        Y_right = Y[attr > threshold]
        info_gain = criterion_func(Y) - (
            (Y_left.size / Y.size) * criterion_func(Y_left) +
            (Y_right.size / Y.size) * criterion_func(Y_right)
        )
        return info_gain
    
    # Case 2: Categorical attribute
    weighted_criterion = 0
    for val in attr.unique():
        subset = Y[attr == val]
        weighted_criterion += (subset.size / Y.size) * criterion_func(subset)
    info_gain = criterion_func(Y) - weighted_criterion
    return info_gain



def opt_split_attribute(X: pd.DataFrame, y: pd.Series, features: pd.Series, criterion: str) -> str:
    """
    Function to find the optimal attribute to split about.
    """
    best_attr = None
    best_score = -np.inf
    for feat in features:
        score = information_gain(y, X[feat], criterion)
        if score > best_score:
            best_attr = feat
            best_score = score
    return best_attr



def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    """
    # Split depending on attribute type
    if check_ifreal(X[attribute]):
        X_left = X[X[attribute] <= value]
        X_right = X[X[attribute] > value]
    else:
        X_left = X[X[attribute] == value]
        X_right = X[X[attribute] != value]

    # Get corresponding labels
    y_left = y.loc[X_left.index]
    y_right = y.loc[X_right.index]

    return X_left, y_left, X_right, y_right
