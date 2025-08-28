"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
import pandas as pd
from tree.utils import *

np.random.seed(42)


@dataclass
class DecisionNode:
    attribute: str
    value: float
    left: "DecisionNode"
    right: "DecisionNode"
    is_leaf: bool
    output: Union[str, float]
    criterion_pair: tuple
    gain: float

    def __init__(self, attribute=None, value=None, left=None, right=None, is_leaf=False, output=None, criterion_pair=None, gain=0):
        # Node properties
        self.attribute = attribute              # Attribute used for split
        self.value = value                      # Threshold/category for split
        self.left = left                        # Left subtree
        self.right = right                      # Right subtree
        self.is_leaf = is_leaf                  # Whether node is leaf
        self.output = output                    # Prediction at leaf
        self.criterion_pair = criterion_pair    # Criterion details (name, value)
        self.gain = gain                        # Info gain at this split
    
    def is_leaf_node(self):
        return self.is_leaf



class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # Criterion (classification only)
    max_depth: int  # Maximum depth allowed

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None  # Root node of tree


    def fit(self, X: pd.DataFrame, y: pd.Series, depth: int = 0) -> None:
        """
        Function to train and construct the decision tree
        """

        def build_tree(X_data: pd.DataFrame, y_data: pd.Series, curr_depth: int) -> DecisionNode:
            """
            Recursive function to build the decision tree.
            """
            # Get criterion function (entropy/gini/mse)
            crit_name, crit_func = check_criteria(y_data, self.criterion)
            crit_value = crit_func(y_data)
            crit_pair = (crit_name, crit_value)     

            # Stop if max depth reached OR only one class/value remains
            if curr_depth >= self.max_depth or y_data.nunique() == 1:
                if check_ifreal(y_data):  # Regression → predict mean
                    return DecisionNode(is_leaf=True, output=np.round(y_data.mean(), 4), criterion_pair=crit_pair)
                else:  # Classification → predict most frequent class
                    return DecisionNode(is_leaf=True, output=y_data.mode()[0], criterion_pair=crit_pair)
            
            # Find best attribute to split on
            best_feature = opt_split_attribute(X_data, y_data, X_data.columns, self.criterion)
            if best_feature is None:  # No valid split → leaf node
                if check_ifreal(y_data):
                    return DecisionNode(is_leaf=True, output=np.round(y_data.mean(), 4), criterion_pair=crit_pair)
                else:
                    return DecisionNode(is_leaf=True, output=y_data.mode()[0], criterion_pair=crit_pair)

            # Handle real-valued attributes
            if check_ifreal(X_data[best_feature]):
                split_threshold = find_optimal_threshold(y_data, X_data[best_feature], self.criterion)
                if split_threshold is None:  # If no threshold found → leaf
                    if check_ifreal(y_data):
                        return DecisionNode(is_leaf=True, output=np.round(y_data.mean(), 4), criterion_pair=crit_pair)
                    else:
                        return DecisionNode(is_leaf=True, output=y_data.mode()[0], criterion_pair=crit_pair)
            else:
                # For categorical features → choose most frequent category
                split_threshold = X_data[best_feature].mode()[0]

            # Split data into left & right subsets
            X_left, y_left, X_right, y_right = split_data(X_data, y_data, best_feature, split_threshold)
            if X_left.empty or X_right.empty:  # If one side empty → leaf
                if check_ifreal(y_data):
                    return DecisionNode(is_leaf=True, output=np.round(y_data.mean(), 4), criterion_pair=crit_pair)
                else:
                    return DecisionNode(is_leaf=True, output=y_data.mode()[0], criterion_pair=crit_pair)
                
            # Recursively build left and right subtrees
            left_branch = build_tree(X_left, y_left, curr_depth + 1)
            right_branch = build_tree(X_right, y_right, curr_depth + 1)

            # Return internal decision node
            return DecisionNode(
                attribute=best_feature, 
                value=split_threshold, 
                left=left_branch, 
                right=right_branch, 
                criterion_pair=crit_pair, 
                gain=information_gain(y_data, X_data[best_feature], self.criterion)
            )

        # Build tree from root
        self.tree = build_tree(X, y, depth)
    

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        def predict_single_row(x_row: pd.Series) -> float:
            """
            Traverse the tree for a single row to make prediction.
            """
            node = self.tree
            while not node.is_leaf_node():
                # Real-valued feature → compare with threshold
                if check_ifreal(pd.Series([x_row[node.attribute]])):
                    if x_row[node.attribute] <= node.value:
                        node = node.left
                    else:
                        node = node.right
                else:
                    # Categorical feature → check equality
                    if x_row[node.attribute] == node.value:
                        node = node.left
                    else:
                        node = node.right
            return node.output 
        
        # Apply prediction row-wise
        return pd.Series([predict_single_row(row) for _, row in X.iterrows()])


    def plot(self, path=None) -> None:
        """
        Function to print the tree structure
        """
        if not self.tree:
            print("Tree not trained yet")
            return

        print("\nDecision Tree Structure:")
        print(self.print_tree())
            
    
    def print_tree(self) -> str:
        """
        Helper function to recursively print tree nodes.
        """
        def print_node(node: DecisionNode, indent: str = '') -> str:
            output_str = ''
            if node.is_leaf:
                output_str += f'Class: {node.output}\n'
            else:
                output_str += f'?(attr {node.attribute} <= {node.value:.2f})\n'
                output_str += indent + '    Yes: '
                output_str += print_node(node.left, indent + '    ')
                output_str += indent + '    No: '
                output_str += print_node(node.right, indent + '    ')
            return output_str

        if not self.tree:
            return "Tree not trained yet"
        else:
            return print_node(self.tree)


    def __repr__(self):
        return f"DecisionTree(criterion={self.criterion}, max_depth={self.max_depth})\n\nTree Structure:\n{self.print_tree()}"
