## Decision Tree Implementation

### Q1. Complete the decision tree implementation in tree/base.py. The code should be written in Python and not use existing libraries other than the ones shared in class or already imported in the code. Your decision tree should work for four cases: i) discrete features, discrete output; ii) discrete features, real output; iii) real features, discrete output; iv) real features, real output. Your model should accept real inputs only (for discrete inputs, you may convert the attributes into one-hot encoded vectors). Your decision tree should be able to use InformationGain using Entropy or GiniIndex as the criteria for splitting for discrete output. Your decision tree should be able to use InformationGain using MSE as the criteria for splitting for real output. Your code should also be able to plot/display the decision tree. 
Ans: We have successfully implemented the decision tree in tree/base.py as per the assignment requirements. The decision tree is capable of handling the following cases:

Discrete features → Discrete output

Discrete features → Real output

Real features → Discrete output

Real features → Real output

Completed files:

tree/base.py – Contains the completed DecisionTree class

tree/utils.py – Provides utility functions required for the decision tree

metrics.py – Implements all necessary performance metric functions

Our implementation also supports displaying the decision tree structure in a properly formatted text format, as specified in tree/base.py. To verify the implementation and view the results, please run the usage.py file.


### Q2. Generate your dataset using the following lines of code. 
### a) Show the usage of your decision tree on the above dataset. The first 70% of the data should be used for training purposes and the remaining 30% for test purposes. Show the accuracy, per-class precision and recall of the decision tree you implemented on the test dataset. 
### b) Use 5 fold cross-validation on the dataset. Using nested cross-validation find the optimum depth of the tree. 

Ans : We created the dataset using the code provided. This is the Scatter plot of the data.    
Shape of X: (100, 2)   
Shape of y: (100,)

![alt text](/Figure_1.png)

Now we have split the data as 70% for training and remaining 30% for testing purposes.     
Shape of X_train: (70, 2)   
Shape of X_test: (30, 2)   
Shape of y_train: (70,)  
Shape of y_test: (30,)  

Here is the accuracy, per-class precision and recall of the decision tree we implemented on the test dataset.   

Criteria : Information Gain   
Accuracy : 0.8667   
Precision for class 0 : 0.75   
Recall for class 0 : 0.9   
Precision for class 1 : 0.9444   
Recall for class 1 : 0.85

KFold Cross Validation

Fold 0 Accuracy : 0.8
Fold 1 Accuracy : 0.9
Fold 2 Accuracy : 0.85
Fold 3 Accuracy : 0.8
Fold 4 Accuracy : 0.95

Mean Accuracy : 0.86
 


Now, we calculate the optimal depth for the tree according to each criteria :   
Criterion: information_gain, Mean Best Depth: 2.0   
Criterion: gini_index, Mean Best Depth: 2.0   

From the above data, we can conclude that 2 is the most optimal depth for the decision tree.

### Q3. a) Show the usage of your decision tree for the automotive efficiency problem. b) Compare the performance of your model with the decision tree module from scikit learn.
Ans : First, we extracted the auto-mpg data.   
Shape of extracted data:  (398, 8)
Then we processed it and split it into train and test dataset. Then we calculated RMSE using our custom decision tree implementation and also the SkLearn's Decision Tree Regressor. Here is the Performance Comparison:   

Our Decision Tree - Train RMSE: 5.7899   
Our Decision Tree - Test RMSE: 7.1421   
Scikit-Learn Decision Tree - Train RMSE: 1.9938   
Scikit-Learn Decision Tree - Test RMSE: 3.2169   

Comparison of Models:   
The performance comparison shows that the Scikit-Learn decision tree outperforms our custom decision tree implementation.   
Our custom decision tree achieved a Train RMSE of 5.7899, while the Scikit-Learn decision tree achieved a significantly lower Train RMSE of 1.9938.   
On the test set, our model resulted in a Test RMSE of 7.1421, compared to 3.2169 from the Scikit-Learn model.    

This shows that the Scikit-Learn model is better at fitting the training data, likely due to more optimized algorithms and better-handling of edge cases. The higher test RMSE for our model indicates that it doesn't generalize to unseen test data as effectively as Scikit-Learn decision tree.

Our implementation, while educational, still provides a relatively reasonable fit and could benefit from further refinement to improve accuracy and reduce overfitting.

### Q4. Create some fake data to do some experiments on the runtime complexity of your decision tree algorithm. Create a dataset with N samples and M binary features. Vary M and N to plot the time taken for : 1) learning the tree, 2) predicting for test data. How do these results compare with theoretical time complexity for decision tree creation and prediction. You should do the comparison for all the four cases of decision trees.

Ans : 

![alt text](/Figure-2.png)
![alt text](/Figure-3.png)

From the graphs, we can observe that prediction times for datasets with real-valued inputs are significantly higher compared to those with discrete inputs.

The depth of a decision tree is typically O(log N), though in the worst case it can grow to O(N).

Training Time:
The theoretical complexity of training a decision tree is O(M·N·log N), where N is the number of samples and M is the number of features. The experimental results align well with this expectation—training time increases almost linearly with both M and N, which is consistent with the O(M·N·log N) complexity (since N·log N often appears nearly linear in practice).

Prediction Time:
The theoretical complexity of predicting a single sample is proportional to the depth of the tree. For balanced trees, this is typically O(log N). In practice, the actual cost depends on the depth of the leaf where the prediction is made:

If the prediction path ends at a shallow leaf, the time is lower.

If it reaches a deeper leaf, the time is higher.

The experimental results reflect this behavior:

Prediction time increases with N, generally following a log N trend.

Prediction time remains almost constant with respect to M, suggesting independence from the number of features.

Conclusion:
The observed experimental trends are consistent with the theoretical complexities. Training scales as O(M·N·log N), while prediction scales approximately as O(log N) with little to no dependence on M.



Here is the data -

Evaluating for N=50, M=1

    Input Type: real, Output Type: real
    Criteria: information_gain
        Average Training Time: 0.1296 seconds (std: 0.0032)
        Average Prediction Time: 0.0015 seconds (std: 0.0001)
    Criteria: gini_index
        Average Training Time: 0.1871 seconds (std: 0.0051)
        Average Prediction Time: 0.0015 seconds (std: 0.0000)

    Input Type: real, Output Type: discrete
    Criteria: information_gain
        Average Training Time: 0.0002 seconds (std: 0.0002)
        Average Prediction Time: 0.0002 seconds (std: 0.0000)
    Criteria: gini_index
        Average Training Time: 0.0002 seconds (std: 0.0000)
        Average Prediction Time: 0.0002 seconds (std: 0.0000)

    Input Type: discrete, Output Type: real
    Criteria: information_gain
        Average Training Time: 0.0024 seconds (std: 0.0014)
        Average Prediction Time: 0.0008 seconds (std: 0.0004)
    Criteria: gini_index
        Average Training Time: 0.0026 seconds (std: 0.0001)
        Average Prediction Time: 0.0007 seconds (std: 0.0000)

    Input Type: discrete, Output Type: discrete
    Criteria: information_gain
        Average Training Time: 0.0002 seconds (std: 0.0000)
        Average Prediction Time: 0.0002 seconds (std: 0.0000)
    Criteria: gini_index
        Average Training Time: 0.0002 seconds (std: 0.0000)
        Average Prediction Time: 0.0002 seconds (std: 0.0000)

==================================================

Evaluating for N=50, M=5

    Input Type: real, Output Type: real
    Criteria: information_gain
        Average Training Time: 0.3035 seconds (std: 0.0157)
        Average Prediction Time: 0.0015 seconds (std: 0.0001)
    Criteria: gini_index
        Average Training Time: 0.4299 seconds (std: 0.0061)
        Average Prediction Time: 0.0015 seconds (std: 0.0001)

    Input Type: real, Output Type: discrete
    Criteria: information_gain
        Average Training Time: 0.5988 seconds (std: 0.0127)
        Average Prediction Time: 0.0014 seconds (std: 0.0001)
    Criteria: gini_index
        Average Training Time: 0.4794 seconds (std: 0.0158)
        Average Prediction Time: 0.0013 seconds (std: 0.0000)

    Input Type: discrete, Output Type: real
    Criteria: information_gain
        Average Training Time: 0.0526 seconds (std: 0.0036)
        Average Prediction Time: 0.0023 seconds (std: 0.0001)
    Criteria: gini_index
        Average Training Time: 0.0653 seconds (std: 0.0030)
        Average Prediction Time: 0.0022 seconds (std: 0.0003)

    Input Type: discrete, Output Type: discrete
    Criteria: information_gain
        Average Training Time: 0.0946 seconds (std: 0.0052)
        Average Prediction Time: 0.0023 seconds (std: 0.0001)
    Criteria: gini_index
        Average Training Time: 0.0720 seconds (std: 0.0025)
        Average Prediction Time: 0.0022 seconds (std: 0.0001)

==================================================

Evaluating for N=50, M=10

    Input Type: real, Output Type: real
    Criteria: information_gain
        Average Training Time: 0.5106 seconds (std: 0.0065)
        Average Prediction Time: 0.0014 seconds (std: 0.0000)
    Criteria: gini_index
        Average Training Time: 0.7355 seconds (std: 0.0125)
        Average Prediction Time: 0.0014 seconds (std: 0.0001)

    Input Type: real, Output Type: discrete
    Criteria: information_gain
        Average Training Time: 1.0217 seconds (std: 0.0245)
        Average Prediction Time: 0.0013 seconds (std: 0.0000)
    Criteria: gini_index
        Average Training Time: 0.8635 seconds (std: 0.0239)
        Average Prediction Time: 0.0014 seconds (std: 0.0000)

    Input Type: discrete, Output Type: real
    Criteria: information_gain
        Average Training Time: 0.0908 seconds (std: 0.0044)
        Average Prediction Time: 0.0023 seconds (std: 0.0005)
    Criteria: gini_index
        Average Training Time: 0.1107 seconds (std: 0.0073)
        Average Prediction Time: 0.0024 seconds (std: 0.0001)

    Input Type: discrete, Output Type: discrete
    Criteria: information_gain
        Average Training Time: 0.1734 seconds (std: 0.0068)
        Average Prediction Time: 0.0024 seconds (std: 0.0007)
    Criteria: gini_index
        Average Training Time: 0.1378 seconds (std: 0.0065)
        Average Prediction Time: 0.0022 seconds (std: 0.0001)

==================================================

Evaluating for N=50, M=20

    Input Type: real, Output Type: real
    Criteria: information_gain
        Average Training Time: 0.9784 seconds (std: 0.0186)
        Average Prediction Time: 0.0016 seconds (std: 0.0005)
    Criteria: gini_index
        Average Training Time: 1.3434 seconds (std: 0.0161)
        Average Prediction Time: 0.0014 seconds (std: 0.0000)

    Input Type: real, Output Type: discrete
    Criteria: information_gain
        Average Training Time: 1.8722 seconds (std: 0.0229)
        Average Prediction Time: 0.0014 seconds (std: 0.0000)
    Criteria: gini_index
        Average Training Time: 1.5865 seconds (std: 0.0326)
        Average Prediction Time: 0.0015 seconds (std: 0.0000)

    Input Type: discrete, Output Type: real
    Criteria: information_gain
        Average Training Time: 0.1568 seconds (std: 0.0047)
        Average Prediction Time: 0.0023 seconds (std: 0.0000)
    Criteria: gini_index
        Average Training Time: 0.2104 seconds (std: 0.0051)
        Average Prediction Time: 0.0024 seconds (std: 0.0000)

    Input Type: discrete, Output Type: discrete
    Criteria: information_gain
        Average Training Time: 0.2845 seconds (std: 0.0072)
        Average Prediction Time: 0.0023 seconds (std: 0.0003)
    Criteria: gini_index
        Average Training Time: 0.2252 seconds (std: 0.0055)
        Average Prediction Time: 0.0023 seconds (std: 0.0000)

==================================================

Evaluating for N=50, M=50

    Input Type: real, Output Type: real
    Criteria: information_gain
        Average Training Time: 2.2646 seconds (std: 0.0249)
        Average Prediction Time: 0.0015 seconds (std: 0.0000)
    Criteria: gini_index
        Average Training Time: 3.1730 seconds (std: 0.0349)
        Average Prediction Time: 0.0015 seconds (std: 0.0000)

    Input Type: real, Output Type: discrete
    Criteria: information_gain
        Average Training Time: 4.5647 seconds (std: 0.0487)
        Average Prediction Time: 0.0014 seconds (std: 0.0000)
    Criteria: gini_index
        Average Training Time: 3.8924 seconds (std: 0.0629)
        Average Prediction Time: 0.0012 seconds (std: 0.0000)

    Input Type: discrete, Output Type: real
    Criteria: information_gain
        Average Training Time: 0.3930 seconds (std: 0.0130)
        Average Prediction Time: 0.0027 seconds (std: 0.0001)
    Criteria: gini_index
        Average Training Time: 0.5094 seconds (std: 0.0120)
        Average Prediction Time: 0.0026 seconds (std: 0.0001)

    Input Type: discrete, Output Type: discrete
    Criteria: information_gain
        Average Training Time: 0.7780 seconds (std: 0.0331)
        Average Prediction Time: 0.0027 seconds (std: 0.0001)
    Criteria: gini_index
        Average Training Time: 0.6181 seconds (std: 0.0247)
        Average Prediction Time: 0.0028 seconds (std: 0.0001)
