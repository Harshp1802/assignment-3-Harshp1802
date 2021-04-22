# ES654-2021 Assignment 3

*Harsh Patel* - *18110062*

------

- In this question I have implemented the non-vectorised and the Autograd version of the Gradient Descent Algorithm for Multi Class Logistic Regression.
- When fit_intercept is set to True, a new column is of 1s is added to the dataset which incorporates the theta-0 in our fitting.
- Using Digits dataset and stratified (K=4 folds), I have shown the visualalisation of the confusion matrix the overall accuracy. I have also inferred some important details like most confusing digits and the easiest digits to predict.

+ Results: 

## |--------- Multi-class Logistic Regression using self-update rules ----------|
``` 
Accuracy:  0.9425925925925925 
```

## |--------- Multi-class Logistic Regression using Autograd ----------|
```
Accuracy:  0.9425925925925925
```

## |--------- 4-Folds Multi-class Logistic Regression over DIGITS ----------|
```
Test_fold = 1
         Test_Accuracy: 0.9064587973273942
Test_fold = 2
         Test_Accuracy: 0.89086859688196
Test_fold = 3
         Test_Accuracy: 0.9042316258351893
Test_fold = 4
         Test_Accuracy: 0.933184855233853
AVERAGE ACCURACY = 0.9086859688195991
```
## -------- Best Confusion Matrix --------

<p align = center>
<img src = " .\q3_confusion_matrix.jpg" >
</p>

### - Here we observe that in the confusion matrix **1 & 8** and **1 & 9** are the most confusing digits as they have the maximum misclassifications (4) of each other!

### - We also observe that the digit 0 is the easiet to predict as it has the minimum no. of missclassified entries. [See the diagonal]. Digit 8 is the toughest to predict!

## -------- Principal Component Analysis (PCA) for DIGITS --------

<p align = center>
<img src = " .\PCA_DIGITS.jpg" >
</p>

### From this PCA plot we can infer that:
- Some of the digits are very easliy linearly separable from the other digits and hence our logistic regression is a very apt approach to solve this problem of classification.

- We verify our results from the confusion matrix from this plot as we can see that 0 is the easiest cluster and 8 is the most difficult cluster to classify which matches with our results from the confusion matrix.