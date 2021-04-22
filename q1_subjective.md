# ES654-2021 Assignment 3

*Harsh Patel* - *18110062*

------

- In this question I have implemented the non-Vectorised version and the Autograd version of the Gradient Descent Algorithm for unregularised Logistic Regression.

- When fit_intercept is set to True, a new column is of 1s is added to the dataset which incorporates the theta-0 in our fitting.

- Used breast cancer dataset and K=3 folds and shown the overall accuracy.

+ Results:

## |--------- Unregularised Logistic Regression ----------|
```

Obtained THETA values: [ 1.22543988  1.65893187 -4.5332299 ]
Accuracy:  0.8
```
<p align = center>
<img src = ".\q1_plots\db_non_vectorised.jpg" >
</p>

## |--------- Unregularised Logistic Regression using Autograd ----------|
```

Obtained THETA values: [ 1.21784697  1.66523405 -4.52000176]
Accuracy_autograd:  0.8
```
<p align = center>
<img src = ".\q1_plots\db_autograd.jpg" >
</p>

## |--------- 3-Fold Unregularised Logistic Regression on Breast Cancer Dataset ----------|
```

Test_fold = 1
         Test_Accuracy: 0.8465608465608465
Test_fold = 2
         Test_Accuracy: 0.8148148148148148
Test_fold = 3
         Test_Accuracy: 0.8994708994708994
AVERAGE ACCURACY = 0.853615520282187
```