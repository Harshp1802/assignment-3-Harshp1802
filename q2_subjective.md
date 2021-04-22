# ES654-2021 Assignment 3

*Harsh Patel* - *18110062*

------

- In this question I have implemented the Autograd version of the Gradient Descent Algorithm for L1 and L2 Regularised Logistic Regression.
- When fit_intercept is set to True, a new column is of 1s is added to the dataset which incorporates the theta-0 in our fitting.
- Using nested cross-validation to find the optimum lambda penalty terms for L2 and L1 regularisation.
- Using the L1 regularisation, I have added the plots and inference on the feature selection!

+ Results: 

## |--------- Nested cross-validation for finding Optimal lambda values ----------|
### ------------------ Regularization Type l1 -----------------
```
Test_fold = 1
         lamda = 1
                 Validation_Fold = 1 Accuracy:  0.9523809523809523
                 Validation_Fold = 2 Accuracy:  0.9523809523809523
                 Validation_Fold = 3 Accuracy:  0.9126984126984127
                         Avg_val_accuracy:  0.9391534391534391
         lamda = 2
                 Validation_Fold = 1 Accuracy:  0.9365079365079365
                 Validation_Fold = 2 Accuracy:  0.9603174603174603
                 Validation_Fold = 3 Accuracy:  0.8968253968253969
                         Avg_val_accuracy:  0.9312169312169312
         lamda = 3
                 Validation_Fold = 1 Accuracy:  0.9365079365079365
                 Validation_Fold = 2 Accuracy:  0.9603174603174603
                 Validation_Fold = 3 Accuracy:  0.8888888888888888
                         Avg_val_accuracy:  0.9285714285714285
         lamda = 4
                 Validation_Fold = 1 Accuracy:  0.9206349206349206
                 Validation_Fold = 2 Accuracy:  0.9523809523809523
                 Validation_Fold = 3 Accuracy:  0.8888888888888888
                         Avg_val_accuracy:  0.9206349206349206
         Optimal_lamda: 1   Optimal_Accuracy: 0.9391534391534391
Test_fold = 2
         lamda = 1
                 Validation_Fold = 1 Accuracy:  0.9523809523809523
                 Validation_Fold = 2 Accuracy:  0.9206349206349206
                 Validation_Fold = 3 Accuracy:  0.9047619047619048
                         Avg_val_accuracy:  0.9259259259259259
         lamda = 2
                 Validation_Fold = 1 Accuracy:  0.9444444444444444
                 Validation_Fold = 2 Accuracy:  0.9285714285714286
                 Validation_Fold = 3 Accuracy:  0.8888888888888888
                         Avg_val_accuracy:  0.9206349206349206
         lamda = 3
                 Validation_Fold = 1 Accuracy:  0.9444444444444444
                 Validation_Fold = 2 Accuracy:  0.9285714285714286
                 Validation_Fold = 3 Accuracy:  0.8968253968253969
                         Avg_val_accuracy:  0.9232804232804233
         lamda = 4
                 Validation_Fold = 1 Accuracy:  0.9285714285714286
                 Validation_Fold = 2 Accuracy:  0.9206349206349206
                 Validation_Fold = 3 Accuracy:  0.8968253968253969
                         Avg_val_accuracy:  0.9153439153439153
         Optimal_lamda: 1   Optimal_Accuracy: 0.9259259259259259
Test_fold = 3
         lamda = 1
                 Validation_Fold = 1 Accuracy:  0.9523809523809523
                 Validation_Fold = 2 Accuracy:  0.9206349206349206
                 Validation_Fold = 3 Accuracy:  0.9365079365079365
                         Avg_val_accuracy:  0.9365079365079364
         lamda = 2
                 Validation_Fold = 1 Accuracy:  0.9523809523809523
                 Validation_Fold = 2 Accuracy:  0.9126984126984127
                 Validation_Fold = 3 Accuracy:  0.9285714285714286
                         Avg_val_accuracy:  0.9312169312169312
         lamda = 3
                 Validation_Fold = 1 Accuracy:  0.9444444444444444
                 Validation_Fold = 2 Accuracy:  0.9126984126984127
                 Validation_Fold = 3 Accuracy:  0.9126984126984127
                         Avg_val_accuracy:  0.9232804232804233
         lamda = 4
                 Validation_Fold = 1 Accuracy:  0.9285714285714286
                 Validation_Fold = 2 Accuracy:  0.8968253968253969
                 Validation_Fold = 3 Accuracy:  0.9206349206349206
                         Avg_val_accuracy:  0.9153439153439153  
         Optimal_lamda: 1   Optimal_Accuracy: 0.9365079365079364
The optimal lamdas for each folds are  [1, 1, 1]
```
### ------------------ Regularization Type l2 -----------------     

```Test_fold = 1
         lamda = 1
                 Validation_Fold = 1 Accuracy:  0.9444444444444444
                 Validation_Fold = 2 Accuracy:  0.9523809523809523
                 Validation_Fold = 3 Accuracy:  0.9047619047619048
                         Avg_val_accuracy:  0.9338624338624338
         lamda = 2
                 Validation_Fold = 1 Accuracy:  0.9444444444444444
                 Validation_Fold = 2 Accuracy:  0.9523809523809523
                 Validation_Fold = 3 Accuracy:  0.8888888888888888
                         Avg_val_accuracy:  0.9285714285714285
         lamda = 3
                 Validation_Fold = 1 Accuracy:  0.9365079365079365
                 Validation_Fold = 2 Accuracy:  0.9523809523809523
                 Validation_Fold = 3 Accuracy:  0.8888888888888888
                         Avg_val_accuracy:  0.9259259259259259
         lamda = 4
                 Validation_Fold = 1 Accuracy:  0.9365079365079365
                 Validation_Fold = 2 Accuracy:  0.9523809523809523
                 Validation_Fold = 3 Accuracy:  0.8888888888888888
                         Avg_val_accuracy:  0.9259259259259259
         Optimal_lamda: 1   Optimal_Accuracy: 0.9338624338624338
Test_fold = 2
         lamda = 1
                 Validation_Fold = 1 Accuracy:  0.9523809523809523
                 Validation_Fold = 2 Accuracy:  0.9285714285714286
                 Validation_Fold = 3 Accuracy:  0.8968253968253969
                         Avg_val_accuracy:  0.9259259259259259
         lamda = 2
                 Validation_Fold = 1 Accuracy:  0.9523809523809523
                 Validation_Fold = 2 Accuracy:  0.9365079365079365
                 Validation_Fold = 3 Accuracy:  0.8968253968253969
                         Avg_val_accuracy:  0.9285714285714285
         lamda = 3
                 Validation_Fold = 1 Accuracy:  0.9365079365079365
                 Validation_Fold = 2 Accuracy:  0.9206349206349206
                 Validation_Fold = 3 Accuracy:  0.8888888888888888
                         Avg_val_accuracy:  0.9153439153439153
         lamda = 4
                 Validation_Fold = 1 Accuracy:  0.9365079365079365
                 Validation_Fold = 2 Accuracy:  0.9206349206349206
                 Validation_Fold = 3 Accuracy:  0.8809523809523809
                         Avg_val_accuracy:  0.9126984126984127
         Optimal_lamda: 2   Optimal_Accuracy: 0.9285714285714285
Test_fold = 3
         lamda = 1
                 Validation_Fold = 1 Accuracy:  0.9523809523809523
                 Validation_Fold = 2 Accuracy:  0.9126984126984127
                 Validation_Fold = 3 Accuracy:  0.9365079365079365
                         Avg_val_accuracy:  0.9338624338624338
         lamda = 2
                 Validation_Fold = 1 Accuracy:  0.9523809523809523
                 Validation_Fold = 2 Accuracy:  0.9126984126984127
                 Validation_Fold = 3 Accuracy:  0.9365079365079365
                         Avg_val_accuracy:  0.9338624338624338
         lamda = 3
                 Validation_Fold = 1 Accuracy:  0.9444444444444444
                 Validation_Fold = 2 Accuracy:  0.9126984126984127
                 Validation_Fold = 3 Accuracy:  0.9365079365079365
                         Avg_val_accuracy:  0.9312169312169312
         lamda = 4
                 Validation_Fold = 1 Accuracy:  0.9444444444444444
                 Validation_Fold = 2 Accuracy:  0.9047619047619048
                 Validation_Fold = 3 Accuracy:  0.9285714285714286
                         Avg_val_accuracy:  0.9259259259259259
         Optimal_lamda: 1   Optimal_Accuracy: 0.9338624338624338
The optimal lamdas for each folds are  [1, 2, 1]
```

|--------- Feature Selection using L1 Regularisation ----------|
```
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [00:58<00:00,  5.09it/s] 
```
<p align = center>
<img src = " ./q2_plots/feature_selection_l1.jpg" >
</p>

### Here we observe that coefficients for features like mean fractal dimension and mean fractal point become zero at highere values of lambda, implying that those features are quite important than others for the output!
