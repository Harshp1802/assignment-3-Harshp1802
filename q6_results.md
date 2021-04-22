# ES654-2021 Assignment 3

*Harsh Patel* - *18110062*

------

- In this question I have implemented the Multi Layer Perceptron (Fully connected NN) for both classification and regression!
- Using Digits dataset and Boston Dataset over this MLP with 3-Folds to test the working of the NN.
- The input is X, y, [n1, n2, …, nh] where ni is the number of neurons in i^th hidden layer, [g1, g2, …, gh] where gi in {‘relu’ ,‘identity’, ‘sigmoid’} are the activations for i^th layer.

+ Results:

## Training loss vs epochs:

### DIGITS Dataset:

<p align = center>
<img src = " ./q6_plots/digits_training.jpg" >
</p>

### BOSTON Dataset:

<p align = center>
<img src = " ./q6_plots/boston_training.jpg" >
</p>

##                    |--------- 3-Fold NN CLassification on DIGITS Dataset ----------|
```

Test_fold = 1
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [03:02<00:00,  5.49it/s] 
Accuracy 0.7913188647746243
         Test_Accuracy: 0.7913188647746243
Test_fold = 2
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [03:15<00:00,  5.13it/s] 
Accuracy 0.8213689482470785
         Test_Accuracy: 0.8213689482470785
Test_fold = 3
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [04:04<00:00,  4.08it/s] 
Accuracy 0.8146911519198664
         Test_Accuracy: 0.8146911519198664
AVERAGE ACCURACY = 0.8091263216471898
```

##                    |--------- 3-Fold NN Regression on BOSTON Dataset ----------|
```
Test_fold = 1
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 700/700 [01:08<00:00, 10.15it/s] 
RMSE:  8.7573679581799
         Test_Accuracy: 8.7573679581799
Test_fold = 2
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 700/700 [01:09<00:00, 10.09it/s] 
RMSE:  9.879562721458688
         Test_Accuracy: 9.879562721458688
Test_fold = 3
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 700/700 [01:07<00:00, 10.43it/s] 
RMSE:  9.028899460668939
         Test_Accuracy: 9.028899460668939
AVERAGE ACCURACY = 9.221943380102509
```