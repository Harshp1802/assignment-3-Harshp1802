# ES654-2020 Assignment 3

*Harsh Patel* - *18110062*

------

> Time Complexity Analysis of Normal Fitting vs Linear Regression using Gradient Descent

1) Normal Fitting:

+ In normal fitting we directly find the predictions using 
    ``` y = (X^T.X)^-1.X^T.y```.
+ The overall complexity of this method is *O(D^2 * N) + O(D^3)*

1) Gradient Descent Fitting:

+ In normal fitting we directly find the predictions using 
    ``` θ = θ − αX^T(Xθ − y)```.
+ The overall complexity of this method is *O((n*D*t)*

where t -> No. of iterations
N -> No. of Samples
D -> No. of Features

> Time taken vs M (No.of Features):
<p align = center>
<img src = ".\Q8_plots\Varying_M.png" >
</p>

- Here we see that for normal fitting the curve increases very fast as compared to the gradient descent fitting because the time complexity of normal fitting is of the order of D^3 whereas its linear in the case with Gradient Descent with fixed N, n_iter.

<p align = center>
<img src = ".\Q8_plots\Varying_N.png" >
</p>

- Here we see that for gradient descent fitting, the curve increases linearly similar to the normal fitting because the time complexity of the order of N for both methods with fixed M and n_iter.

<p align = center>
<img src = ".\Q8_plots\Varying_n_iter.png" >
</p>

- Here we see that for gradient descent fitting, the curve increases linearly as we increase the no. of iterations because the time complexity is directly proportional in this method. For this eg., I have fixed the batch_size to the length of the dataset.


