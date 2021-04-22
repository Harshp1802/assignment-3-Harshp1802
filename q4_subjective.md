# ES654-2021 Assignment 3

*Harsh Patel* - *18110062*

------

> Time Complexity Analysis of Logistic Regression 
1) Learning/ Training:
    + The overall time complexity of this step is *O(M * N * n_iter * k)*
    + Gradient Descent Step!
    + Loss for each of the k-classes!

2) Predicting:
    + The overall time complexity of this step is *O(M * N * k)*
    + Multiplication with weight matrix! [N * M x M * K]

> Space Complexity Analysis of Logistic Regression
1) Learning/ Training:
    + The overall space complexity of this step is *O(M * N) + O(N * k) + O(M * k)*
    + X, y, Weight Matrix

2) Predicting:
    + The overall space complexity of this step is *O(M * k)*
    + We just need the Weight Matrix to predict the outputs

where t -> No. of iterations
N -> No. of Samples
M -> No. of Features

> Time taken vs M (No.of Features):
<p align = center>
<img src = ".\q4_plots\Varying_M.png" >
</p>

- Here we see that the Logistic regression while training and predicting takes a linear curve w.r.t to the no. of features with fixed N, n_iter.

> Time taken vs N (No.of Samples):
<p align = center>
<img src = ".\q4_plots\Varying_N.png" >
</p>

- Here we see that the Logistic regression while training takes a linear curve w.r.t to the no. of samples whereas the prediction time is almost constant with fixed M, n_iter.

> Time taken vs n_iter (No.of Iterations):
<p align = center>
<img src = ".\q4_plots\Varying_n_iter.png" >
</p>

- Here we see that the Logistic regression while training takes a linear curve w.r.t to the no. of iterations whereas the prediction time is almost constant with fixed N, M.


