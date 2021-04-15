import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression
from sklearn.linear_model import LogisticRegression as LLR
from metrics import *

np.random.seed(42)

N = 10
P = 1
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(2,size=N))

for fit_intercept in [True]:

    LR = LogisticRegression(fit_intercept=fit_intercept)
    LR.fit_non_vectorised(X, y, n_iter=515,batch_size = 1,lr = 0.01) # here you can use fit_non_vectorised / fit_autograd methods
    # LR.fit_autograd(X, y, n_iter=1000,batch_size = 10,lr = 0.0001)
    y_hat = LR.predict(X)
    print("THETA:", LR.coef_)
    print('Accuracy: ', accuracy(y_hat, y))
    # for cls in y.unique():
    #     print('Precision: ', precision(y_hat, y, cls))
    #     print('Recall: ', recall(y_hat, y, cls))

# model = LLR(fit_intercept = True, penalty="none", max_iter=515)
# model.fit(X, y)
# print(model.predict(X))
# print(model.intercept_, model.coef_)
# print(model.score(X, y))