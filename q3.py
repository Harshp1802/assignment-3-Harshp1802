import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression
from sklearn.linear_model import LogisticRegression as LLR
from metrics import *
from sklearn.datasets import load_digits

np.random.seed(42)


X, y = load_digits(return_X_y=True,as_frame=True)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X))
# X = (X - X.min( )) / (X.max( ) - X.min( )) # This doesn't work, time to use skelarn... LOL!
data = pd.concat([X, y.rename("y")],axis=1, ignore_index=True)
data = data.sample(frac=1).reset_index(drop=True) # RANDOMLY SHUFFLING THE DATASET
split = int(0.7*len(data)) # TRAIN-TEST SPLIT
X_train, y_train = data.iloc[:split].iloc[:,:-1], data.iloc[:split].iloc[:,-1]
X_test, y_test = data.iloc[split:].iloc[:,:-1], data.iloc[split:].iloc[:,-1]

for fit_intercept in [True]:
    LR = LogisticRegression(fit_intercept=fit_intercept)
    LR.fit_multi(X, y, n_iter=100,batch_size = len(X),lr = 3) # here you can use fit_non_vectorised / fit_autograd methods
    y_hat = LR.predict_multi(X)
    # print("THETA:", LR.coef_)
    print('Accuracy: ', accuracy(y_hat, y))
    # LR.plot_desicion_boundary(X,y)