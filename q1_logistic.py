import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression
from metrics import *
from sklearn.datasets import load_breast_cancer

np.random.seed(42)

#--------- Random 2-class Dataset -------- #

N = 30
P = 2
X = pd.DataFrame(np.random.randn(N, P))
X = (X - X.min( )) / (X.max( ) - X.min( )) # Normalisation is necessary here to avoid overflows!
y = pd.Series(np.random.randint(2,size=N))
fit_intercept = True

# a) --------- Unregularised Logistic Regression: implemented in LogisticRegression.py --------#

print("\n|--------- Unregularised Logistic Regression ----------|")
LR = LogisticRegression(fit_intercept=fit_intercept)
LR.fit_non_vectorised(X, y, n_iter=200,batch_size = 5,lr = 2.5)
y_hat = LR.predict(X)
print("Obtained THETA values:", LR.coef_)
print('Accuracy: ', accuracy(y_hat, y))
# d) --------- Plotting Desicion Boundary --------#
LR.plot_desicion_boundary(X,y, "./q1_plots/db_non_vectorised.jpg")
print("Desicion Boundary plot at ./q1_plots/db_non_vectorised.jpg")

# a) --------- Unregularised Logistic Regression using Autograd: implemented in LogisticRegression.py--------#
print("\n|--------- Unregularised Logistic Regression using Autograd ----------|")
LR = LogisticRegression(fit_intercept=fit_intercept)
LR.fit_autograd(X, y, n_iter=200,batch_size = 5,lr = 2.5)
y_hat = LR.predict(X)
print("Obtained THETA values:", LR.coef_)
print('Accuracy_autograd: ', accuracy(y_hat, y))
LR.plot_desicion_boundary(X,y, "./q1_plots/db_autograd.jpg")
print("Desicion Boundary plot at ./q1_plots/db_autograd.jpg")

print("\n|--------- 3-Fold Unregularised Logistic Regression on Breast Cancer Dataset ----------|")
X, y = load_breast_cancer(return_X_y=True,as_frame=True)
X = (X - X.min( )) / (X.max( ) - X.min( ))
data = pd.concat([X, y.rename("y")],axis=1, ignore_index=True)
data = data.sample(frac=1).reset_index(drop=True) # RANDOMLY SHUFFLING THE DATASET
FOLDS = 3
size = len(data)//FOLDS
#__________ 3 folds created ____________#
Xfolds = [data.iloc[i*size:(i+1)*size].iloc[:,:-1] for i in range(FOLDS)]
yfolds = [data.iloc[i*size:(i+1)*size].iloc[:,-1] for i in range(FOLDS)]
avg_accuracy = 0
for i in range(FOLDS):
    print("Test_fold = {}".format(i+1))
    Xdash, ydash = Xfolds.copy(), yfolds.copy()
    #__________ Use one of them as Test fold ____________#
    X_test, y_test = Xdash[i], ydash[i]
    Xdash.pop(i)
    ydash.pop(i)
    #__________ Concat the rest to create the Train fold ____________#
    X_train,y_train = pd.concat(Xdash), pd.concat(ydash)
    LR = LogisticRegression(fit_intercept=True)
    LR.fit_autograd(X_train, y_train, n_iter=1000,batch_size = 20,lr = 3, lr_type="inverse")
    # LR.plot_desicion_boundary(X_train,y_train,f"./q1_plots/fold_{i}.jpg")
    y_hat = LR.predict(X_test)
    test_accuracy = accuracy(y_hat,y_test.reset_index(drop=True))
    print("\t Test_Accuracy: {}".format(test_accuracy))
    avg_accuracy += test_accuracy
avg_accuracy = avg_accuracy/FOLDS
print("AVERAGE ACCURACY = {}".format(avg_accuracy))