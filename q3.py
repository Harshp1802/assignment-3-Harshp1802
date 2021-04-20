import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression
from sklearn.linear_model import LogisticRegression as LLR
from metrics import *
from sklearn.datasets import load_digits
np.random.seed(42)


# X, y = load_digits(return_X_y=True,as_frame=True)
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# X = pd.DataFrame(scaler.fit_transform(X))
# # X = (X - X.min( )) / (X.max( ) - X.min( )) # This doesn't work, time to use skelarn... LOL!
# data = pd.concat([X, y.rename("y")],axis=1, ignore_index=True)
# data = data.sample(frac=1).reset_index(drop=True) # RANDOMLY SHUFFLING THE DATASET
# split = int(0.7*len(data)) # TRAIN-TEST SPLIT
# X_train, y_train = data.iloc[:split].iloc[:,:-1], data.iloc[:split].iloc[:,-1]
# X_test, y_test = data.iloc[split:].iloc[:,:-1], data.iloc[split:].iloc[:,-1]

# for fit_intercept in [True]:
#     LR = LogisticRegression(fit_intercept=fit_intercept)
#     LR.fit_multi(X, y, n_iter=100,batch_size = len(X),lr = 3) # here you can use fit_non_vectorised / fit_autograd methods
#     y_hat = LR.predict_multi(X)
#     # print("THETA:", LR.coef_)
#     print('Accuracy: ', accuracy(y_hat, y))
#     # LR.plot_desicion_boundary(X,y)

# # C)


# X, y = load_digits(return_X_y=True,as_frame=True)
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# X = pd.DataFrame(scaler.fit_transform(X))
# data = pd.concat([X, y.rename("y")],axis=1, ignore_index=True)
# data = data.sample(frac=1).reset_index(drop=True) # RANDOMLY SHUFFLING THE DATASET
# FOLDS = 4
# size = len(data)//FOLDS
# #__________ 4 folds created ____________#
# Xfolds = [data.iloc[i*size:(i+1)*size].iloc[:,:-1] for i in range(FOLDS)]
# yfolds = [data.iloc[i*size:(i+1)*size].iloc[:,-1] for i in range(FOLDS)]
# avg_accuracy = 0
# accs = []
# for i in range(FOLDS):
#     print("Test_fold = {}".format(i+1))
#     Xdash, ydash = Xfolds.copy(), yfolds.copy()
#     #__________ Use one of them as Test fold ____________#
#     X_test, y_test = Xdash[i], ydash[i]
#     Xdash.pop(i)
#     ydash.pop(i)
#     #__________ Concat the rest to create the Train fold ____________#
#     X_train,y_train = pd.concat(Xdash), pd.concat(ydash)
#     LR = LogisticRegression(fit_intercept=True)
#     LR.fit_multi(X_train, y_train, n_iter=500,batch_size = 20,lr = 3, lr_type="inverse")
#     # LR.plot_desicion_boundary(X_train,y_train)
#     y_hat = LR.predict_multi(X_test)
#     test_accuracy = accuracy(y_hat,y_test.reset_index(drop=True))
#     accs.append([test_accuracy,y_hat, y_test])
#     print("\t Test_Accuracy: {}".format(test_accuracy))
#     avg_accuracy += test_accuracy

# avg_accuracy = avg_accuracy/FOLDS
# print("AVERAGE ACCURACY = {}".format(avg_accuracy))

# accs = sorted(accs, key=lambda x: x[0], reverse=True)
# best_y_hat = accs[0][1]
# best_y_test = accs[0][2]
# from sklearn.metrics import confusion_matrix
# print("-------- Best Confusion Matrix --------")
# print(confusion_matrix(np.array(best_y_test), np.array(best_y_hat)))

# d)
from sklearn.datasets import load_digits
digits = load_digits()
plt.figure()
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
proj = pca.fit_transform(digits.data)
plt.scatter(proj[:, 0], proj[:, 1], c=digits.target, cmap="tab10")
plt.colorbar()
plt.show()