import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression
from sklearn.linear_model import LogisticRegression as LLR
from metrics import *
from sklearn.datasets import load_breast_cancer

np.random.seed(42)

# A)
# N = 10
# P = 2
# X = pd.DataFrame(np.random.randn(N, P))
# X = (X - X.min( )) / (X.max( ) - X.min( ))
# y = pd.Series(np.random.randint(2,size=N))

# for fit_intercept in [True]:

#     LR = LogisticRegression(fit_intercept=fit_intercept)
#     LR.fit_autograd(X, y, n_iter=100,batch_size = 5,lr = 2, reg_type = "l2") # here you can use fit_non_vectorised / fit_autograd methods
#     # LR.fit_autograd(X, y, n_iter=1000,batch_size = 10,lr = 0.0001)
#     y_hat = LR.predict(X)
#     # print("THETA:", LR.coef_)
#     print('Accuracy: ', accuracy(y_hat, y))
#     LR.plot_desicion_boundary(X,y)

# model = LLR(fit_intercept = True, penalty="none", max_iter=515)
# model.fit(X, y)
# print(model.predict(X))
# print(model.intercept_, model.coef_)
# print(model.score(X, y))

# C)
from sklearn.datasets import make_classification

X, y = make_classification(n_features=2, random_state=444,n_redundant=0)
X = pd.DataFrame(X)
y = pd.Series(y)
# N = 30
# P = 2
# X = pd.DataFrame(np.random.randn(N, P))
# y = pd.Series(np.random.randint(2,size=N))


X = (X - X.min( )) / (X.max( ) - X.min( ))
data = pd.concat([X, y.rename("y")],axis=1, ignore_index=True)
FOLDS = 3
size = len(data)//FOLDS
#__________ 5 folds created ____________#
Xfolds = [data.iloc[i*size:(i+1)*size].iloc[:,:-1] for i in range(FOLDS)]
yfolds = [data.iloc[i*size:(i+1)*size].iloc[:,-1] for i in range(FOLDS)]
cross_val_folds = 3
lamda_range  = [0,0.0001,0.001,0.01,0.1,1,10,40]

for reg_type in ["l1","l2"]:
    Optimals = []
    print(f"------------------ Regularization Type {reg_type} -----------------")
    for i in range(FOLDS):
        print("Test_fold = {}".format(i+1))
        Xdash, ydash = Xfolds.copy(), yfolds.copy()
        #__________ Use one of them as Test fold ____________#
        X_test,y_test = Xdash[i], ydash[i]
        Xdash.pop(i)
        ydash.pop(i)
        #__________ Concat the rest to create the Train fold ____________#
        X_train,y_train = pd.concat(Xdash), pd.concat(ydash)
        size = len(X_train)//cross_val_folds
        X_train_folds = [X_train.iloc[j*size:(j+1)*size] for j in range(cross_val_folds)]
        y_train_folds = [y_train.iloc[j*size:(j+1)*size] for j in range(cross_val_folds)]
        val_accuracies = []

        for lamda in lamda_range:
            #__________ Create Trees for multiple lamdas and find the one with the best avg_val_accuracy ____________#
            print("\t lamda = {}".format(lamda))
            avg_validation_accuracy = 0
            for k in range(cross_val_folds):
                #__________ Further splitting into multiple Folds ____________#
                print("\t \t Validation_Fold = {}".format(k + 1), end =" ")
                X_traindash, y_traindash = X_train_folds.copy(), y_train_folds.copy()
                #__________ Use one of them as Validation fold ____________#
                X_valid,y_valid = X_train_folds[k],y_train_folds[k]
                X_traindash.pop(k)
                y_traindash.pop(k)
                #__________ Concat the rest to create the nested Train Fold ____________#
                train_X, train_y =  pd.concat(X_traindash), pd.concat(y_traindash)            
                LR = LogisticRegression()
                LR.fit_autograd(train_X.reset_index(drop=True), train_y.reset_index(drop=True), n_iter=100,batch_size = len(train_X),lr = 3, reg_type = reg_type, lamda=lamda)
                y_hat = LR.predict(X_valid.reset_index(drop=True))
                # LR.plot_desicion_boundary(X_valid,y_valid)
                valid_accuracy = accuracy(y_hat,y_valid.reset_index(drop=True))
                print("Accuracy:  {}".format(valid_accuracy))
                avg_validation_accuracy += valid_accuracy
            avg_validation_accuracy = avg_validation_accuracy/cross_val_folds
            print("\t \t \t Avg_val_accuracy:  {}".format(avg_validation_accuracy))
            val_accuracies.append([avg_validation_accuracy, lamda])
        val_accuracies.sort(reverse = True)
        opt_lamda = 0
        opt_acc = 0
        for i in range(len(val_accuracies)):
            if(val_accuracies[i][0]>= opt_acc):
                opt_lamda = val_accuracies[i][1]
                opt_acc = val_accuracies[i][0]

        print("\t Optimal_lamda: {}   Optimal_Accuracy: {}".format(opt_lamda, opt_acc))
        Optimals.append(opt_lamda)
            
    print("The optimal lamdas for each folds are ", Optimals)

# X, y = load_breast_cancer(return_X_y=True,as_frame=True)
# X = (X - X.min( )) / (X.max( ) - X.min( ))
# data = pd.concat([X, y.rename("y")],axis=1, ignore_index=True)
# data = data.sample(frac=1).reset_index(drop=True) # RANDOMLY SHUFFLING THE DATASET
# FOLDS = 3
# size = len(data)//FOLDS
# #__________ 3 folds created ____________#
# Xfolds = [data.iloc[i*size:(i+1)*size].iloc[:,:-1] for i in range(FOLDS)]
# yfolds = [data.iloc[i*size:(i+1)*size].iloc[:,-1] for i in range(FOLDS)]
# avg_accuracy = 0
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
#     LR.fit_autograd(X_train, y_train, n_iter=1000,batch_size = 20,lr = 3, lr_type="inverse")
#     # LR.plot_desicion_boundary(X_train,y_train)
#     y_hat = LR.predict(X_test)
#     test_accuracy = accuracy(y_hat,y_test.reset_index(drop=True))
#     print("\t Test_Accuracy: {}".format(test_accuracy))
#     avg_accuracy += test_accuracy

# avg_accuracy = avg_accuracy/FOLDS
# print("AVERAGE ACCURACY = {}".format(avg_accuracy))

# D) Done Above!

