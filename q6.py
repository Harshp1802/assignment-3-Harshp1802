import autograd.numpy as anp
import pandas as pd
from autograd import grad
from sklearn.datasets import load_digits, load_boston
from sklearn.preprocessing import MinMaxScaler
from tqdm import trange
from metrics import *
from math import e
from multi_layer_perceptron import MLP
import matplotlib.pyplot as plt
anp.random.seed(42)

# WITHOUT 3-FOLD!
#--------------------------------------------------------------------
# #a) --------- NN CLassification- DIgits Dataset --------#
#--------------------------------------------------------------------

# print("\n|--------- NN CLassification on DIGITS Dataset ----------|")
# X, y = load_digits(return_X_y=True,as_frame=True)
# scaler = MinMaxScaler()
# X = pd.DataFrame(scaler.fit_transform(X))
# y = pd.Series(y)
# data = pd.concat([X, y.rename("y")],axis=1, ignore_index=True)
# data = data.sample(frac=1).reset_index(drop=True) # RANDOMLY SHUFFLING THE DATASET
# split = int(0.7*len(data)) # TRAIN-TEST SPLIT
# X_train, y_train = data.iloc[:split].iloc[:,:-1], data.iloc[:split].iloc[:,-1]
# X_test, y_test = data.iloc[split:].iloc[:,:-1], data.iloc[split:].iloc[:,-1]

# NN = MLP(
#     [20],
#     ['sigmoid'],
#     "classification",
#     X_train.shape[1],
#     len(list(y_train.unique()))
# )
# n_epochs = 1000
# lr = 2
# losses = []
# for epoch in trange(n_epochs):
#     output = NN.forward(X_train, NN.WEIGHTS, NN.BIASES)
#     if(NN.fit_type =="classification"):
#         epoch_loss = NN.CrossE_func(NN.WEIGHTS, NN.BIASES, y_train)
#     else:
#         epoch_loss = NN.mse_func(NN.WEIGHTS, NN.BIASES, anp.array(y_train))
#     # print(f"Epoch {epoch}: Loss = {epoch_loss}")
#     losses.append(epoch_loss)
#     NN.backprop(lr, anp.array(y_train))

# y_hat = NN.predict(X_test)

# if(NN.fit_type =="classification"):
#     print('Accuracy', accuracy(y_hat, y_test))
# else:
#     print('RMSE: ', rmse(y_hat, y_test))

# plt.figure()
# plt.title("Training Loss vs Epochs: DIGITS Dataset")
# plt.plot(list(range(n_epochs)), losses)
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.savefig("./q6_plots/digits_training.jpg")
# print("\n|--------- ./q6_plots/digits_training.jpg ----------|")


#--------------------------------------------------------------------
# # a) --------- NN Regression- Boston Dataset --------#
#--------------------------------------------------------------------

# print("\n|--------- NN Regression on Boston Datasett ----------|")
# X, y = load_boston(return_X_y=True)
# scaler = MinMaxScaler()
# X = pd.DataFrame(scaler.fit_transform(X))
# y = pd.Series(y)
# data = pd.concat([X, y.rename("y")],axis=1, ignore_index=True)
# data = data.sample(frac=1).reset_index(drop=True) # RANDOMLY SHUFFLING THE DATASET
# split = int(0.7*len(data)) # TRAIN-TEST SPLIT
# X_train, y_train = data.iloc[:split].iloc[:,:-1], data.iloc[:split].iloc[:,-1]
# X_test, y_test = data.iloc[split:].iloc[:,:-1], data.iloc[split:].iloc[:,-1]

# NN = MLP(
#     [20, 10],
#     ['sigmoid', 'relu'],
#     'regression',
#     X_train.shape[1],
#     len(list(y_train.unique()))
# )
# n_epochs = 700
# lr = 1
# losses = []
# for epoch in trange(n_epochs):
#     output = NN.forward(X_train, NN.WEIGHTS, NN.BIASES)
#     if(NN.fit_type =="classification"):
#         epoch_loss = NN.CrossE_func(NN.WEIGHTS, NN.BIASES, y_train)
#     else:
#         epoch_loss = NN.mse_func(NN.WEIGHTS, NN.BIASES, anp.array(y_train))
#     # print(f"Epoch {epoch}: Loss = {epoch_loss}")
#     losses.append(epoch_loss)
#     NN.backprop(lr, anp.array(y_train))

# y_hat = NN.predict(X_test)

# if(NN.fit_type =="classification"):
#     print('Accuracy', accuracy(y_hat, y_test))
# else:
#     print('RMSE: ', rmse(y_hat, y_test))

# plt.figure()
# plt.title("Training Loss vs Epochs: BOSTON Dataset")
# plt.plot(list(range(n_epochs)), losses)
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.savefig("./q6_plots/boston_training.jpg")
# print("\n|--------- ./q6_plots/boston_training.jpg ----------|")


# WITH 3-FOLD!

#--------------------------------------------------------------------
print("\n|--------- 3-Fold NN CLassification on DIGITS Dataset ----------|")
#--------------------------------------------------------------------

X, y = load_digits(return_X_y=True,as_frame=True)
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X))
y = pd.Series(y)
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
        
    NN = MLP(
        [20],
        ['sigmoid'],
        "classification",
        X_train.shape[1],
        len(list(y_train.unique()))
    )
    n_epochs = 1000
    lr = 2
    losses = []
    for epoch in trange(n_epochs):
        output = NN.forward(X_train, NN.WEIGHTS, NN.BIASES)
        if(NN.fit_type =="classification"):
            epoch_loss = NN.CrossE_func(NN.WEIGHTS, NN.BIASES, y_train)
        else:
            epoch_loss = NN.mse_func(NN.WEIGHTS, NN.BIASES, anp.array(y_train))
        # print(f"Epoch {epoch}: Loss = {epoch_loss}")
        losses.append(epoch_loss)
        NN.backprop(lr, anp.array(y_train))

    y_hat = NN.predict(X_test)

    if(NN.fit_type =="classification"):
        print('Accuracy', accuracy(y_hat, y_test))
        test_accuracy = accuracy(y_hat, y_test)
    else:
        print('RMSE: ', rmse(y_hat, y_test))
        test_accuracy = rmse(y_hat, y_test)
    print("\t Test_Accuracy: {}".format(test_accuracy))
    avg_accuracy += test_accuracy
avg_accuracy = avg_accuracy/FOLDS
print("AVERAGE ACCURACY = {}".format(avg_accuracy))


#--------------------------------------------------------------------
print("\n|--------- 3-Fold NN Regression on BOSTON Dataset ----------|")
#--------------------------------------------------------------------


X, y = load_boston(return_X_y=True)
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X))
y = pd.Series(y)
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
    NN = MLP(
        [20, 10],
        ['sigmoid', 'relu'],
        'regression',
        X_train.shape[1],
        len(list(y_train.unique()))
    )
    n_epochs = 700
    lr = 1
    losses = []
    for epoch in trange(n_epochs):
        output = NN.forward(X_train, NN.WEIGHTS, NN.BIASES)
        if(NN.fit_type =="classification"):
            epoch_loss = NN.CrossE_func(NN.WEIGHTS, NN.BIASES, y_train)
        else:
            epoch_loss = NN.mse_func(NN.WEIGHTS, NN.BIASES, anp.array(y_train))
        # print(f"Epoch {epoch}: Loss = {epoch_loss}")
        losses.append(epoch_loss)
        NN.backprop(lr, anp.array(y_train))

    y_hat = NN.predict(X_test)

    if(NN.fit_type =="classification"):
        print('Accuracy', accuracy(y_hat, y_test))
        test_accuracy = accuracy(y_hat, y_test)
    else:
        print('RMSE: ', rmse(y_hat, y_test))
        test_accuracy = rmse(y_hat, y_test)
    print("\t Test_Accuracy: {}".format(test_accuracy))
    avg_accuracy += test_accuracy
avg_accuracy = avg_accuracy/FOLDS
print("AVERAGE ACCURACY = {}".format(avg_accuracy))
