import autograd.numpy as anp
import pandas as pd
from autograd import grad
from sklearn.datasets import load_digits, load_boston
from sklearn.preprocessing import MinMaxScaler
from tqdm import trange
from metrics import *
from math import e
anp.random.seed(42)


def softmax(X):
    P = anp.exp(X)
    return P/anp.sum(P,axis=1).reshape(-1,1)

class MLP:
    def __init__(self, n_hidden_layers, a_hidden_layers, fit_type = "classification", n_features = 0, n_classes = 10):
        
        self.N = len(n_hidden_layers)
        if(self.N == 0):
            n_hidden_layers = [n_features]
        self.n_hidden_layers = n_hidden_layers
        self.a_hidden_layers = a_hidden_layers
        self.fit_type = fit_type
        self.WEIGHTS = []
        self.BIASES = []
        self.n_classes = n_classes
        inp_size = n_features # N x M input data
        for i in range(self.N):
            self.WEIGHTS.append( anp.array([[0.0]*inp_size]*self.n_hidden_layers[i]).T  )
            self.BIASES.append( anp.array([0.0]*self.n_hidden_layers[i]).T  )
            inp_size = self.n_hidden_layers[i]
        if(fit_type == "classification"):
            self.WEIGHTS.append( anp.array([[0.0]*self.n_hidden_layers[-1]]*self.n_classes).T  )
            self.BIASES.append( anp.array([0.0]*self.n_classes).T  )
        else:
            self.WEIGHTS.append( anp.array([[0.0]*self.n_hidden_layers[-1]]*1).T  )
            self.BIASES.append( anp.array([0.0]).T  )
        return

    def forward(self, X, WEIGHTS, BIASES):
        self.X = X
        inp_next = X
        for i in range(self.N):
            output =  anp.dot(anp.array(inp_next),WEIGHTS[i]) + anp.array([BIASES[i]]*inp_next.shape[0])
            activation = self.a_hidden_layers[i]
            if(activation == "relu"):
                output = anp.maximum(output, 0.0)
            elif(activation == "sigmoid"):
                output = (1.0)/(1+e**(-anp.array(output)))
            else:
                pass
            inp_next = output

        output =  anp.dot(anp.array(inp_next),WEIGHTS[-1]) + anp.array([BIASES[-1]]*inp_next.shape[0])
        
        if(self.fit_type == "classification"):            
            output = softmax(output)
        else:
            output = anp.maximum(output, 0.0)

        return output
    
    def CrossE_func(self, weights, biases, y):
        y_hat = self.forward(self.X, weights, biases)
        CrossE = 0
        for k in range(self.n_classes):
            CrossE -= anp.dot((y == k).astype(float),anp.log(y_hat[:,k]))
        return CrossE

    def mse_func(self, weights, biases, y):
        # Helper function for autograd calculations
        y_hat = self.forward(self.X, weights, biases)
        MSE = anp.sum(anp.square(anp.subtract(y_hat, y.reshape(-1,1))))/len(y)
        return MSE

    def backprop(self, lr, y):
        if(self.fit_type =="classification"):
            dJw = grad(self.CrossE_func,0)(self.WEIGHTS,self.BIASES, y)
            dJb = grad(self.CrossE_func,1)(self.WEIGHTS,self.BIASES, y)
        else:
            dJw = grad(self.mse_func,0)(self.WEIGHTS,self.BIASES, y)
            dJb = grad(self.mse_func,1)(self.WEIGHTS,self.BIASES, y)
        for i in range(self.N + 1):
            self.WEIGHTS[i] -= lr*dJw[i]/len(self.X)
            self.BIASES[i] -= lr*dJb[i]/len(self.X)
        return

    def predict(self, X):
        y_hat = self.forward(X, self.WEIGHTS, self.BIASES)
        if(self.fit_type =="classification"):
            return anp.argmax(y_hat,axis=1)
        return y_hat

if __name__ == '__main__' :

    X, y = load_digits(return_X_y=True,as_frame=True)
    # X, y = load_boston(return_X_y=True)
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X))
    y = pd.Series(y)
    data = pd.concat([X, y.rename("y")],axis=1, ignore_index=True)
    data = data.sample(frac=1).reset_index(drop=True) # RANDOMLY SHUFFLING THE DATASET
    split = int(0.7*len(data)) # TRAIN-TEST SPLIT
    X_train, y_train = data.iloc[:split].iloc[:,:-1], data.iloc[:split].iloc[:,-1]
    X_test, y_test = data.iloc[split:].iloc[:,:-1], data.iloc[split:].iloc[:,-1]

    NN = MLP(
        [20],
        ['sigmoid'],
        "classification",
        # 'regression',
        X_train.shape[1],
        len(list(y_train.unique()))
    )
    n_epochs = 700
    lr = 2
    for epoch in trange(n_epochs):
        output = NN.forward(X_train, NN.WEIGHTS, NN.BIASES)
        if(NN.fit_type =="classification"):
            epoch_loss = NN.CrossE_func(NN.WEIGHTS, NN.BIASES, y_train)
        else:
            epoch_loss = NN.mse_func(NN.WEIGHTS, NN.BIASES, anp.array(y_train))
        print(f"Epoch {epoch}: Loss = {epoch_loss}")
        NN.backprop(lr, anp.array(y_train))
    
    y_hat = NN.predict(X_test)
    
    if(NN.fit_type =="classification"):
        print('Accuracy', accuracy(y_hat, y_test))
    else:
        print('RMSE: ', rmse(y_hat, y_test))