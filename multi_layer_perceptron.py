import autograd.numpy as anp
import pandas as pd
from autograd import grad
from tqdm import trange
from math import e

class MLP: # Multi-Layer-Perceptron: Class Object
    def __init__(self, n_hidden_layers, a_hidden_layers, fit_type = "classification", n_features = 0, n_classes = 10):
        
        self.N = len(n_hidden_layers)
        if(self.N == 0):
            n_hidden_layers = [n_features]
        self.n_hidden_layers = n_hidden_layers # Nuerorons in the hidden layes
        self.a_hidden_layers = a_hidden_layers # Activations at the hidden layers
        self.fit_type = fit_type # Classification/Regression
        self.WEIGHTS = []
        self.BIASES = []
        self.n_classes = n_classes
        inp_size = n_features # N x M input data

        # Initializing weights and biases for each layer of the MLP
        for i in range(self.N):
            self.WEIGHTS.append( anp.array([[0.0]*inp_size]*self.n_hidden_layers[i]).T  )
            self.BIASES.append( anp.array([0.0]*self.n_hidden_layers[i]).T  )
            inp_size = self.n_hidden_layers[i]
        if(fit_type == "classification"): # IF CLASSIFICATION, LAST LAYER no. of neurons = no. of classes
            self.WEIGHTS.append( anp.array([[0.0]*self.n_hidden_layers[-1]]*self.n_classes).T  )
            self.BIASES.append( anp.array([0.0]*self.n_classes).T  )
        else: # IF REGRESSION, LAST LAYER no. of neurons = 1
            self.WEIGHTS.append( anp.array([[0.0]*self.n_hidden_layers[-1]]*1).T  )
            self.BIASES.append( anp.array([0.0]).T  )
        return

    def softmax(self,X):
        # Softmax Acitvation
        P = anp.exp(X) # To avoid overflows, we can subtract the max term (axis = 1) !
        return P/anp.sum(P,axis=1).reshape(-1,1)

    def forward(self, X, WEIGHTS, BIASES):
        # Forward passs through the network... input = activation(previous_layer_output)*Weights + Biases
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
        
        if(self.fit_type == "classification"): # LAst layer is softmax if classifiaction
            output = self.softmax(output)
        else:
            output = anp.maximum(output, 0.0)

        return output
    
    def CrossE_func(self, weights, biases, y):
        # Cross Entropy calculation: Helper function for autograd calculations
        y_hat = self.forward(self.X, weights, biases)
        CrossE = 0
        for k in range(self.n_classes):
            CrossE -= anp.dot((y == k).astype(float),anp.log(y_hat[:,k]))
        return CrossE

    def mse_func(self, weights, biases, y):
        # Mean Squared Error calculation : Helper function for autograd calculations
        y_hat = self.forward(self.X, weights, biases)
        MSE = anp.sum(anp.square(anp.subtract(y_hat, y.reshape(-1,1))))/len(y)
        return MSE

    def backprop(self, lr, y):
        # Backpropogation to find the gradients of loss w.r.t each parameter of the network
        if(self.fit_type =="classification"):
            dJw = grad(self.CrossE_func,0)(self.WEIGHTS,self.BIASES, y)
            dJb = grad(self.CrossE_func,1)(self.WEIGHTS,self.BIASES, y)
        else:
            dJw = grad(self.mse_func,0)(self.WEIGHTS,self.BIASES, y)
            dJb = grad(self.mse_func,1)(self.WEIGHTS,self.BIASES, y)
        
        # Gradient Descent Step
        for i in range(self.N + 1):
            self.WEIGHTS[i] -= lr*dJw[i]/len(self.X)
            self.BIASES[i] -= lr*dJb[i]/len(self.X)
        return

    def predict(self, X):
        y_hat = self.forward(X, self.WEIGHTS, self.BIASES)
        if(self.fit_type =="classification"): # Taking argmax of probabilities, if classification!
            return anp.argmax(y_hat,axis=1)
        return y_hat


    