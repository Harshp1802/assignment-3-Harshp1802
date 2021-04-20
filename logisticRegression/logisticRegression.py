import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import Autograd modules here
from autograd import grad, numpy as anp
import matplotlib.animation as animation
from math import e


class LogisticRegression():
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods

        return
    
    def sigmoid(self,y_hat):
        # return (1.0)/(1+np.exp(-np.array(y_hat,dtype=float)))
        return (1.0)/(1+e**(-anp.array(y_hat)))

    def fit_non_vectorised(self, X, y, batch_size, n_iter=10, lr=0.01, lr_type='constant'):
        '''
        Function to train model using non-vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data. 
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        #______ NON-VECTORISED FITTING _______#
        self.n_iter = n_iter
        N = len(X)
        n_batches = N//batch_size
        LR = lr
        if(self.fit_intercept):
            #______ ADD 1s column if fitting intercept _______#
            X = pd.concat([pd.Series([1]*N, index = X.index),X],axis=1, ignore_index=True)

        # Dividing into batches
        Xbatches = [X.iloc[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]
        ybatches = [y.iloc[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]
        
        # Initializing with all Thetas = 0
        THETA = np.array([0.0]*len(X.columns)).T 
        
        for i in range(n_iter):
            if(lr_type=='inverse'):
                lr = LR/(i+1)
            X_train, y_train = Xbatches[i%n_batches], ybatches[i%n_batches]
            theta_old = THETA.copy() # Copy is necessary here!
            for j in range(len(X.columns)):
                # For each feature
                dCrossE = 0 # Gradient
                for k in range(batch_size):
                    y_hat = 0
                    for l in range(len(X.columns)):
                        y_hat += theta_old[l]*(X_train.iloc[k,l]) # y = x0 + x1*theta1 ....
                    dCrossE += (self.sigmoid(y_hat) - y_train.iloc[k])*X_train.iloc[k,j] # err: (y_hat-y)
                
                # THETA UPDATED!
                THETA[j] = theta_old[j] - (lr/batch_size)*dCrossE     
        self.coef_= THETA
        return

    def CrossE_function(self,THETA):
        # Helper function for autograd calculations
        y_hat = self.sigmoid(anp.dot(np.array(self.X),THETA))
        CrossE = -(anp.dot(self.y.T,anp.log(y_hat)) + anp.dot((anp.ones(self.y.shape)-self.y).T,anp.log(anp.ones(self.y.shape)-y_hat)))
        # MSE = anp.sum(anp.square(anp.subtract(y_hat, self.y)))/len(self.y)
        return CrossE

    def CrossE_l1_function(self,THETA):
        # Helper function for autograd calculations
        y_hat = self.sigmoid(anp.dot(np.array(self.X),THETA))
        CrossE = -(anp.dot(self.y.T,anp.log(y_hat)) + anp.dot((anp.ones(self.y.shape)-self.y).T,anp.log(anp.ones(self.y.shape)-y_hat)))
        # MSE = anp.sum(anp.square(anp.subtract(y_hat, self.y)))/len(self.y)
        return CrossE + self.lamda*anp.sum(anp.abs(THETA))

    def CrossE_l2_function(self,THETA):
        # Helper function for autograd calculations
        y_hat = self.sigmoid(anp.dot(np.array(self.X),THETA))
        CrossE = -(anp.dot(self.y.T,anp.log(y_hat)) + anp.dot((anp.ones(self.y.shape)-self.y).T,anp.log(anp.ones(self.y.shape)-y_hat)))
        # MSE = anp.sum(anp.square(anp.subtract(y_hat, self.y)))/len(self.y)
        return CrossE + self.lamda*anp.dot(THETA,THETA)
    

    def fit_autograd(self, X, y, batch_size, n_iter=10, lr=0.01, lr_type='constant', reg_type = "", lamda = 0.5):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''

        #______ VECTORISED FITTING WITH AUTOGRAD_______#
        N = len(X)
        self.n_iter = n_iter
        n_batches = N//batch_size
        LR = lr
        if(self.fit_intercept):
            X = pd.concat([pd.Series([1.0]*N, index = X.index),X],axis=1, ignore_index=True)

        # Dividing into batches
        Xbatches = [X.iloc[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]
        ybatches = [y.iloc[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]

        # Initializing with all Thetas = 0
        THETA = np.array([0.0]*len(X.columns)).T
        self.lamda = lamda
        for i in range(n_iter):
            if(lr_type=='inverse'):
                lr = LR/(i+1)
            X_train, y_train = Xbatches[i%n_batches], ybatches[i%n_batches]
            y_hat = X_train.dot(THETA) # y_hat = X0 
            self.X = X_train
            self.y = y_train
            
            # Gradient calculation using Autograd
            if(reg_type == "l1"):
                dCrossE_constructor =  grad(self.CrossE_l1_function)
            elif(reg_type == "l2"):
                dCrossE_constructor =  grad(self.CrossE_l2_function)
            else:
                dCrossE_constructor =  grad(self.CrossE_function)
            dCrossE = dCrossE_constructor(THETA)          
            THETA = THETA - (lr/batch_size)* dCrossE # 0 = 0 - alpha.(GRAD)/batch_size
        self.coef_= THETA

    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        if(self.fit_intercept):
            X = pd.concat([pd.Series([1]*len(X), index = X.index),X],axis=1, ignore_index=True)
        y_hat = pd.Series(X.dot(self.coef_))
        y_hat[y_hat<0] = 0
        y_hat[y_hat>0] = 1
        return y_hat

    def plot_desicion_boundary(self, X, y): # x1.X1 + x2.X2 + x3 = 0
        fig = plt.figure()
        c,x1,x2 = list(self.coef_)
        m = -x1/x2
        c /= -x2
        xmin, xmax, ymin, ymax = -1.5, 1.5, -1, 1
        Xs = np.array([xmin, xmax])
        ys = m*Xs + c
        plt.plot(Xs, ys, 'k', lw=1, ls='--')
        plt.fill_between(Xs, ys, ymin, color='tab:blue', alpha=0.2)
        plt.fill_between(Xs, ys, ymax, color='tab:orange', alpha=0.2)
        plt.scatter(X[y==0][0],X[y==0][1],s=8,alpha=0.5,cmap='Paired',label = "0")
        plt.scatter(X[y==1][0],X[y==1][1],s=8,alpha=0.5,cmap='Paired',label = "1")
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.ylabel(r'$x_2$')
        plt.xlabel(r'$x_1$')
        plt.show()
    

    def softmax(self, X, k, THETA):
        P = anp.exp(anp.dot(X,THETA))
        return P[:,k] / anp.sum(P,axis=1)

    def CrossE_multi(self, THETA):
        P = anp.exp(anp.dot(np.array(self.X),THETA)) 
        P /= anp.sum(P,axis=1).reshape(-1,1)
        CrossE = 0
        for k in self.classes:
            CrossE -= anp.dot((self.y == k).astype(float),anp.log(P[:,k]))        
            
        return CrossE

    def fit_multi(self, X, y, batch_size, n_iter=10, lr=0.01, lr_type='constant', reg_type = "", classes = None):
        #______ VECTORISED FITTING WITH AUTOGRAD_______#
        N = len(X)
        self.n_iter = n_iter
        n_batches = N//batch_size
        LR = lr
        if(self.fit_intercept):
            X = pd.concat([pd.Series([1.0]*N, index = X.index),X],axis=1, ignore_index=True)
        # Dividing into batches
        Xbatches = [X.iloc[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]
        ybatches = [y.iloc[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]

        # Initializing with all Thetas = 0
        # THETA = (np.array([0.0]*len(X.columns)).T)*len(classes)
        classes = sorted(list(y.unique()))
        self.classes = classes
        THETA = np.array([[0.0]*len(X.columns)]*len(classes)).T

        for i in range(n_iter):
            if(lr_type=='inverse'):
                lr = LR/(i+1)
            X_train, y_train = Xbatches[i%n_batches], ybatches[i%n_batches]

            for k in classes:
                loss = - ((y_train == k).astype(float) - self.softmax(X_train, k, THETA))
                THETA[:,k] = THETA[:,k] - (lr/batch_size)* X_train.T.dot(loss)
        self.coef_= THETA
    
    def fit_multi_autograd(self, X, y, batch_size, n_iter=10, lr=0.01, lr_type='constant', reg_type = "", classes = None):
        #______ VECTORISED FITTING WITH AUTOGRAD_______#
        N = len(X)
        self.n_iter = n_iter
        n_batches = N//batch_size
        LR = lr
        if(self.fit_intercept):
            X = pd.concat([pd.Series([1.0]*N, index = X.index),X],axis=1, ignore_index=True)
        # Dividing into batches
        Xbatches = [X.iloc[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]
        ybatches = [y.iloc[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]

        # Initializing with all Thetas = 0
        # THETA = (np.array([0.0]*len(X.columns)).T)*len(classes)
        classes = sorted(list(y.unique()))
        self.classes = classes
        THETA = np.array([[0.0]*len(X.columns)]*len(classes)).T

        for i in range(n_iter):
            if(lr_type=='inverse'):
                lr = LR/(i+1)
            X_train, y_train = Xbatches[i%n_batches], ybatches[i%n_batches]
            self.X = X_train
            self.y = y_train
            # Gradient calculation using Autograd
            dCrossE_constructor =  grad(self.CrossE_multi)
            dCrossE = dCrossE_constructor(THETA)          
            THETA = THETA - (lr/batch_size)* dCrossE # 0 = 0 - alpha.(GRAD)/batch_size
        self.coef_= THETA

    def predict_multi(self,X):
        if(self.fit_intercept):
            X = pd.concat([pd.Series([1]*len(X), index = X.index),X],axis=1, ignore_index=True)
        y_hat = np.zeros_like(X.dot(self.coef_))
        for k in self.classes:
            y_hat[:,k] = self.softmax(X, k, self.coef_)
        return np.argmax(y_hat,axis=1)