import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import Autograd modules here
from autograd import grad, numpy as anp
import matplotlib.animation as animation
from math import e

class LogisticRegression():
    def __init__(self, fit_intercept=False):
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
            X = pd.concat([pd.Series([1]*N),X],axis=1, ignore_index=True)

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
        T1 =  -anp.dot(self.y.T,anp.log(y_hat))
        T2 = -anp.dot((anp.ones(self.y.shape)-self.y).T,anp.log(anp.ones(self.y.shape)-y_hat))
        CrossE = anp.sum(T1 + T2)
        # MSE = anp.sum(anp.square(anp.subtract(y_hat, self.y)))/len(self.y)
        return CrossE

    def fit_autograd(self, X, y, batch_size, n_iter=10, lr=0.01, lr_type='constant'):
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
            X = pd.concat([pd.Series([1]*N),X],axis=1, ignore_index=True)

        # Dividing into batches
        Xbatches = [X.iloc[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]
        ybatches = [y.iloc[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]

        # Initializing with all Thetas = 0
        THETA = np.array([0.0]*len(X.columns)).T
        
        for i in range(n_iter):
            if(lr_type=='inverse'):
                lr = LR/(i+1)
            X_train, y_train = Xbatches[i%n_batches], ybatches[i%n_batches]
            y_hat = X_train.dot(THETA) # y_hat = X0 
            self.X = X_train
            self.y = y_train
            
            # Gradient calculation using Autograd
            dCrossE_constructor =  grad(self.CrossE_function)
            dCrossE = dCrossE_constructor(THETA)          
            THETA = THETA - (lr/batch_size)* dCrossE # 0 = 0 - alpha.(GRAD)/batch_size
        self.coef_= THETA


    def fit_normal(self, X, y):
        '''
        Function to train model using the normal equation method.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))

        :return None
        '''
        #______ NORMAL FITTING _______#
        if(self.fit_intercept):
            X = pd.concat([pd.Series([1]*len(X)),X],axis=1, ignore_index=True)
        self.coef_ = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
        return

    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        if(self.fit_intercept):
            X = pd.concat([pd.Series([1]*len(X)),X],axis=1, ignore_index=True)
        y_hat = pd.Series(X.dot(self.coef_))
        y_hat[y_hat<0] = 0
        y_hat[y_hat>0] = 1
        print(list(y_hat))
        return y_hat

    def RSS(self, t_0, t_1):
        # Resuidal Sum of Squares calculation for Plots
        return np.sum(np.square(np.subtract(self.X*t_1 + t_0, self.y)))

    def plot_surface(self, X, y, t_0, t_1):
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        THETA_0 and THETA_1 over a range. Indicates the RSS based on given value of t_0 and t_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.

        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of THETA_0 for which to indicate RSS
        :param t_1: Value of THETA_1 for which to indicate RSS

        :return matplotlib figure plotting RSS
        """
        fig = plt.figure()
        # Centering around the optimal values
        x1_values = np.arange(-10 + t_0, 10 + t_0, 0.25) 
        x2_values = np.arange(-10 + t_1, 10 + t_1, 0.25)
        Xs, Ys = np.meshgrid(x1_values, x2_values) # RECTANGULAR GRID!
        XYs = pd.concat([pd.Series(Xs.flatten()), pd.Series(Ys.flatten())],axis = 1)
        self.X = X
        self.y = y
        # Corresponding RSS for each pair of thetas in the grid
        Zs = XYs.apply(lambda x: self.RSS(x[0], x[1]), axis = 1).to_numpy().reshape(Xs.shape)
        ax = fig.add_subplot(projection='3d')
        ax.scatter(self.coef_[0], self.coef_[1], self.RSS(self.coef_[0], self.coef_[1]), s=100, color="red")
        surf = ax.plot_surface(Xs, Ys, Zs, cmap='viridis', alpha=0.2, linewidth=0)
        fig.colorbar(surf, shrink=0.5, aspect=10)
        return fig

    def plot_line_fit(self, X, y, t_0, t_1):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of t_0, t_1. Plot must
        indicate t_0 and t_1 as the title.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of THETA_0 for which to plot the fit
        :param t_1: Value of THETA_1 for which to plot the fit

        :return matplotlib figure plotting line fit
        """
        fig = plt.figure()
        plt.scatter(X,y)
        plt.plot(np.array(X), np.array(X*t_1 + t_0), "r")
        plt.title("t_0 = {}   |   t_1 = {}".format(t_0,t_1))
        plt.xlabel('X')
        plt.ylabel('y')
        plt.ylim(y.min()-10, y.max()+10)
        return fig

    def plot_contour(self, X, y, t_0, t_1):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        THETA_0 and THETA_1 over a range. Indicates t   he RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of THETA_0 for which to plot the fit
        :param t_1: Value of THETA_1 for which to plot the fit

        :return matplotlib figure plotting the contour
        """
        fig = plt.figure()
        # Centering around the optimal values
        x1_values = np.arange(-10 + t_0, 10 + t_0, 0.25)
        x2_values = np.arange(-10 + t_1, 10 + t_1, 0.25)
        Xs, Ys = np.meshgrid(x1_values, x2_values) # RECTANGULAR GRID!
        XYs = pd.concat([pd.Series(Xs.flatten()), pd.Series(Ys.flatten())],axis = 1)
        self.X = X
        self.y = y
        # Corresponding RSS for each pair of thetas in the grid
        Zs = XYs.apply(lambda x: self.RSS(x[0], x[1]), axis = 1).to_numpy().reshape(Xs.shape)
        ax = fig.add_subplot()
        contour = ax.contourf(Xs, Ys, Zs, cmap='viridis', alpha=0.35)
        if(self.fit_intercept):
            X = pd.concat([pd.Series([1]*len(X)),X],axis=1, ignore_index=True)
        dMSE = 0.01*(X.T.dot(X.dot(self.coef_) - y))/len(X)
        
        ax.annotate('', xy=(self.coef_[0],self.coef_[1]), xytext=(self.coef_[0]+dMSE[0],self.coef_[1]+dMSE[1]),
                    arrowprops={'arrowstyle': '-|>', 'color': 'r', 'lw': 2}, va='center', ha='center')
        fig.colorbar(contour, shrink=0.6, aspect=10)
        return fig