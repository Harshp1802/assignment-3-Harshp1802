import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *
import time
np.random.seed(42)

fit_intercept = True


# PLOT-1 (Varying N with fixed no. of features = 5 [Time vs N])

P = 15
n_iter = 10
n_t,g_t = [],[]
N_range = list(range(1000,10000,200))
for N in N_range:

    #_____ NORMAL _____#
    X = pd.DataFrame(np.random.randn(N, P))
    y = pd.Series(np.random.randn(N))
    
    LR = LinearRegression(fit_intercept=fit_intercept)
    start = time.time()
    LR.fit_normal(X, y)
    end = time.time()
    normal_time = end - start
    n_t.append(normal_time)

    #_____ Gradient Descent _____#
    
    LR = LinearRegression(fit_intercept=fit_intercept)
    start = time.time()
    LR.fit_vectorised(X, y, n_iter=n_iter,batch_size=len(X),lr=0.01) # here you can use fit_non_vectorised / fit_autograd methods
    end = time.time()
    gradient_time = end - start
    g_t.append(gradient_time)
fig1 = plt.figure()
plt.title("Time vs varying N with m ={}, n_iter = {}".format(P,n_iter))
plt.xlabel("N")
plt.plot(N_range[5:],n_t[5:], label= "Normal Time")
plt.plot(N_range[5:],g_t[5:], label= "Gradient Descent Time")
plt.legend()
fig1.savefig("Q8_plots/Varying_N.png")

# PLOT-2 (Varying M with fixed no. of samples = 60 [Time vs M])

N = 200
n_iter = 10
n_t,g_t = [],[]
M_range = list(range(5,400,10))
for P in M_range:

    #_____ NORMAL _____#
    X = pd.DataFrame(np.random.randn(N, P))
    y = pd.Series(np.random.randn(N))
    
    LR = LinearRegression(fit_intercept=fit_intercept)
    start = time.time()
    LR.fit_normal(X, y)
    end = time.time()
    normal_time = end - start
    n_t.append(normal_time)

    #_____ Gradient Descent _____#
    LR = LinearRegression(fit_intercept=fit_intercept)
    start = time.time()
    LR.fit_vectorised(X, y, n_iter=n_iter,batch_size=len(X),lr=0.01) # here you can use fit_non_vectorised / fit_autograd methods
    end = time.time()
    gradient_time = end - start
    g_t.append(gradient_time)
fig2 = plt.figure()
plt.title("Time vs varying M with N ={}, n_iter = {}".format(N,n_iter))
plt.xlabel("M")
plt.plot(M_range,n_t, label= "Normal Time")
plt.plot(M_range,g_t, label= "Gradient Descent Time")
plt.legend()
fig2.savefig("Q8_plots/Varying_M.png")

# PLOT-3 (Varying n_iter with fixed no. of samples = 60 , fixed_M = 5[Time vs M])

N = 300
P = 25
n_t,g_t = [],[]
N_iter_range = list(range(5,400,10))
for n_iter in N_iter_range:

    #_____ NORMAL _____#
    X = pd.DataFrame(np.random.randn(N, P))
    y = pd.Series(np.random.randn(N))
    LR = LinearRegression(fit_intercept=fit_intercept)
    start = time.time()
    LR.fit_normal(X, y)
    end = time.time()
    normal_time = end - start
    n_t.append(normal_time)

    #_____ Gradient Descent _____#
    LR = LinearRegression(fit_intercept=fit_intercept)
    start = time.time()
    LR.fit_vectorised(X, y, n_iter=n_iter,batch_size=len(X),lr=0.01) # here you can use fit_non_vectorised / fit_autograd methods
    end = time.time()
    gradient_time = end - start
    g_t.append(gradient_time)
fig3 = plt.figure()
plt.title("Time vs varying n_iter with N ={}, M = {}".format(N,P))
plt.xlabel("n_iter")
plt.plot(N_iter_range,n_t, label= "Normal Time")
plt.plot(N_iter_range,g_t, label= "Gradient Descent Time")
plt.legend()
fig3.savefig("Q8_plots/Varying_n_iter.png")

