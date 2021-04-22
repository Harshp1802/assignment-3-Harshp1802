import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression
from metrics import *
import time
from tqdm import tqdm
np.random.seed(42)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
fit_intercept = True

# PLOT-1 (Varying N with fixed no. of features = 15 [Time vs N])

P = 15
n_iter = 100
n_t, p_t = [], []
N_range = list(range(400,2800,200))
print("\n|--------- Plotting Time with varying N ----------|")

for N in tqdm(N_range):
    X = pd.DataFrame(np.random.randn(N, P))
    y = pd.Series(np.random.randint(2,size=N))
    LR = LogisticRegression(fit_intercept=fit_intercept)
    start = time.time()
    LR.fit_autograd(X, y, n_iter = n_iter, batch_size=len(X))
    end = time.time()
    normal_time = end - start
    n_t.append(normal_time)

    start = time.time()
    LR.predict(X)
    end = time.time()
    normal_time = end - start
    p_t.append(normal_time)
    

fig1 = plt.figure()
plt.title("Time vs varying N with m ={}, n_iter = {}".format(P,n_iter))
plt.xlabel("N")
plt.plot(N_range[5:],n_t[5:], label= "Train")
plt.plot(N_range[5:],p_t[5:], label= "Predict")
plt.legend()
fig1.savefig("q4_plots/Varying_N.png")

# PLOT-2 (Varying M with fixed no. of samples = 60 [Time vs M])

N = 200
n_iter = 30
n_t,p_t = [], []
M_range = list(range(10,500,10))
print("\n|--------- Plotting Time with varying M ----------|")

for P in tqdm(M_range):
    X = pd.DataFrame(np.random.randn(N, P))
    y = pd.Series(np.random.randint(2,size=N))
    LR = LogisticRegression(fit_intercept=fit_intercept)
    start = time.time()
    LR.fit_autograd(X, y, n_iter = n_iter, batch_size=len(X))
    end = time.time()
    normal_time = end - start
    n_t.append(normal_time)
    
    start = time.time()
    LR.predict(X)
    end = time.time()
    normal_time = end - start
    p_t.append(normal_time)
    

fig2 = plt.figure()
plt.title("Time vs varying M with N ={}, n_iter = {}".format(N,n_iter))
plt.xlabel("M")
plt.plot(M_range,n_t, label = "Train")
plt.plot(M_range,p_t, label= "Predict")
plt.legend()
fig2.savefig("q4_plots/Varying_M.png")

# PLOT-3 (Varying n_iter with fixed no. of samples = 60 , fixed_M = 5[Time vs M])

N = 300
P = 25
n_t,p_t = [],[]
N_iter_range = list(range(10,200,10))
print("\n|--------- Plotting Time with varying n_iter ----------|")
for n_iter in tqdm(N_iter_range):
    X = pd.DataFrame(np.random.randn(N, P))
    y = pd.Series(np.random.randint(2,size=N))
    LR = LogisticRegression(fit_intercept=fit_intercept)
    start = time.time()
    LR.fit_autograd(X, y, n_iter = n_iter, batch_size=len(X))
    end = time.time()
    normal_time = end - start
    n_t.append(normal_time)

    start = time.time()
    LR.predict(X)
    end = time.time()
    normal_time = end - start
    p_t.append(normal_time)

fig3 = plt.figure()
plt.title("Time vs varying n_iter with N ={}, M = {}".format(N,P))
plt.xlabel("n_iter")
plt.plot(N_iter_range,n_t, label = "Train")
plt.plot(N_iter_range,p_t, label= "Predict")
plt.legend()
fig3.savefig("q4_plots/Varying_n_iter.png")