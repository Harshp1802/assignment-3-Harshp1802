import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression

np.random.seed(42)

X = pd.Series(np.random.randn(100,))
y = 4*X + 7 + pd.Series(np.random.normal(0,5,len(X)))

LR = LinearRegression(fit_intercept=True)
LR.fit_vectorised(X, y, n_iter = 100,batch_size = 10,lr = 0.085)
Central_THETA = LR.coef_

for i in range(1,11):
    LR = LinearRegression(fit_intercept=True)
    LR.fit_vectorised(X, y, n_iter=i,batch_size = 10,lr = 0.085)
    THETA = LR.coef_
    #___________LINE PLOT___________#
    fig = LR.plot_line_fit(X,y,THETA[0],THETA[1])
    fig.savefig("./Q7_line_plots/{}.png".format(i))
    #___________SURFACE PLOT___________#
    fig = LR.plot_surface(X,y,Central_THETA[0],Central_THETA[1])
    fig.savefig("./Q7_surface_plots/{}.png".format(i))
    #___________CONTOUR PLOT___________#
    fig = LR.plot_contour(X,y,Central_THETA[0],Central_THETA[1])
    fig.savefig("./Q7_contour_plots/{}.png".format(i))

    