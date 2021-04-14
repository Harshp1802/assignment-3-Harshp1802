import numpy as np
import pandas as pd
def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    y_hat = pd.Series(y_hat)
    return np.mean(y.reset_index(drop = True) == y_hat.reset_index(drop = True))

def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """

    assert(y_hat.size == y.size)
    y_hat = pd.Series(y_hat)
    y = y.reset_index(drop = True)
    y_hat = y_hat.reset_index(drop = True)
    counter = y_hat.value_counts().to_dict() # Counter stores the frequencies of each class in the Target
    try:
        if(counter[cls] != 0):
            return len(y[(y == y_hat)  & (y == cls) & (y_hat == cls)])/counter[cls]
    except:
        pass
    return 0

def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """

    assert(y_hat.size == y.size)
    y_hat = pd.Series(y_hat)
    y = y.reset_index(drop = True)
    y_hat = y_hat.reset_index(drop = True)
    counter = y.value_counts().to_dict() # Counter stores the frequencies of each class in the Expected Target
    try:
        if(counter[cls] != 0):
            return len(y[(y == y_hat)  & (y == cls) & (y_hat == cls)])/counter[cls]
    except:
        pass
    return 0
   
def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """

    assert(y_hat.size == y.size)
    y_hat = pd.Series(y_hat)
    y, y_hat = y.reset_index(drop = True), y_hat.reset_index(drop = True)
    return np.sqrt(np.square(np.subtract(y_hat,y)).mean())

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """

    assert(y_hat.size == y.size)
    y, y_hat = y.reset_index(drop = True), y_hat.reset_index(drop = True)
    return np.absolute(np.subtract(y_hat,y)).mean()
