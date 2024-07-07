import numpy as np
from scipy.sparse import hstack
from scipy.io import loadmat

def load_data(fname, add_bias=True):
    '''
    Parameters
    ----------
    fname: string
           location of .mat data
    add_bias: boolean
              whether or not to add a column of 1s to the X data

    Returns
    -------
    X_train: matrix of shape (n, d)
             Training data

    y_train: matrix of shape (n, 1)
             Train target values

    X_test: matrix of shape (m, d)
            Test data

    y_test: matrix of shape (m, 1)
            Test target values
    '''
    d = loadmat(fname)
    X_train = d['X_train'].astype(float)
    X_test = d['X_test'].astype(float)
    y_train = d['y_train'].reshape(-1, 1).astype(float)
    y_test = d['y_test'].reshape(-1, 1).astype(float)

    if add_bias:
        X_train = hstack([X_train, np.ones((X_train.shape[0],1))])
        X_test = hstack([X_test, np.ones((X_test.shape[0],1))])

    # convert to compressed sparse row format for ease of row indexing
    return X_train.tocsr(), y_train, X_test.tocsr(), y_test
