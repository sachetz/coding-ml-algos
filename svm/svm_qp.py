import argparse
import cvxopt
import utils
import numpy as np


def predict(X, w, bias):
    '''
    Parameters
    ----------
    X: matrix of shape (n, d)
       Training data

    w: matrix of shape (d, 1)
       SVM weight vector

    bias: scalar

    Returns
    -------
    y_pred: matrix of shape (n, 1)
            Predicted values
    '''
    y_pred = np.sign(X.dot(w) + bias)
    return y_pred

def make_P(X, y):
    '''
    Parameters
    ----------
    X: matrix of shape (n, d)
       Training data
    
    y: matrix of shape (n, 1)
       Target values

    Returns
    -------
    P: matrix of shape (n, n)
       positive semidefinite matrix for the quadratic program
    '''
    P = X.dot(X.T).multiply(np.dot(y, y.T))
    return P.toarray()

def make_q(n):
    '''
    Return the q vector in the standard quadratic program formulation of the SVM dual problem

    Parameters
    ----------
    n: dim of the matrix

    Returns
    -------
    q: matrix of shape (n, 1)
       positive semidefinite matrix for the quadratic program
    '''
    q = -1 * np.ones((n, 1))
    return q

def make_inequality_constraints(C, n):
    '''
    Return the G, h matrices/vectors in the standard quadratic program formulation
        for the SVM dual problem

    Parameters
    ----------
    C: regularization parameter

    n: dim of the matrix

    Returns
    -------
    G: matrix of shape (m, n)

    h: matrix of shape (m, 1)
    '''
    G = np.concatenate([np.identity(n), -1 * np.identity(n)], axis=0)
    h = np.concatenate([C * np.ones((n, 1)), np.zeros((n, 1))], axis=0)
    return G, h

def make_equality_constraints(y):
    '''
    Return the A, b matrices/vectors in the standard quadratic program for the SVM dual problem

    Parameters
    ----------
    y: matrix of shape (n, 1)
       Target values

    Returns
    -------
    A: matrix of shape (p, n)

    b: matrix of shape (p, 1)
    '''
    A = y.T
    b = np.zeros((1, 1))
    return A, b

def accuracy(X, y, w, bias):
    '''
    Compute the accuracy of the prediction rule determined by the
        given weight and bias vector on input data X, y

    Parameters
    ----------
    X: matrix of shape (n, d)
       Training data

    y: matrix of shape (n, 1)
       Target values

    w: matrix of shape (d, 1)
       SVM weight vector

    bias: scalar
          SVM bias term

    Returns
    -------
    acc: float
         accuracy
    '''
    return np.mean(y == predict(X, w, bias))

def make_weight_bias(X, y, qp_solution):
    '''
    Given the solution of the SVM dual quadratic program
    construct the corresponding w weight vector

    Parameters
    ----------
    X: matrix of shape (n, d)
       Training data

    y: matrix of shape (n, 1)
       Target values

    qp_solution: output of cvxopt.solvers.qp

    Returns
    -------
    w: vector of shape (d, 1)
       SVM weight vector
    bias: scalar
          bias term
    '''
    w = X.T * np.multiply(np.array(qp_solution["x"]), y)
    mask = (np.array(qp_solution["x"]) > 0).flatten()
    bias = np.mean(y[mask] - X[mask].dot(w).flatten())
    return w, bias

def dual_svm(X_train, y_train, X_test, y_test, C):
    '''
    Minimize     1/2 alpha^T P alpha - q^T x
    Subject to   Gx <= h
                 Ax  = b

    here alphas = x
    G = X @ X.T
    '''

    P = make_P(X_train, y_train)
    P = cvxopt.matrix(P)
    q = make_q(X_train.shape[0])
    q = cvxopt.matrix(q)
    G, h = make_inequality_constraints(C, X_train.shape[0])
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)
    A, b = make_equality_constraints(y_train)
    A = cvxopt.matrix(A)
    b = cvxopt.matrix(b)

    # Note that the cvxopt.solvers.qp function expects objects of type cvxopt.matrix
    # See: https://cvxopt.org/examples/tutorial/numpy.html
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    weight, bias = make_weight_bias(X_train, y_train, sol)
    test_acc = accuracy(X_test, y_test, weight, bias)
    train_acc = accuracy(X_train, y_train, weight, bias)
    print("Train acc: {:.3f}".format(train_acc))
    print("Test acc: {:.3f}".format(test_acc))

def main(args):
    # Note that we do not add bias here
    X_train, y_train, X_test, y_test = utils.load_data(args.fname, add_bias=False)
    dual_svm(X_train, y_train, X_test, y_test, args.C)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, default='news.mat')
    parser.add_argument('--C', type=float, default=1.0)
    args = parser.parse_args()
    main(args)
