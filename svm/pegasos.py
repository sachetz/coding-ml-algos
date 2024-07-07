import argparse
import numpy as np
import utils
import matplotlib.pyplot as plt


def hinge_loss(w, X, y):
    '''
    Compute the hinge loss:
    hinge loss = \frac{1}{n} \sum_{i=1}^{n} max(0, 1 - y_i w^\top x_i)

    Note: this is mostly used to compute the loss to keep track of
    the training progress so we do not include the regularization term here.

    Parameters
    ----------
    w: matrix of shape (d, 1)
       Weight vector

    X: matrix of shape (n, d)
        Training data

    y: matrix of shape (n, 1)
        Target values

    Returns
    -------
    loss: scalar
          SVM hinge loss
    '''
    loss = np.mean(np.maximum(0, 1 - np.multiply(y, X.dot(w))))
    return loss

def eval_model(w, X, y):
    '''
    Return a tuple of the hinge loss and accuracy

    Parameters
    ----------
    w: matrix of shape (d, 1)
       Weight vector

    X: matrix of shape (n, d)
       Training data

    y: matrix of shape (n, 1)
       Target values

    Returns
    -------
    loss: float
          SVM hinge loss

    acc: float
         accuracy
    '''
    loss = hinge_loss(w, X, y)
    acc = np.mean(y == np.sign(X.dot(w)))
    return (loss, acc)

def grad(w, X, y, lamda):
    '''
    Parameters
    ----------
    w: matrix of shape (d, 1)
       Weight vector

    X: matrix of shape (n, d)
        Training data

    y: matrix of shape (n, 1)
        Target values

    lamda: scalar
           learning rate scaling parameter

    Returns
    -------
    grad_w: matrix of shape (d, 1)
            Gradient of of the SVM primal objective with respect to w over the given
            data X, y
    '''
    mask = np.where(np.multiply(y, X.dot(w)) < 1)[0]
    gradient = np.multiply(w, lamda) - X[mask].multiply(y[mask]).mean(axis=0).reshape(-1, 1)
    return gradient

def get_batch(X, y, batch_size):
    '''
    Parameters
    ----------
    X: matrix of shape (n, d)
        Training data

    y: matrix of shape (n, 1)
        Target values

    batch_size: int
                size of batch

    Returns
    -------
    X_batch: matrix of shape (batch_size, d)

    y_batch: matrix of shape (batch_size, 1)
    '''
    idx = np.random.choice(X.shape[0], size=batch_size)
    X_batch = X[idx]
    y_batch = y[idx]
    return X_batch, y_batch

def objective(w, X, y, lamda):
    return (lamda * (np.linalg.norm(w) ** 2)) - hinge_loss(w, X, y)

def train_svm(args):
    X_train, y_train, X_test, y_test = utils.load_data(args.fname, add_bias=True)
    n, d = X_train.shape
    w = np.zeros((d, 1))

    # Pegasos training loop
    # 1) grab a batch of data
    # 2) compute the gradient
    # 3) update the weight vector
    train_arr_loss = []
    train_arr_accuracy = []
    train_arr_objective = []
    test_arr_loss = []
    test_arr_accuracy = []
    test_arr_objective = []
    for i in range(args.epochs):
        X_batch, y_batch = get_batch(X_train, y_train, args.batch_size)
        eta_t = 1 / (args.lamda * (i + 1))
        w = w - eta_t * grad(w, X_batch, y_batch, args.lamda)

        train_loss, train_acc = eval_model(w, X_train, y_train)
        test_loss, test_acc = eval_model(w, X_test, y_test)

        train_arr_loss.append(train_loss)
        train_arr_accuracy.append(train_acc)
        test_arr_loss.append(test_loss)
        test_arr_accuracy.append(test_acc)
        train_arr_objective.append(objective(w, X_train, y_train, args.lamda))
        test_arr_objective.append(objective(w, X_test, y_test, args.lamda))
    
    print('Train acc: {:.3f} | Test acc: {:.3f}'.format(train_acc, test_acc))
    
    plt.figure()
    plt.plot([i for i in range(args.epochs)], train_arr_loss, label="Training Loss")
    plt.plot([i for i in range(args.epochs)], test_arr_loss, label="Test Loss")
    plt.title("Loss vs Number of Epochs")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot([i for i in range(args.epochs)], train_arr_accuracy, label="Training Accuracy")
    plt.plot([i for i in range(args.epochs)], test_arr_accuracy, label="Test Accuracy")
    plt.title("Accuracy vs Number of Epochs")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot([i for i in range(args.epochs)], train_arr_objective, label="Training Objective")
    plt.plot([i for i in range(args.epochs)], test_arr_objective, label="Test Objective")
    plt.title("Objective vs Number of Epochs")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, default='news.mat')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lamda', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    train_svm(args)
