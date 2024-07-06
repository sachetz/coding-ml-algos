import pdb
import time
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt

DT = np.float32
eps = 1e-12 # Used for gradient testing

# Utility function for shape inference with broadcasting.
def bcast(a, b):
    xs = np.array(a.shape)
    ys = np.array(b.shape)
    pad = len(xs) - len(ys)
    if pad > 0:
        ys = np.pad(ys, [[pad, 0]], 'constant')
    elif pad < 0:
        xs = np.pad(xs, [[-pad, 0]], 'constant')
    os = np.maximum(xs, ys)
    xred = tuple([idx for idx in np.where(xs < os)][0])
    yred = tuple([idx for idx in np.where(ys < os)][0])
    return xred, yred


def xavier(shape, seed=None):
    # Return an np array of given shape in which each element is chosen uniformly
    # at random from the Xavier initialization interval.
    n_in, n_out = shape
    if seed is not None:
        # Set seed to fixed number (e.g. layer idx) for predictable results
        np.random.seed(seed)
    # Initialize uniformly at random from [-sqrt(6/(n_in+n_out)), sqrt(6/(n_in+n_out))]
    m = np.sqrt(6 / (n_in + n_out))
    weight = np.random.uniform(-1 * m, m, size=(n_in, n_out))
    return weight


# Values
# This is used for nodes corresponding to input values or other constants
# that are not parameters (i.e., nodes that are not updated via gradient descent).
class Value:
    def __init__(self, value=None):
        self.value = DT(value).copy()
        self.grad = None

    def set(self, value):
        self.value = DT(value).copy()


# Parameters
class Param:
    def __init__(self, value):
        self.value = DT(value).copy()
        self.grad = DT(0)


'''
  Class name: Add
  Class usage: add two matrices a, b with broadcasting supported by numpy "+" operation.
  Class function:
      forward: calculate a + b with possible broadcasting
      backward: calculate derivative w.r.t to a and b
'''
class Add:  # Add with broadcasting
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.grad = None if a.grad is None and b.grad is None else DT(0)
        self.value = None

    def forward(self):
        self.value = self.a.value + self.b.value

    def backward(self):
        xred, yred = bcast(self.a.value, self.b.value)
        if self.a.grad is not None:
            self.a.grad = self.a.grad + np.reshape(
                np.sum(self.grad, axis=xred, keepdims=True),
                self.a.value.shape)

        if self.b.grad is not None:
            self.b.grad = self.b.grad + np.reshape(
                np.sum(self.grad, axis=yred, keepdims=True),
                self.b.value.shape)


'''
Class Name: Mul
Class Usage: elementwise multiplication with two matrix 
Class Functions:
    forward: compute the result a*b
    backward: compute the derivative w.r.t a and b
'''
class Mul:  # Multiply with broadcasting
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.grad = None if a.grad is None and b.grad is None else DT(0)
        self.value = None

    def forward(self):
        self.value = self.a.value * self.b.value

    def backward(self):
        xred, yred = bcast(self.a.value, self.b.value)    
        if self.a.grad is not None:
            self.a.grad = self.a.grad + np.reshape(
                np.sum(self.grad * self.b.value, axis=xred, keepdims=True),
                self.a.value.shape)

        if self.b.grad is not None:
            self.b.grad = self.b.grad + np.reshape(
                np.sum(self.grad * self.a.value, axis=yred, keepdims=True),
                self.b.value.shape)



"""
Class Name: VDot
Class Usage: matrix multiplication where a is a vector and b is a matrix
    b is expected to be a parameter and there is a convention that parameters come last. 
    Typical usage is a is a feature vector with shape (f_dim, ), b a parameter with shape (f_dim, f_dim2).
Class Functions:
     forward: compute the vector matrix multplication result
     backward: compute the derivative w.r.t a and b, where derivative of a and b are both matrices 
"""
class VDot:  # Matrix multiply (fully-connected layer)
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.grad = None if a.grad is None and b.grad is None else DT(0)
        self.value = None

    def forward(self):
        self.value = self.a.value @ self.b.value

    def backward(self):
        if self.a.grad is not None:
            self.a.grad += self.grad @ self.b.value.T
        if self.b.grad is not None:
            self.b.grad += self.a.value.reshape(-1, 1) @ self.grad.reshape(1, -1)


'''
Class Name: Sigmoid
Class Usage: compute the elementwise sigmoid activation. Input is vector or matrix. 
    In case of vector, [a_{0}, a_{1}, ..., a_{n}], output is vector [b_{0}, b_{1}, ..., b_{n}] where b_{i} = 1/(1 + exp(-a_{i}))
Class Functions:
    forward: compute activation b_{i} for all i.
    backward: compute the derivative w.r.t input vector/matrix a  
'''
class Sigmoid:
    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DT(0)
        self.value = None

    def forward(self):
        self.value = 1 / (1 + np.exp(-1 * self.a.value))

    def backward(self):
        if self.a.grad is not None:
            self.a.grad += self.grad * (np.exp(-1 * self.a.value) / (1 + np.exp(-1 * self.a.value)) ** 2)


'''
Class Name: RELU
Class Usage: compute the elementwise RELU activation. Input is vector or matrix. In case of vector, 
    [a_{0}, a_{1}, ..., a_{n}], output is vector [b_{0}, b_{1}, ..., b_{n}] where b_{i} = max(0, a_{i})
Class Functions:
    forward: compute activation b_{i} for all i.
    backward: compute the derivative w.r.t input vector/matrix a  
'''
class RELU:
    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DT(0)
        self.value = None

    def forward(self):
        self.value = np.maximum(0, self.a.value)

    def backward(self):
        if self.a.grad is not None:
            self.a.grad += (self.a.value > 0).astype(int) * self.grad


'''
Class Name: SoftMax
Class Usage: compute the softmax activation for each element in the matrix, normalization by each all elements 
    in each batch (row). Specifically, input is matrix [a_{00}, a_{01}, ..., a_{0n}, ..., a_{b0}, a_{b1}, ..., a_{bn}], 
    output is a matrix [p_{00}, p_{01}, ..., p_{0n},...,p_{b0},,,p_{bn} ] where p_{bi} = exp(a_{bi})/(exp(a_{b0}) + ... + exp(a_{bn}))
Class Functions:
    forward: compute probability p_{bi} for all b, i.
    backward: compute the derivative w.r.t input matrix a 
'''
class SoftMax:
    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DT(0)
        self.value = None

    def forward(self):
        self.value = np.exp(self.a.value) / np.sum(np.exp(self.a.value))

    def backward(self):
        if self.a.grad is not None:
            self.a.grad += (self.grad * self.value) - (self.value * np.dot(self.grad, self.value))


'''
Class Name: Log
Class Usage: compute the elementwise log(a) given a.
Class Functions:
    forward: compute log(a)
    backward: compute the derivative w.r.t input vector a
'''
class Log: # Elementwise Log
    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DT(0)
        self.value = None

    def forward(self):
        self.value = np.log(self.a.value)

    def backward(self):
        if self.a.grad is not None:
            self.a.grad += self.grad * (1 / self.a.value)


'''
Class Name: Aref
Class Usage: get some specific entry in a matrix. a is the matrix with shape (batch_size, N) and idx is vector containing 
    the entry index and a is differentiable.
Class Functions:
    forward: compute a[batch_size, idx]
    backward: compute the derivative w.r.t input matrix a
'''
class Aref:
    def __init__(self, a, idx):
        self.a = a
        self.idx = idx
        self.grad = None if a.grad is None else DT(0)

    def forward(self):
        xflat = self.a.value.reshape(-1)
        iflat = self.idx.value.reshape(-1)
        outer_dim = len(iflat)
        inner_dim = len(xflat) / outer_dim
        self.pick = np.int32(np.array(range(outer_dim)) * inner_dim + iflat)
        self.value = xflat[self.pick].reshape(self.idx.value.shape)

    def backward(self):
        if self.a.grad is not None:
            grad = np.zeros_like(self.a.value)
            gflat = grad.reshape(-1)
            gflat[self.pick] = self.grad.reshape(-1)
            self.a.grad = self.a.grad + grad


'''
Class Name: Accuracy
Class Usage: check the predicted label is correct or not. a is the probability vector where each probability is 
            for each class. idx is ground truth label.
Class Functions:
    forward: find the label that has maximum probability and compare it with the ground truth label.
    backward: None 
'''
class Accuracy:
    def __init__(self, a, idx):
        self.a = a
        self.idx = idx
        self.grad = None
        self.value = None

    def forward(self):
        self.value = np.mean(np.argmax(self.a.value, axis=-1) == self.idx.value)

    def backward(self):
        pass


# Set of allowed/implemented activation functions
ACTIVATIONS = {'relu': RELU,
               'sigmoid': Sigmoid}


class NN:
    def __init__(self, nodes_array, activation):
        # Assert nodes_array is a list of positive integers.
        assert all(isinstance(item, int) and item > 0 for item in nodes_array)
        # Assert activation is supported.
        assert activation in ACTIVATIONS.keys()
        self.nodes_array = nodes_array
        self.activation = activation
        self.activation_func = ACTIVATIONS[self.activation]
        # self.layer_number is the number of layers of neurons.  It is one 
        # less than len(nodes_array) because of the input layer.
        self.layer_number = len(nodes_array) - 1
        self.weights = []
        # self.params is a dictionary of trainable parameters
        self.params = {}
        # self.components is an ordered list representing the computational
        # graph
        self.components = []
        self.sample_placeholder = Value()
        self.label_placeholder = Value()
        self.pred_placeholder = None
        self.loss_placeholder = None
        self.accy_placeholder = None

    def nn_unary_op(self, op, a):
        """Add a unary operation object to the computational graph."""
        unary_op = op(a)
        print(f"Append <{unary_op.__class__.__name__}> to the computational"
              f" graph")
        self.components.append(unary_op)
        return unary_op

    def nn_binary_op(self, op, a, b):
        """Add a binary operation object to the computation graph."""
        binary_op = op(a, b)
        print(f"Append <{binary_op.__class__.__name__}> to the computational"
              f" graph")
        self.components.append(binary_op)
        return binary_op

    def set_params_by_dict(self, param_dict: dict):
        """
        :param param_dict: a dict of parameters (numpy arrays) with parameter 
                           names as keys
        """
        # self.params is a dictionary of trainable Param objects (not simply 
        # numpy arrays) with their names in the keys.  
        # First reset self.params to an empty dict. 
        # Then create a Param object for each item in param_dict and add 
        # it to self.params.
        self.params = {}
        for name, value in param_dict.items():
            self.params[name] = Param(value)

    def set_weights(self, weights):
        """
        :param weights: a list of tuples (matrices and vectors)
        :return:
        """
        # Assert weights have the right shapes.
        if len(weights) != self.layer_number:
            raise ValueError(
                f"You should provide weights for {self.layer_number}" 
                f" layers instead of {len(weights)}")
        for i, item in enumerate(weights):
            weight, bias = item
            if weight.shape != (self.nodes_array[i], self.nodes_array[i + 1]):
                raise ValueError(
                    f"The weight for the layer {i} should have shape"
                    f" ({self.nodes_array[i]}, "
                    f"{self.nodes_array[i + 1]}) instead of" 
                    f" {weight.shape}")
            if bias.shape != (self.nodes_array[i + 1],):
                raise ValueError(
                    f"The bias for the layer {i} should have shape"
                    f" ({self.nodes_array[i + 1]}, ) instead of "
                    f"{bias.shape}")

        # param_dict contains the trainable parameters; for each parameter
        # set a key to its name and the value to the corresponding numpy
        # array of weights.  Then call self.set_params_by_dict().
        param_dict = {}
        for i, item in enumerate(weights):
            weight, bias = item
            param_dict[f"weight{i}"] = weight
            param_dict[f"bias{i}"] = bias
        self.set_params_by_dict(param_dict)

    def get_weights(self):
        weights = []
        # Extract weight values from the list of Params
        for i in range(self.layer_number):
            weights.append((self.params[f"weight{i}"].value, self.params[f"bias{i}"].value))
        return weights

    def init_weights_with_xavier(self):
        """Initialize weight matrices using xavier initialization.

        Initialize weight matrices using xavier initialization, but initialize bias
        arrays using all zeros.
        """
        # Store the Xavier initialized matrices/arrays in a dictionary with
        # names as the keys, following the format in set_weights. Then
        # call set_params_by_dict to create corresponding Param
        # objects.
        params = {}
        for i in range(self.layer_number):
            n_in = self.nodes_array[i]
            n_out = self.nodes_array[i + 1]
            weight = xavier((n_in, n_out))
            params[f"weight{i}"] = weight
            params[f"bias{i}"] = np.zeros(n_out)
        self.set_params_by_dict(params)

    def build_computational_graph(self):
        """Build the computation graph.

        Specifically, add operations to the list of components; for each operation
        add its corresponding parameters.  End at the final SoftMax object.
        """
        # Reset computational graph to empty list
        self.components = []

        prev_output = self.sample_placeholder
        # Call from among self.nn_binary_op(VDot, x, y),
        # self.nn_binary_op(Add, x, y),
        # self.nn_unary_op(self.activation_func, x),
        # self.nn_unary_op(SoftMax, x), as needed, to construct the neural
        # network. Supports different number of layers.
        for i in range(self.layer_number):
            mul = self.nn_binary_op(VDot, prev_output, self.params[f"weight{i}"])
            add = self.nn_binary_op(Add, mul, self.params[f"bias{i}"])
            if i == self.layer_number - 1:
                prev_output = self.nn_unary_op(SoftMax, add)
            else:
                prev_output = self.nn_unary_op(self.activation_func, add)

        pred = prev_output
        return pred

    def cross_entropy_loss(self):
        label_prob = self.nn_binary_op(Aref, self.pred_placeholder, self.label_placeholder)
        log_prob = self.nn_unary_op(Log, label_prob)
        loss = self.nn_binary_op(Mul, log_prob, Value(-1))
        return loss

    def eval(self, X, y):
        if len(self.components)==0:
            raise ValueError("Computational graph not built yet. Call build_computational_graph first.")
        accuracy = 0.
        objective = 0.
        for k in range(len(y)):
            self.sample_placeholder.set(X[k])
            self.label_placeholder.set(y[k])
            self.forward()
            accuracy += self.accy_placeholder.value
            objective += self.loss_placeholder.value
        accuracy /= len(y)
        objective /= len(y)
        return accuracy, objective

    def fit(self, X, y, alpha, t):
        """
            Uses the cross entropy loss. The stochastic
            gradient descent goes through the examples in order, so
            that the output is deterministic and can be verified.
        :param X: an (m, n)-shaped numpy input matrix
        :param y: an (m,1)-shaped numpy output
        :param alpha: the learning rate
        :param t: the number of iterations
        :return:
        """
        # Create sample and input placeholder. Build the computation graph.
        # Add the operations corresponding to the loss to the computation graph.
        self.pred_placeholder = self.build_computational_graph()
        self.loss_placeholder = self.cross_entropy_loss()
        self.accy_placeholder = self.nn_binary_op(Accuracy, self.pred_placeholder, self.label_placeholder)

        train_loss = []
        train_acc = []
        since = time.time()
        for epoch in range(t):
            for i in tqdm(range(X.shape[0])):
                for p in self.params.values():
                    p.grad = DT(0)
                self.sample_placeholder.set(X[i])
                self.label_placeholder.set(y[i])
                self.forward()
                self.backward(self.loss_placeholder)
                self.sgd_update_parameter(alpha)
                
            # Evaluate on train set
            avg_acc, avg_loss = self.eval(X, y)
            print("Epoch %d: train loss = %.4f, accy = %.4f, [%.3f secs]" % (epoch, avg_loss, avg_acc, time.time()-since))
            train_loss.append(avg_loss)
            train_acc.append(avg_acc)
            since = time.time()
    
    def fit_plot(self, X, y, alpha, t):
        """
            Uses the cross entropy loss. The stochastic
            gradient descent goes through the examples in order, so
            that the output is deterministic and can be verified.
        :param X: an (m, n)-shaped numpy input matrix
        :param y: an (m,1)-shaped numpy output
        :param alpha: the learning rate
        :param t: the number of iterations
        :return:
        """
        # Create sample and input placeholder.  Build the computation graph.
        # Add the operations corresponding to the loss to the computation graph.
        self.pred_placeholder = self.build_computational_graph()
        self.loss_placeholder = self.cross_entropy_loss()
        self.accy_placeholder = self.nn_binary_op(Accuracy, self.pred_placeholder, self.label_placeholder)

        train_loss = []
        train_acc = []
        since = time.time()
        for epoch in range(t):
            for i in tqdm(range(X.shape[0])):
                for p in self.params.values():
                    p.grad = DT(0)
                self.sample_placeholder.set(X[i])
                self.label_placeholder.set(y[i])
                self.forward()
                self.backward(self.loss_placeholder)
                self.sgd_update_parameter(alpha)
                
            # Evaluate on train set
            avg_acc, avg_loss = self.eval(X, y)
            print("Epoch %d: train loss = %.4f, accy = %.4f, [%.3f secs]" % (epoch, avg_loss, avg_acc, time.time()-since))
            train_loss.append(avg_loss)
            train_acc.append(avg_acc)
            since = time.time()
        plt.figure()
        plt.plot([i+1 for i in range(t)], train_acc)
        plt.title("Training Accuracy vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Training Accuracy")
        plt.show()

        plt.figure()
        plt.plot([i+1 for i in range(t)], train_loss)
        plt.title("Training Loss vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Training Loss")
        plt.show()

    def forward(self):
        for c in self.components:
            c.forward()

    def backward(self, loss):
        for c in self.components:
            if c.grad is not None:
                c.grad = DT(0)
        loss.grad = np.ones_like(loss.value)
        for c in self.components[::-1]:
            c.backward()

    # Optimization functions
    def sgd_update_parameter(self, lr):
        # Update the parameter values in self.params
        for _, param in self.params.items():
            if param.grad is not None:
                param.value -= lr * param.grad


def test_set_and_get_weights():
    """Test NN.set_weights() and NN.get_weights()."""
    nodes_array = [4, 5, 5, 3]
    nn = NN(nodes_array, activation="sigmoid")
    weights = []
    for i in range(nn.layer_number):
        w = np.random.random((nodes_array[i], nodes_array[i+1])).astype(DT)
        b = np.random.random((nodes_array[i+1],)).astype(DT)
        weights.append((w, b))

    nn.set_weights(weights)
    nn_weights = nn.get_weights()

    for i in range(nn.layer_number):
        weight, bias = weights[i]
        nn_weight, nn_bias = nn_weights[i]
        if not np.array_equal(weight, nn_weight):
            raise AssertionError(f"The weight on layer {i} is not consistent.\n Set as {weight}, returned as {nn_weight}")
        if not np.array_equal(bias, nn_bias):
            raise AssertionError(f"The bias on layer {i} is not consistent.\n Set as {bias}, returned as {nn_bias}")
    print("Passed the test for set_weights and get_weights.")


def main():
    test_set_and_get_weights()


if __name__ == "__main__":
    main()
