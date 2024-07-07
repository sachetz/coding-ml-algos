import time
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt

DATA_TYPE = np.float32
EPSILON = 1e-12


def calculate_fan_in_and_fan_out(shape):
    """

    :param shape: Tuple of shape, e.g. (120,84) for the weight in a FC layer or (5,5,3,6) for the filter in a conv layer
    :return: fan_in, fan_out, representing the number of input parameter and output parameter
    """
    if len(shape)<2:
        raise ValueError("Unable to calculate fan_in and fan_out with dimension less than 2")
    elif len(shape)==2:  # Weight of a FC Layer
        fan_in, fan_out = shape[0], shape[1]
    elif len(shape)==4:  # filter of a convolutional layer
        # Calculate fan_in and fan_out (i.e. the number of input/output parameters
        # for the filter of a convolutional layer)
        num_in, num_out = shape[2], shape[3]
        kernel_size = shape[0] * shape[1]
        fan_in = kernel_size * num_in
        fan_out = kernel_size * num_out
    else:
        raise ValueError(f"Shape {shape} not supported in calculate_fan_in_and_fan_out")
    return fan_in, fan_out


def xavier(shape, seed=None):
    n_in, n_out = calculate_fan_in_and_fan_out(shape)
    if seed is not None:
        # Set seed to fixed number (e.g. layer idx) for predictable results
        np.random.seed(seed)
    # Initialize uniformly at random from [-sqrt(6/(n_in+n_out)), sqrt(6/(n_in+n_out))]
    m = np.sqrt(6 / (n_in + n_out))
    weight = np.random.uniform(-1 * m, m, size=shape)
    return weight

# InputValue: These are input values. They are leaves in the computational graph.
# Do not need to compute the gradient wrt them.
class InputValue:
    def __init__(self, value=None):
        self.value = DATA_TYPE(value).copy()
        self.grad = None

    def set(self, value):
        self.value = DATA_TYPE(value).copy()


# Parameters: Class for weight and biases, the trainable parameters whose values need to be updated
class Param:
    def __init__(self, value):
        self.value = DATA_TYPE(value).copy()
        self.grad = DATA_TYPE(0)


'''
  Class name: Add
  Class usage: add two matrices a, b with broadcasting supported by numpy "+" operation.
  Class function:
      forward: calculates a + b with possible broadcasting
      backward: calculates derivative w.r.t to a and b
'''   
class Add:  # Add with broadcasting
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.grad = None if a.grad is None and b.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = self.a.value + self.b.value

    def backward(self):
        if self.a.grad is not None:
            self.a.grad = self.a.grad + self.grad

        if self.b.grad is not None:
            self.b.grad = self.b.grad + np.sum(self.grad.reshape([-1, len(self.b.value)]), axis=0)            


'''
Class Name: Mul
Class Usage: element-wise multiplication with two matrix 
Class Functions:
    forward: computes the result a*b
    backward: computes the derivative w.r.t a and b
'''
class Mul:  # Multiply with broadcasting
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.grad = None if a.grad is None and b.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = self.a.value * self.b.value

    def backward(self):
        if self.a.grad is not None:
            self.a.grad = self.a.grad + self.grad * self.b.value

        if self.b.grad is not None:
            self.b.grad = self.b.grad + self.grad * self.a.value    
            
            
'''
Class Name: VDot
Class Usage: matrix multiplication where a is a vector and b is a matrix
    b is expected to be a parameter and there is a convention that parameters come last. 
    Typical usage is a is a feature vector with shape (f_dim, ), b a parameter with shape (f_dim, f_dim2).
Class Functions:
     forward: computes the vector matrix multplication result
     backward: computes the derivative w.r.t a and b, where derivative of a and b are both matrices 
'''
class VDot:  # Matrix multiply (fully-connected layer)
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.grad = None if a.grad is None and b.grad is None else DATA_TYPE(0)
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
Class Usage: computes the elementwise sigmoid activation. Input is vector or matrix. 
    In case of vector, [a_{0}, a_{1}, ..., a_{n}], output is vector [b_{0}, b_{1}, ..., b_{n}] where b_{i} = 1/(1 + exp(-a_{i}))
Class Functions:
    forward: computes activation b_{i} for all i.
    backward: computes the derivative w.r.t input vector/matrix a  
'''
class Sigmoid:
    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = 1 / (1 + np.exp(-1 * self.a.value))

    def backward(self):
        if self.a.grad is not None:
            self.a.grad += self.grad * (np.exp(-1 * self.a.value) / (1 + np.exp(-1 * self.a.value)) ** 2)
        


'''
Class Name: RELU
Class Usage: computes the elementwise RELU activation. Input is vector or matrix. In case of vector, 
    [a_{0}, a_{1}, ..., a_{n}], output is vector [b_{0}, b_{1}, ..., b_{n}] where b_{i} = max(0, a_{i})
Class Functions:
    forward: computes activation b_{i} for all i.
    backward: computes the derivative w.r.t input vector/matrix a  
'''
class RELU:
    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = self.value = np.maximum(0, self.a.value)

    def backward(self):
        if self.a.grad is not None:
            self.a.grad += (self.a.value > 0).astype(int) * self.grad


'''
Class Name: SoftMax
Class Usage: computes the softmax activation for each element in the matrix, normalization by each all elements 
    in each batch (row). Specifically, input is matrix [a_{00}, a_{01}, ..., a_{0n}, ..., a_{b0}, a_{b1}, ..., a_{bn}], 
    output is a matrix [p_{00}, p_{01}, ..., p_{0n},...,p_{b0},,,p_{bn} ] where p_{bi} = exp(a_{bi})/(exp(a_{b0}) + ... + exp(a_{bn}))
Class Functions:
    forward: computes probability p_{bi} for all b, i.
    backward: computes the derivative w.r.t input matrix a 
'''
class SoftMax:
    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = np.exp(self.a.value) / np.sum(np.exp(self.a.value))

    def backward(self):
        if self.a.grad is not None:
            self.a.grad += (self.grad * self.value) - (self.value * np.dot(self.grad, self.value))


'''
Class Name: Log
Class Usage: computes the elementwise log(a) given a.
Class Functions:
    forward: computes log(a)
    backward: computes the derivative w.r.t input vector a
'''
class Log: # Elementwise Log
    def __init__(self, a):
        self.a = a
        self.grad = None if a.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = np.log(self.a.value)

    def backward(self):
        if self.a.grad is not None:
            self.a.grad += self.grad * (1 / self.a.value)


'''
Class Name: Aref
Class Usage: gets a specific entry in a matrix. a is the matrix with shape (batch_size, N) and idx is vector containing 
    the entry index and a is differentiable.
Class Functions:
    forward: computes a[batch_size, idx]
    backward: computes the derivative w.r.t input matrix a
'''
class Aref:
    def __init__(self, a, idx):
        self.a = a
        self.idx = idx
        self.grad = None if a.grad is None else DATA_TYPE(0)

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
Class Usage: checks whether the predicted label is correct or not. a is the probability vector where each probability is 
            for each class. idx is ground truth label.
Class Functions:
    forward: finds the label that has maximum probability and compare it with the ground truth label.
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


'''
Class Name: Conv
Class Usage: convolutional layer that performs elementwise multiplication within the rolling window 
            and output the sum of products to the corresponding cell.
Class Functions:
    forward: Calculates the output of convolutional layer
    backward: Calculates the derivative w.r.t. the input tensor and kernel
'''
class Conv:

    def __init__(self, input_tensor, kernel, stride=1, padding=0):
        """
        :param input_tensor: input tensor of size (height, width, in_channels)
        :param kernel: convolving kernel of size (kernel_size, kernel_size, in_channels, out_channels),
                        only square kernels of size (kernel_size, kernel_size) are supported
        :param stride: stride of convolution. Default: 1
        :param padding: zero-padding added to both sides of the input. Default: 0
        """
        self.kernel = kernel
        self.input_tensor = input_tensor
        self.padding = padding
        self.stride = stride
        self.grad = None if kernel.grad is None and input_tensor.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        """
        Calculates self.value of size (output_height, output_width, out_channels)
        Supports stride>1 and padding>0.
        """
        height, width, in_channels = self.input_tensor.value.shape
        kernel_size = self.kernel.value.shape[0]
        output_channels = self.kernel.value.shape[3]
        padded_input = np.zeros((height + 2 * self.padding, width + 2 * self.padding, in_channels))
        padded_input[self.padding:(self.padding + height), self.padding:(self.padding + width), :] = self.input_tensor.value
        output_height = int((height + 2 * self.padding - kernel_size) / self.stride) + 1
        output_width = int((width + 2 * self.padding - kernel_size) / self.stride) + 1
        self.value = np.zeros((output_height, output_width, output_channels))
        for i in range(output_height):
            for j in range(output_width):
                for c in range(output_channels):
                    # todo
                    i0 = i * self.stride
                    j0 = j * self.stride
                    self.value[i, j, c] = np.sum(padded_input[i0:(i0 + kernel_size), j0:(j0 + kernel_size), :] * self.kernel.value[:, :, :, c])

    def backward(self):
        """
        Calculates gradient of kernel.grad and input_tensor
        Supports stride>1 and padding>0.
        """
        height, width, in_channels = self.input_tensor.value.shape
        kernel_size = self.kernel.value.shape[0]
        output_channels = self.kernel.value.shape[3]
        kernel_grad = np.zeros(self.kernel.value.shape)
        padded_input = np.zeros((height + 2 * self.padding, width + 2 * self.padding, in_channels))
        padded_input[self.padding:(self.padding + height), self.padding:(self.padding + width), :] = self.input_tensor.value
        input_grad = np.zeros(padded_input.shape)
        for i in range(self.value.shape[0]):
            for j in range(self.value.shape[1]):
                for c in range(output_channels):
                    i0 = i * self.stride
                    j0 = j * self.stride
                    # todo
                    kernel_grad[:, :, :, c] += padded_input[i0:(i0 + kernel_size), j0:(j0 + kernel_size), :] * self.grad[i, j, c]
                    input_grad[i0:(i0 + kernel_size), j0:(j0 + kernel_size), :] += self.kernel.value[:, :, :, c] * self.grad[i, j, c]
        if self.kernel.grad is not None:
            self.kernel.grad = self.kernel.grad + kernel_grad
        if self.input_tensor.grad is not None:
            self.input_tensor.grad = self.input_tensor.grad + input_grad[self.padding:(self.padding + height),
                                                              self.padding:(self.padding + width), :]


'''
Class Name: MaxPool
Class Usage: Applies a max pooling over an input signal composed of several input planes.
Class Functions:
    forward: Calculates the output of convolutional layer
    backward: Calculates the derivative w.r.t. the input tensor and kernel
'''
class MaxPool:
    def __init__(self, input_tensor, kernel_size=2, stride=None):
        """
        :param input_tensor: input tensor of size (height, width, in_channels)
        :param kernel_size: the size of the window to take a max over. Default: 2
        :param stride: the stride of the window. Default value is kernel_size
        """
        self.input_tensor = input_tensor
        self.kernel_size = kernel_size
        if stride is None:
            self.stride = kernel_size
        else:
            self.stride = stride
        self.grad = None if input_tensor.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        """
        Calculates self.value of size (int(height / self.stride), int(width / self.stride), in_channels)
        Supports stride!=kernel_size.
        """
        # This implementation makes the simplifying assumption that stride is at 
        # least as large as the kernel size in maxpool and that no padding is
        # needed. This simplifies the computation of output height/width.
        height, width, in_channels = self.input_tensor.value.shape
        output_height = int(height / self.stride)
        output_width = int(width / self.stride)
        self.value = np.zeros((output_height, output_width, in_channels))
        for c in range(in_channels):
            for i in range(output_height):
                for j in range(output_width):
                    # todo
                    i0 = i * self.stride
                    j0 = j * self.stride
                    self.value[i, j, c] = np.max(self.input_tensor.value[i0:(i0 + self.kernel_size), j0:(j0 + self.kernel_size), c])

    def backward(self):
        """
        Calculates the gradient for input_tensor
        Supports stride!=kernel_size.
        """
        height, width, in_channels = self.input_tensor.value.shape
        input_grad = np.zeros(self.input_tensor.value.shape)
        
        output_height = int(height / self.stride)
        output_width = int(width / self.stride)
        for c in range(in_channels):
            for i in range(output_height):
                for j in range(output_width):
                    # todo
                    i0 = i * self.stride
                    j0 = j * self.stride
                    vals = self.input_tensor.value[i0:(i0 + self.kernel_size), j0:(j0 + self.kernel_size), c]
                    mask = (self.value[i, j, c] == vals)
                    input_grad[i0:(i0 + self.kernel_size), j0:(j0 + self.kernel_size), c] += mask * self.grad[i, j, c]
        self.input_tensor.grad = self.input_tensor.grad + input_grad


'''
  Class name: Flatten
  Class usage: Flatten the input tensor to a 1d vector.
  Class function:
      forward: Flatten the input tensor to a 1d vector.
      backward: Calculates derivative w.r.t to input_tensor, 
                which is simply reshaping the output gradient to input_tensor's original shape
'''
class Flatten:
    def __init__(self, input_tensor):
        self.input_tensor = input_tensor
        self.grad = None if input_tensor.grad is None else DATA_TYPE(0)
        self.value = None

    def forward(self):
        self.value = self.input_tensor.value.flatten()

    def backward(self):
        if self.input_tensor.grad is not None:
            self.input_tensor.grad += self.grad.reshape(self.input_tensor.value.shape)


class CNN:
    def __init__(self, num_labels=10):
        self.num_labels = num_labels
        # dictionary of trainable parameters
        self.params = {}
        # list of computational graph
        self.components = []
        self.sample_placeholder = InputValue()
        self.label_placeholder = InputValue()
        self.pred_placeholder = None
        self.loss_placeholder = None
        self.accy_placeholder = None

    # helper function for creating a unary operation object and adding it to the computational graph
    def nn_unary_op(self, op, a):
        unary_op = op(a)
        print(f"Append <{unary_op.__class__.__name__}> to the computational graph")
        self.components.append(unary_op)
        return unary_op

    # helper function for creating a binary operation object and adding it to the computational graph
    def nn_binary_op(self, op, a, b):
        binary_op = op(a, b)
        print(f"Append <{binary_op.__class__.__name__}> to the computational graph")
        self.components.append(binary_op)
        return binary_op

    def conv_op(self, input_tensor, kernel, stride=1, padding=0):
        conv = Conv(input_tensor, kernel, stride=stride, padding=padding)
        print(f"Append <{conv.__class__.__name__}> to the computational graph")
        self.components.append(conv)
        return conv

    def maxpool_op(self, input_tensor, kernel_size=2, stride=None):
        maxpool = MaxPool(input_tensor, kernel_size=kernel_size, stride=stride)
        print(f"Append <{maxpool.__class__.__name__}> to the computational graph")
        self.components.append(maxpool)
        return maxpool

    def set_params_by_dict(self, param_dict: dict):
        """
        :param param_dict: a dict of parameters with parameter names as keys and numpy arrays as values
        """
        # reset params to an empty dict before setting new values
        self.params = {}
        # add Param objects to the dictionary of trainable paramters with names and values
        for name, value in param_dict.items():
            self.params[name] = Param(value)

    def get_param_dict(self):
        """
        :return: param_dict: a dict of parameters with parameter names as keys and numpy arrays as values
        """
        # Extract trainable parameter values from the dict of Params
        param_dict = {
            "conv1_kernel": self.params["conv1_kernel"],
            "conv1_bias": self.params["conv1_bias"],
            "conv2_kernel": self.params["conv2_kernel"],
            "conv2_bias": self.params["conv2_bias"],
            "fc1_weight": self.params["fc1_weight"],
            "fc1_bias": self.params["fc1_bias"],
            "fc2_weight": self.params["fc2_weight"],
            "fc2_bias": self.params["fc2_bias"],
            "fc3_weight": self.params["fc3_weight"],
            "fc3_bias": self.params["fc3_bias"],
        }
        return param_dict

    def init_params_with_xavier(self):
        # Initialize param_dict such that each key is mapped to a numpy array of the corresponding size
        param_dict = {
            "conv1_kernel": xavier(shape=(5, 5, 3, 6)),
            "conv1_bias": np.zeros(6),
            "conv2_kernel": xavier((5, 5, 6, 16)),
            "conv2_bias": np.zeros(16),
            "fc1_weight": xavier((400, 120)),
            "fc1_bias": np.zeros(120),
            "fc2_weight": xavier((120, 84)),
            "fc2_bias": np.zeros(84),
            "fc3_weight": xavier((84, self.num_labels)),
            "fc3_bias": np.zeros(self.num_labels),
        }
        self.set_params_by_dict(param_dict)

    def build_computational_graph(self):
        # Reset computational graph to empty list
        self.components = []

        input_tensor = self.sample_placeholder
        # Build the computational graph with the following architecture in order:
        #  0. input_tensor of size (32, 32, 3) which matches to (height, width, channel)
        #  1. Conv: kernel size: (5,5), input channel: 3, output channel: 6, output shape: (28, 28, 6)
        conv1_kernel = self.params["conv1_kernel"]
        conv1_without_bias = self.conv_op(input_tensor, conv1_kernel)
        conv1_bias = self.params["conv1_bias"]
        conv1 = self.nn_binary_op(Add, conv1_without_bias, conv1_bias)
        #  2. RELU: activation function on conv1's output, output shape: (28, 28, 6)
        relu1 = self.nn_unary_op(RELU, conv1)
        #  3. MaxPool: kernel size: (2,2), output shape: (14, 14, 6)
        maxpool1 = self.maxpool_op(relu1)
        #  4. Conv: kernel size: (5,5), input channel: 6, output channel: 16, output shape: (10, 10, 16)
        conv2_kernel = self.params["conv2_kernel"]
        conv2_without_bias = self.conv_op(maxpool1, conv2_kernel)
        conv2_bias = self.params["conv2_bias"]
        conv2 = self.nn_binary_op(Add, conv2_without_bias, conv2_bias)
        #  5. RELU: activation function on conv1's output, output shape: (10, 10, 16)
        relu2 = self.nn_unary_op(RELU, conv2)
        #  6. MaxPool: kernel size: (2,2), output shape: (5, 5, 16)
        maxpool2 = self.maxpool_op(relu2)
        #  7. Flatten: output shape: (400,)
        flat = self.nn_unary_op(Flatten, maxpool2)
        #  8. (Fully Connected Layer): input shape (400,), output size: (120, )
        fc1_kernel = self.params["fc1_weight"]
        fc1_without_bias = self.nn_binary_op(VDot, flat, fc1_kernel)
        fc1_bias = self.params["fc1_bias"]
        fc1 = self.nn_binary_op(Add, fc1_without_bias, fc1_bias)
        #  9. RELU: activation function on previous output, output shape: (120, )
        relu3 = self.nn_unary_op(RELU, fc1)
        #  10. (Fully Connected Layer): input size: 120, output size: (84, )
        fc2_kernel = self.params["fc2_weight"]
        fc2_without_bias = self.nn_binary_op(VDot, relu3, fc2_kernel)
        fc2_bias = self.params["fc2_bias"]
        fc2 = self.nn_binary_op(Add, fc2_without_bias, fc2_bias)
        #  11. RELU: activation function on conv1's output, output shape: (84, )
        relu4 = self.nn_unary_op(RELU, fc2)
        #  12. (Fully Connected Layer): input size: 84, output size: (self.num_labels, )
        fc3_kernel = self.params["fc3_weight"]
        fc3_without_bias = self.nn_binary_op(VDot, relu4, fc3_kernel)
        fc3_bias = self.params["fc3_bias"]
        fc3 = self.nn_binary_op(Add, fc3_without_bias, fc3_bias)
        #  13. SoftMax: activation function on conv1's output, output shape: (self.num_labels, )
        pred = self.nn_unary_op(SoftMax, fc3)
        return pred

    def cross_entropy_loss(self):
        # Construct cross entropy loss using self.pred_placeholder and self.label_placeholder
        #  as well as self.nn_binary_op and self.nn_unary_op
        label_prob = self.nn_binary_op(Aref, self.pred_placeholder, self.label_placeholder)
        log_prob = self.nn_unary_op(Log, label_prob)
        loss = self.nn_binary_op(Mul, log_prob, InputValue(-1))
        return loss

    def eval(self, X, y):
        if len(self.components) == 0:
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
            Uses the cross entropy loss.  The stochastic
            gradient descent should go through the examples in order, so
            that the output is deterministic and can be verified.
        :param X: an (m, n)-shaped numpy input matrix
        :param y: an (m,1)-shaped numpy output
        :param alpha: the learning rate
        :param t: the number of iterations
        :return:
        """
        # create sample and input placeholder
        self.pred_placeholder = self.build_computational_graph()
        self.loss_placeholder = self.cross_entropy_loss()
        self.accy_placeholder = self.nn_binary_op(Accuracy, self.pred_placeholder, self.label_placeholder)

        train_loss = []
        train_acc = []
        since = time.time()
        for epoch in range(t):
            for i in tqdm(range(X.shape[0])):
                # tqdm adds a progress bar
                for p in self.params.values():
                    p.grad = DATA_TYPE(0)
                for c in self.components:
                    if c.grad is not None:
                        c.grad = DATA_TYPE(0)
                self.sample_placeholder.set(X[i])
                self.label_placeholder.set(y[i])
                # Training process
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
            Uses the cross entropy loss.  The stochastic
            gradient descent should go through the examples in order, so
            that the output is deterministic and can be verified.
        :param X: an (m, n)-shaped numpy input matrix  # todo change doc
        :param y: an (m,1)-shaped numpy output
        :param alpha: the learning rate
        :param t: the number of iterations
        :return:
        """
        # create sample and input placeholder
        self.pred_placeholder = self.build_computational_graph()
        self.loss_placeholder = self.cross_entropy_loss()
        self.accy_placeholder = self.nn_binary_op(Accuracy, self.pred_placeholder, self.label_placeholder)

        train_loss = []
        train_acc = []
        since = time.time()
        for epoch in range(t):
            for i in tqdm(range(X.shape[0])):
                # tqdm adds a progress bar
                for p in self.params.values():
                    p.grad = DATA_TYPE(0)
                for c in self.components:
                    if c.grad is not None:
                        c.grad = DATA_TYPE(0)
                self.sample_placeholder.set(X[i])
                self.label_placeholder.set(y[i])
                # Training process
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
        loss.grad = np.ones_like(loss.value)
        for c in self.components[::-1]:
            c.backward()

    def sgd_update_parameter(self, lr):
        # Update the parameter values in self.params
        for p in self.params.values():
            p.value = p.value - lr * p.grad


def main():
    pass


if __name__ == "__main__":
    main()
