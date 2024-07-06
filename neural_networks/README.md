# Neural Networks

## Overview
- The architecture of the neural network is a vanilla version of that of Pytorch/Tensorflow consisting of nodes of *Operation Classes*. Each *Operation Class* has a forward method (to calculate the operation result stored in self.value) and a backward method (to calculate the gradient of the operation w.r.t. the final loss function).

- Forward and backward methods are implemented for *Operation Classes* <code>Mul</code>, <code>VDot</code>, <code>Sigmoid</code>, <code>Relu</code>, <code>Softmax</code>, and <code>Log</code>, <code>Add</code>, <code>Aref</code>, and <code>Accuracy</code>.

- <code>self.params</code> is the list to store the trainable parameters (objects of Class <code>Param</code>). In <code>set_weights(weights)</code>, the provided weights and biases are converted to <code>Param</code> objects and stored in <code>self.params</code>.

- </code>self.components</code> is a list that mimics the function of the computational graph. Helper functions <code>nn_unary_op(op, x)</code> and <code>nn_binary_op(op, x, y)</code> facilitate creating an operation and adding it to the computational graph.

- <code>self.forward()</code> evaluates each operation object in the ordered list <code>self.components</code> in order

- <code>self.backward()</code> computes the derivative of each operation object in <code>self.components</code> in reverse order.

## MNIST dataset of handwritten digits

The implementation above is used to build an artificial neural network for the MNIST dataset of handwritten digits.

The neural network performs well on the MNIST dataset with 96.46% accuracy on the test set and 0.1145 loss value, when used with the sigmoid activation function and 1 hidden layer.

Adding another hidden layer does not have a significant impact on the model performance, with 96.4% test accuracy and 0.1175 loss.

Using the ReLU activation function provides a performance boost for the model, with 97.31% accuracy on the test set and 0.1050 loss.
