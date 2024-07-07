# CNN

## Overview

- The architecture is designed to classify 32x32 color images. The training data has 3000 images which fall into 4 classes. This architecture is adapted from the [CNN tutorial at PyTorch](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html). The specific layers are listed below. In these all convolutional layers use zero padding and a stride of 1; and all maxpooling layers use zero padding and a stride equal to the kernel size (which is always 2).

    1. A convolutional layer with kernel size (5, 5) and 6 channels.

    2. A maxpooling layer with kernel size (2, 2).

    3. A convolutional layer with kernel size (5, 5) and 16 channels.

    4. A maxpooling layer with kernel size (2, 2).

    5. A sequence of three fully connected layers of sizes 120, 84, and K, in order, in which K is the number of labels. (K is 4.)

- Further, each of the convolutional or fully connected layer is followed by a ReLU activation function layer, except for the last layer, which is followed by a softmax layer. n addition, a flatten layer, between the last max-pooling layer and the first fully connected layer, is needed to rearrange the 3-dimensional image tensor into a 1-dimensional vector.

- The CNN operation classes: In addition to the operation classes used with NN, CNN uses classes Conv, MaxPool, and Flatten.

- The following steps are performed:

    1. Instantiate an CNN object by specifying the number of classes to predict. (4)

    2. Call init_weights_with_xavier(), which uses Xavier initialization for the parameters. In CNNs this initialization uses the function xavier() to generate the random values and a helper function calculate_fan_in_and_fan_out() to calculates the number of input and output parameters.

    3. Train the network by calling fit(X, y, alpha, t), in which X is the training set and y contains the labels. alpha is the learning rate to use in SGD, and t is the number of epochs10 to train for.

## CIFAR10 dataset of colored images

The original dataset consists of 50,000 images for training and 10,000 for testing, evenly distributed over 10 classes. For simplicity, we train on a subset of 3000 images and test on 1000 images, which fall in 4 classes.

The graphs show an upward trend in the training accuracy, indicating an improvement in the model performance with a number of epochs. The training of the model across 10 epochs increase the accuracy from less than 55% to over 75% on the training dataset, and 65% on the test dataset, indicating that the CNN is effective in the classification task it is trained for.

## Data

https://drive.google.com/drive/folders/1xSa66GGYVbJlJXSS1XgISfpdP5Duk9WV?usp=sharing