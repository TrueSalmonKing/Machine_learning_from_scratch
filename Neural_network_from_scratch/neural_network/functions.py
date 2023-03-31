import numpy as np


def tanh_activation(a, drv=False):
    """
    Method that implements the tanh activation function
        a : weighted sum of the layer input
        drv : in the case where we want the derivative of this function to use in the backpropagation function
        we can pass the argument drv=True
    """
    if drv:
        return 1 - np.square(np.tanh(a))
    return np.tanh(a)


def sigmoid_activation(a, drv=False):
    """
    Method that implements the sigmoid activation function
        a : weighted sum of the layer input
        drv : in the case where we want the derivative of this function to use in the backpropagation function (we can
         pass the argument drv=True
    """
    if drv:
        g = sigmoid_activation(a)
        return g * (1 - g)
    return 1 / (1 + np.exp(-a))


def cross_entropy(y, t, drv=False):
    """
    Method that implements the cross entropy loss function
        y : output of the network
        t : class labels
        drv : in the case where we want the derivative of this function to use in the backpropagation function (we can
        pass the argument drv=True
    """
    N = y.size
    if drv:
        return -t / y
    return -np.sum(t * np.log(y)) / N


def mean_square_error(y, t, drv=False):
    """
    Method that implements the mean square error function
        y : output of the network
        t : class labels
        drv : in the case where we want the derivative of this function to use in the backpropagation function (we can
        pass the argument drv=True
    """
    N = y.size
    if drv:
        return y - t
    return (1 / 2) * np.sum(np.square(y - t)) / N


def accuracy_of_prediction(y, t):
    """
    Method that returns the percentage of correct class predictions made by the model
        y : output of the network
        t : class labels
    """
    return 100 * (np.sum(y == t) / t.size)
