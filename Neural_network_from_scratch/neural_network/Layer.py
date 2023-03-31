import numpy as np


class Layer:
    """Each layer class object encapsulates the logic behind nodes in a layer in addition to containing the input,
    output, weights of the nodes with the biases, activation function used and the activation output when using said
    function on the output of the nodes"""
    def __init__(self, ip=None, op=None, activation_function=None):
        self._input = None
        self._weights = np.random.rand(op, ip) - 0.5
        self._biases = np.random.rand(op, 1) - 0.5
        self._output = None
        self._activation_function = activation_function
        self._activation_output = None

    @property
    def input(self):
        return self._input

    @property
    def weights(self):
        return self._weights

    @property
    def biases(self):
        return self._biases

    @weights.setter
    def weights(self, w):
        self._weights = w

    @biases.setter
    def biases(self, b):
        self._biases = b

    @property
    def output(self):
        return self._output

    @property
    def activation_function(self):
        return self._activation_function

    @property
    def activation_output(self):
        return self._activation_output

    def reset(self):
        """Method to reset the layer's weights and input/output to the initial randomized state following any number of
            forward passes"""
        self._input = None
        self.weights = np.random.rand(self.weights.shape[0], self.weights.shape[1]) - 0.5
        self.biases = np.random.rand(self.biases.size, 1) - 0.5
        self._output = None
        self._activation_function = self.activation_function
        self._activation_output = None

    def forward_pass(self, ip):
        """Method that implements the forward pass to compute the output (activation_output) of the network for a
            given input ip"""
        self._input = ip
        self._output = np.dot(self._weights, self._input) + self._biases
        self._activation_output = self._activation_function(self._output)

        return self._activation_output

    def back_prop(self, weights, de_y_o, dy_a_o, n):
        """Method that implements the backward propagation to compute the partial derivatives of dw, db, dE/dy, dy/da the
            last two of which, in addition to the current layer's weights, are used to compute the partial derivatives and
            continue the backpropagation algorithm to the layer before the current one"""
        # m : number of samples in the training set
        # h_curr : number of nodes in the current hidden layer
        # h_prev : number of nodes in the previous hidden layer
        # (o, m)
        dE_y_h_2 = np.dot(weights.T, (de_y_o * dy_a_o))
        # (1, m)
        dy_a_h_2 = self._activation_function(self._output, True)
        # (h_prev, m)
        da_w_h_2 = self.input

        # (h_curr, h_prev) = (h_curr, m) . (m, h_prev)
        dw = np.dot((dE_y_h_2 * dy_a_h_2), da_w_h_2.T) / n
        # (h_curr,) = (h_curr, m) . (m,)
        db = np.sum(np.dot(dE_y_h_2, dy_a_h_2.T), axis=1) / n
        # (h_curr, 1)
        db = db[:, np.newaxis]

        return dw, db, dE_y_h_2, dy_a_h_2, self.weights
