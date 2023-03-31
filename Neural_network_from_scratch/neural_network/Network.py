import neural_network.functions as functions
import numpy as np
import math


class Network:
    """Each network class object that contains the layers in the neural network model and the implementation of the
    forward pass, the backpropagation algorithm, the class prediction, the loss function used, the duration of training
    the model on a set of inputs in epochs and the implementation of two techniques to estimate the generalization
    capability which are the hold-out technique and the k-fold cross validation"""
    def __init__(self, loss_function=None,
                 epochs=1000):
        self._layers = []
        self._loss_function = loss_function
        self._epochs = epochs

    @property
    def layers(self):
        return self._layers

    @property
    def loss_function(self):
        return self._loss_function

    @property
    def epochs(self):
        return self._epochs

    def add_layer(self, layer):
        """
        Method that performs backpropagation
            x : attributes
            t : class_labels
            learning_rate : the learning rate used in backpropagation
            epochs : the number of epochs each model will be trained for
        """
        self._layers.append(layer)

    def reset_layers(self):
        """Method that resets all layers in the network"""
        for layer in self.layers:
            layer.reset()

    def forward_pass(self, x):
        """
        Method that computes the output of the network
             x : attributes
        Returns an array containing the predicted class for each sample in the input
        """
        ip = x
        op = None
        for e in self.layers:
            op = e.forward_pass(ip)
            ip = op

        return op

    def backpropagation(self, x, t, learning_rate=0.01, epochs=1000):
        """
        Method that performs backpropagation
            x : attributes
            t : class_labels
            learning_rate : the learning rate used in backpropagation
            epochs : the number of epochs each model will be trained for
        """
        N = t.size

        for i in range(epochs):
            # Forward propagation
            network_op = self.forward_pass(x)

            # Back propagation
            # Output layer
            layer = self._layers[-1]

            # m : number of samples in the training set
            # o : number of nodes in the current output layer
            # h : number of nodes in the previous hidden layer
            # (o, m)
            dE_y_o = self._loss_function(network_op, t, True)
            # (1, m)
            dy_a_o = layer.activation_function(layer.output, True)
            # (h, m)
            da_w_o = layer.input

            # (o, h) = (o, m) . (m, h)
            dw = np.dot((dE_y_o * dy_a_o), da_w_o.T) / N
            # (o,) = (o, m) . (m,)
            db = np.sum(np.dot(dE_y_o, dy_a_o.T), axis=1) / N
            # (o, 1)
            db = db[:, np.newaxis]

            bp = [dw, db, dE_y_o, dy_a_o, layer.weights]

            layer.weights -= bp[0] * learning_rate
            layer.biases -= bp[1] * learning_rate

            # Hidden layer(s)
            for j in reversed(range(len(self._layers)-1)):
                layer = self._layers[j]
                bp = layer.back_prop(bp[4], bp[2], bp[3], N)

                layer.weights -= bp[0] * learning_rate
                layer.biases -= bp[1] * learning_rate

    def class_predict(self, x):
        """
        Method that performs class prediction for all given samples in x
            x : attributes
            learning_rate : the learning rate used in backpropagation
            epochs : the number of epochs each model will be trained for
        Returns an array containing the predicted class for each sample in the input
        """
        y = self.forward_pass(x).T
        """2-class prediction with one node in the output layer"""
        if y.shape[1] == 1:
            return np.array([1 if i > 0.5 else 0 for i in y])
        """multi-class predictions"""
        return np.argmax(y, axis=1)

    def fit(self, x, y, learning_rate, epochs):
        """
        Method that encapsulates the training of one model, and outputs the accuracy and error for a given training set
            x : attributes
            y : class labels
            learning_rate : the learning rate used in backpropagation
            epochs : the number of epochs each model will be trained for
        Returns the model's accuracy and error after the training
        """
        accuracy = functions.accuracy_of_prediction(self.class_predict(x), y)
        error = self._loss_function(self.forward_pass(x), y)
        self.backpropagation(x, y, learning_rate, epochs)
        print("initial accuracy = %.2f and initial error = %.2f" % (accuracy, error))
        accuracy = functions.accuracy_of_prediction(self.class_predict(x), y)
        error = self._loss_function(self.forward_pass(x), y)
        print("updated accuracy = %.2f and updated error = %.2f" % (accuracy, error))

        return accuracy, error

    def hold_out(self, x, y, learning_rate, epochs):
        """
        Method that implements the hold-out technique
            x : attributes
            y : class labels
            learning_rate : the learning rate used in backpropagation
            epochs : the number of epochs each model will be trained for
        """
        N = y.shape[0]
        N_split = math.ceil(N * 0.7)

        X_train = np.array(x.T[:, 0:N_split])
        Y_train = np.array(y.T[0:N_split])

        X_test = np.array(x.T[:, N_split:N])
        Y_test = np.array(y.T[N_split:N])

        self.fit(X_train, Y_train, learning_rate, epochs)
        y_test = self.class_predict(X_test)
        print(functions.accuracy_of_prediction(y_test, Y_test))

    def cross_validation(self, x, y, k, learning_rate, epochs):
        """
        Method that implements the k-fold cross validation technique
            x : attributes
            y : class labels
            k = number of folds
            learning_rate : the learning rate used in backpropagation
            epochs : the number of epochs each model will be trained for
        """
        N = y.shape[0]
        Error = 0

        print("initiating k-fold cross validation with k=%d" % k)
        k_folds = [math.floor(i) for i in np.arange(0, N, N/k)]

        for i in range(len(k_folds)-1):
            self.reset_layers()
            print("testing folds : [%d, %d]" % (k_folds[i], k_folds[i + 1]))
            print("training folds : [%d, %d] and [%d, %d]" % (k_folds[0], k_folds[i], k_folds[i + 1], N))

            X_test = x.T[:, k_folds[i]:k_folds[i+1]]
            Y_test = y.T[k_folds[i]:k_folds[i+1]]

            X_train = np.concatenate((x.T[:, k_folds[0]:k_folds[i]], x.T[:, k_folds[i+1]:N]), axis=1)
            Y_train = np.concatenate((y.T[k_folds[0]:k_folds[i]], y.T[k_folds[i+1]:N]), axis=0)

            Error += self.fit(X_train, Y_train, learning_rate, epochs)[1]
            y_test = self.class_predict(X_test)
            print("accuracy on testing fold = %.2f \n" % functions.accuracy_of_prediction(y_test, Y_test))

        Error = Error/(k-1)
        print("Mean error of all models trained = %.2f" % Error)
