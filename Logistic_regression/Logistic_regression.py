import pandas as pd
import matplotlib.pyplot as mp
import numpy as np

# Taken from https://www.kaggle.com/datasets/muratkokludataset/raisin-dataset

data = pd.read_excel("../data/Raisin_Dataset/Raisin_Dataset.xlsx", engine='openpyxl')


def mv_normalize(features):
    for i in range(8):
        f_mean = np.mean(features[:, i])
        f_range = np.amax(features[:, i]) - np.amin(features[:, i])
        if f_range:
            features[:, i] = (features[:, i] - f_mean) / f_range

    return features


def features_selection(features):
    # Initial value of the bias is set to 1
    # With the addition of a bias, we allow our prediction function to be more accurate as it no longer strictly goes
    # through the origin (0, 0)
    bias = np.ones(shape=(len(features), 1))
    return mv_normalize(np.append(bias, features.iloc[:, 0:7].values, axis=1))


# Prediction function for binary Logistic regression (linear function closest to predicting the values in the data set)
# predictions (X,1) = features (X,7) . weights (7,1)
def predict(features, weights):
    return sigmoid_activation(np.dot(features, weights))


# We map our predictions into probabilities (values between 0 and 1) using the sigmoid function:
# S(z) = 1 / (1 + e(-z)) with z(Xi) being the prediction function
# This function is similar to the one used in the multivariable linear regression
# Using a decision boundary, we can then, using the provided features, classify the object using the model
# as a "Kecimen" or "Besni" raisin:
# For example if our decision boundary was 0.5, and if the model returns 0.3, it is classified in the class which
# occupies the range of probabilities [0, 0.5]
def sigmoid_activation(z):
    return 1.0 / (1 + np.exp(-z))


# We will use the cost function cross entropy loss instead of mean square error, to optimize our weights for
# better predictions
# cross entropy loss => cel() = (Sum{1..n}(-Yi*(log(Si) - (1-Yi) * log(1-Si))/(n)
# with Si(Zi) = 1 / 1 + exp(-Zi) (sigmoid function) and Zi(Xi) = Sum{1..n}(Wi*Xi) + b
# predictions (414,1) = features (414,3) . weights (3,1)
def cross_entropy_loss(features, data_values, weights):
    predictions = predict(features, weights)
    n = len(data_values)

    cost = -np.dot(data_values, np.log(predictions))
    cost -= np.dot((1 - data_values), np.log(1 - predictions))
    cost = cost.sum() / n

    return cost


# Calculating the gradient descent of the sigmoid function with respect to the weights in order to minimize the error
# function (sample variance) by updating the weights:
# partial derivative d(cle()/d(Wi) = (1/n)*(Sum{1..n}((Si-y) * Xi)
# partial derivative d(cle()/d(b) = (1/n)*(Sum{1..n}((Si-y))
# with Si(Zi) = 1 / 1 + exp(-Zi) (sigmoid function) and Zi(Xi) = Sum{1..n}(Wi*Xi) + b
def update_weights(features, data_values, weights, learning_rate):
    predicted_values = predict(features, weights)
    n = len(data_values)

    # gradient (3,1) = features.T (3,414) . ( data_values - predicted_values ) (414,1)
    gradient = (np.dot(features.T, (predicted_values - data_values))) / n

    weights -= (learning_rate * gradient)

    return weights


# function that maps a class names to a value between [0, ..., 1/(n-1)] with n being the number of classes, in our
# classes, it is n = 2, thus the "binary" in binary logistic regression
def class_name_id(name):
    return 1 if name == 'Kecimen' else 0


# function that applies the function class_name_id onto the list of data values containing the list of both Kecimen and
# Besni raisins
def class_name_id_creation(data_values, class_name_id_func):
    class_name_id_func = np.vectorize(class_name_id_func)
    return class_name_id_func(data_values)


# function that maps the predictions onto the raisin class they "should" be mapped to, meaning it will allow us to
# visualize the cross entropy loss by viewing the predictions that are correct and that ones that aren't
def classify(predictions, data_values, class_1, class_2):
    i = 0
    for prediction in predictions:
        if data_values[i] == 'Kecimen':
            class_1.append(prediction)
        elif data_values[i] == 'Besni':
            class_2.append(prediction)
        i += 1

    return class_1, class_2


# Binary logistic regression implementation, where we use gradient descent to minimize the cross entropy loss
# In order to reach the closest possible line function that can predict values to the upmost accuracy
def binary_logistic_regression():
    train_rate = 0.009
    epoch = 30001
    cel_values = [0] * epoch
    kecimens = []
    besnis = []

    features = features_selection(data)
    data_values = class_name_id_creation(data.iloc[:, 7].values, class_name_id)
    n = len(features.T)
    weights = [0.0] * n
    cel = cross_entropy_loss(features, data_values, weights)
    kecimens, besnis = classify(predict(features, weights), data.iloc[:, 7].values, kecimens, besnis)

    mp.ion()
    fig = mp.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([0, max(len(kecimens), len(besnis))])
    ax.set_title("Predictions Comparison to the set Decision Boundary")
    mp.axhline(0.5, c='black')

    kecimens, besnis = classify(predict(features, weights), data.iloc[:, 7].values, kecimens, besnis)
    k_plot, b_plot = plot_decision_boundary(fig, ax, cel, 0, 0, 0, kecimens, besnis)

    for i in range(epoch):
        print(i)
        weights = update_weights(features, data_values, weights, train_rate)
        cel = cross_entropy_loss(features, data_values, weights)
        cel_values[i] = cel
        if not i % 5000:
            kecimens, besnis = classify(predict(features, weights), data.iloc[:, 7].values, kecimens, besnis)
            k_plot, b_plot = plot_decision_boundary(fig, ax, cel, i, k_plot, b_plot, kecimens, besnis)
        kecimens = []
        besnis = []
    mp.show()
    plot_cross_entropy_loss(cel_values, cel, weights, n)


# function that visualizes how accurate our model is, by plotting the decision boundary (0.5) on top of our predictions
# to see how predictions compare to the actual raisin classes (labels) originally assigned to them from the dataset
def plot_decision_boundary(fig, ax, cel, epoch, k_plot, b_plot, kecimens, besnis):
    if k_plot and b_plot:
        k_plot.remove()
        b_plot.remove()
    ax.set_title("Epoch: %s with Cross Entropy Loss %s" % (str(epoch), format(cel, '.2f')))
    k_plot = ax.scatter([i for i in range(len(kecimens))], kecimens, c='red', label='Kecimens')
    b_plot = ax.scatter([i for i in range(len(besnis))], besnis, c='blue', label='Besnis')
    ax.legend(loc="upper right", fontsize='small')
    ax.set_ylabel("Predicted Probability")
    fig.canvas.draw()
    fig.canvas.flush_events()
    return k_plot, b_plot


def plot_cross_entropy_loss(cel_values, cel, weights, n):
    mp.ioff()
    fig = mp.figure()
    ax = fig.add_subplot(111)
    ax.plot(cel_values, color='red')
    ax.set_title("Cross Entropy Loss %s" % format(cel, '.2f'))
    legend = "with prediction function "
    for i in range(n):
        legend += "%s * x%s + " % (format(weights[i], '.2f'), str(i))
    legend += format(weights[0], '.2f')
    ax.legend([legend, "data_values"],
              fontsize='small')
    mp.show(block=True)
    return fig, ax


# Main program entry point
def main():
    binary_logistic_regression()


if __name__ == '__main__':
    main()
