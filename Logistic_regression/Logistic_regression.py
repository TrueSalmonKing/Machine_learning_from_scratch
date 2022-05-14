import pandas as pd
import matplotlib.pyplot as mp
import numpy as np

# Taken from https://www.kaggle.com/datasets/muratkokludataset/raisin-dataset
data = pd.read_excel("../data/Raisin_Dataset/Raisin_Dataset.xlsx", engine='openpyxl')


def features_selection(features):
    # Initial value of the bias is set to 1
    # With the addition of a bias, we allow our prediction function to be more accurate as it no longer strictly goes
    # through the origin (0, 0)
    # bias = np.ones(shape=(len(features), 1))
    # return np.append(bias, features.iloc[:, 0:7].values, axis=1)
    return features[:, 0:7]


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


# We will use the cost function called Cross_entropy instead of mean square error,
def cross_entropy_loss(features, data_values, weights):
    predictions = predict(features, weights)
    print(features_selection(predictions))
    data_values = np.dot(data_values, np.log(predictions))
    n = len(data_values)

    cost = -np.dot(data_values, np.log(predictions))
    cost -= np.dot(1 - data_values, np.log(1 - data_values))
    cost = cost.sum() / n

    return cost


def update_weights(features, data_values, weights, learning_rate):
    predicted_values = predict(features, weights)
    n = len(data_values)

    # gradient (3,1) = features.T (3,414) . ( data_values - predicted_values ) (414,1)
    gradient = (np.dot(features.T, (predicted_values - data_values))) / n

    weights -= (learning_rate * gradient)

    return weights


def decision_boundary(prob):
    return 1 if prob >= 0.5 else 0


def class_name_id(name):
    return 1 if name == "Kecimen" else 0


def class_name_id_creation(data_values, class_name_id_func):
    class_name_id_func = np.vectorize(class_name_id_func)
    return class_name_id_func(data_values)


def classify(predictions, decision_boundary_func):
    decision_boundary_func = np.vectorize(decision_boundary_func)
    return decision_boundary_func(predictions)


def binary_linear_regression():
    train_rate = 0.009
    epoch = 30000
    cel_values = [0] * epoch

    features = features_selection(data)
    weights = [0.0] * 7
    data_values = class_name_id_creation(data.iloc[:, 7].values, class_name_id)
    cel = cross_entropy_loss(features, data_values, weights)
    # print("Initial mean square error is : " + str(mse))

    # Plotting the value-approximation linear function y = m*x + b, on top of the sample scatter plot data
    # in order to showcase the linear function enclosing on the best possible values for the two features: slope m and
    # constant b

    for i in range(epoch):
        weights = update_weights(features, data_values, weights, train_rate)
        cel = cross_entropy_loss(features, data_values, weights)
        # print("mean square error is : " + str(mse))
        # print("function is : " + str(weights[1]) + "*x1 + " + str(weights[2]) + "*x2 + " +
        #      str(weights[3]) + "*x3 + " + str(weights[0]))
        cel_values[i] = cel

    fig = mp.figure()
    ax = fig.add_subplot(111)
    ax.plot(cel_values, color="red")
    ax.set_title("Mean Square Error " + format(cel, '.2f'))
    ax.legend(["with prediction function" + format(weights[1], '.2f') + " * x1 + " + format(weights[2], '.2f') +
               " * x2 + " + format(weights[3], '.2f') + " * x3 + " + format(weights[0]), "data_values"],
              fontsize="small")
    mp.show(block=True)


# Main program entry point
def main():
    binary_linear_regression()
    # weights = [0.0] * 8
    # print(features_selection(data))
    # predictions = predict(features_selection(data), weights)
    # weights = update_weights(features_selection(data), class_name_id_creation(data.loc[:, 7].values, class_name_id),
    # weights, 0.004)
    # print(weights)
    # print(class_name_id_creation(data.loc[:, 7].values, class_name_id))


if __name__ == '__main__':
    main()
