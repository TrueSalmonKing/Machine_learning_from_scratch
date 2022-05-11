import pandas as pd
import matplotlib.pyplot as mp
import numpy as np

# Taken from https://www.kaggle.com/datasets/quantbruce/real-estate-price-prediction
data = pd.read_csv('../data/Real estate.csv')


# ---------------------------------------Simple regression--------------------------------------- #


# Calculating the sample variance (mean square error)
# mean square error => E = (Sum{1..n}((Yi - mean(Xi))**2)/(n-1) with mean(Xi) = m*Xi + b
def mean_square_error(features, m_i, b_i):
    variance = int(0)
    for i in range(len(features)):
        x = features.iloc[i, 2]
        y = features.iloc[i, 7]
        variance += (y - (x * m_i + b_i)) ** 2
    return variance / (len(features) - 1)


# Calculating the gradient descent of the square mean error in function of the slope and the feature b
# in order to minimize the error function (sample variance) by updating the weights
# partial derivative d(E(m))/d(m) = (1/n)*(Sum{1..n}(2*(Yi-(m*Xi+b))*(-Xi)) = (-2/n)*(Sum{1..n}(Xi*(Yi-(m*Xi+b)))
# partial derivative d(E(b))/d(b) = (1/n)*(Sum{1..n}(2*(Yi-(m*Xi+b))*(-1)) = (-2/n)*(Sum{1..n}(Yi-(m*Xi+b))
def update_weights(features, m_i, b_i, learning_rate):
    m_gradient = 0
    b_gradient = 0
    for i in range(len(features)):
        x = features.iloc[i, 2]
        y = features.iloc[i, 7]
        m_gradient += x * (y - (m_i * x + b_i))
        b_gradient += (y - (m_i * x + b_i))
    m_gradient = (-2 * m_gradient) / len(features)
    b_gradient = (-2 * b_gradient) / len(features)
    print("m_gradient descent yields : " + str(m_gradient))
    print("b_gradient descent yields : " + str(b_gradient))
    m = m_i - learning_rate * m_gradient
    b = b_i - learning_rate * b_gradient
    return m, b


# Linear regression implementation, where we use gradient descent to minimize the mean square error
# In order to reach the closest possible line function that can predict values to the upmost accuracy
def linear_regression():
    train_rate = 0.0003
    m = 0
    b = 0
    epoch = 300

    x = data.iloc[:, 2]
    # Prediction function (i.e: linear function closest to predicting the values in the date set)
    y = m * x + b

    # Plotting the value-approximation linear function y = m*x + b, on top of the sample scatter plot data
    # in order to showcase the linear function enclosing on the best possible values for the two features: slope m and
    # constant b
    mp.ion()
    fig = mp.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data.iloc[:, 2], data.iloc[:, 7], color="black")
    line, = ax.plot(x, y, color="red")

    # Mean square error
    mse = mean_square_error(data, m, b)
    ax.set_title("Mean Square Error " + format(mse, '.2f'))

    for i in range(epoch):
        (m, b) = update_weights(data, m, b, train_rate)
        mse = mean_square_error(data, m, b)
        line.set_ydata((m * x + b))
        ax.set_title("Mean Square Error " + format(mse, '.2f'))
        ax.legend(["with prediction function" + format(m, '.2f') + " * x + " + format(b, '.2f'), "data_values"],
                  fontsize="small")
        fig.canvas.draw()
        fig.canvas.flush_events()
        print("variance is : " + str(mse))
        print("function is : " + str(m) + " * x + " + str(b))

    mp.show(block=True)


# ---------------------------------------Multivariable regression--------------------------------------- #

# Calculating gradient can take a long time as the number of features grow, as such we only focus on two features at
# a time (Transaction date, House age and Number of convenience stores)
def mv_features_selection(features):
    return mv_normalize(features.iloc[:, 2:5].values)


# We will also speed up the process by using Zero Mean Normalization. This will allow us to effectively turn the
# range of values for the features into [-1:1]
def mv_normalize(features):

    for i in range(3):
        f_mean = np.mean(features[:, i])
        f_range = np.max(features[:, i]) - np.min(features[:, i])
        features[:, i] = (features[:, i] - f_mean)/f_range

    return features


# Prediction function for multivariable regression (linear function closest to predicting the values in the data set)
# predictions (414,1) = features (414,3) . weights (3,1)
def mv_predict(features, weights):
    return np.dot(features, weights)


# Function that calculates the mean square error in order to optimize our weights for better predictions
# variance => E = (Sum{1..n}((Yi - mean(Xi))**2)/(n) with mean(Xi) = W1*X1 + W2*X2 + W3*X3
# predictions (414,1) = features (414,3) . weights (3,1)
def mv_mean_square_error(features, data_values, weights):
    # Matrix math allows us to do without loops
    square_error = ((data_values - mv_predict(features, weights))**2).sum() / len(data_values)

    return square_error


# Calculating the gradient descent of the square mean error in function of the weights in order to minimize the error
# function (sample variance) by updating the weights:
# partial derivative d(E(W1))/d(W1) = (-2/n)*(Sum{1..n}(Xi*(Yi-(W1*X1 + W2*X2 + W3*X3)))
# partial derivative d(E(W2))/d(W2) = (-2/n)*(Sum{1..n}(Xi*(Yi-(W1*X1 + W2*X2 + W3*X3)))
# partial derivative d(E(W3))/d(W3) = (-2/n)*(Sum{1..n}(Xi*(Yi-(W1*X1 + W2*X2 + W3*X3)))
def mv_update_weights(features, data_values, weights, learning_rate):
    predicted_values = mv_predict(features, weights)
    n = len(data_values)

    # Extracting the features
    x1 = features[:, 0]
    x2 = [0]*414
    x3 = features[:, 2]

    # print(np.dot(x3, (data_values - predicted_values)))

    # Using the dot product to calculate the partial derivative for each weight
    dw1 = (-2.0 * np.dot(x1, (data_values - predicted_values))) / n
    dw2 = (-2.0 * np.dot(x2, (data_values - predicted_values))) / n
    dw3 = (-2.0 * np.dot(x3, (data_values - predicted_values))) / n

    weights[0] -= (learning_rate * dw1)
    weights[1] -= (learning_rate * dw2)
    weights[2] -= (learning_rate * dw3)

    return weights


# Linear regression implementation, where we use gradient descent to minimize the mean square error
# In order to reach the closest possible line function that can predict values to the upmost accuracy
def multivariable_regression():
    train_rate = 0.005
    weights = [0.0, 0.0, 0.0]
    epoch = 300
    mse_values = [0]*300

    features = mv_features_selection(data)
    data_values = data.iloc[:, 7]
    mse = mv_mean_square_error(features, data_values, weights)
    print("mean square error is : " + str(mse))

    # Plotting the value-approximation linear function y = m*x + b, on top of the sample scatter plot data
    # in order to showcase the linear function enclosing on the best possible values for the two features: slope m and
    # constant b

    for i in range(epoch):
        weights = mv_update_weights(features, data_values, weights, train_rate)
        mse = mv_mean_square_error(features, data_values, weights)
        print("mean square error is : " + str(mse))
        print("function is : " + str(weights[0]) + "*x1 + " + str(weights[1]) + "*x2 + " + str(weights[2]) + "*x3")
        mse_values[i] = mse

    fig = mp.figure()
    ax = fig.add_subplot(111)
    ax.plot(mse_values, color="red")
    ax.set_title("Mean Square Error " + format(mse, '.2f'))
    ax.legend(["with prediction function" + format(weights[0], '.2f') + " * x1 + " + format(weights[1], '.2f') +
               " * x2 + " + format(weights[2], '.2f') + " * x3", "data_values"],
              fontsize="small")
    mp.show(block=True)


# Main program entry point
def main():
    # linear_regression()
    multivariable_regression()
    # print(mv_features_selection(data))
    # print("normalize")
    # print(mv_normalize(mv_features_selection(data)))


if __name__ == '__main__':
    main()
