import pandas as pd
import matplotlib.pyplot as mp
import numpy as np

# Taken from https://www.kaggle.com/datasets/muratkokludataset/date-fruit-datasets

data = pd.read_excel("../data/Date_Fruit_Datasets/Date_Fruit_Datasets.xlsx", engine='openpyxl')


# In order to avoid the issue of exploding gradient and to converge towards the minimum faster, we normalize the
# range of values for the features into [-1:1]
def mv_normalize(features):
    for i in range(35):
        f_mean = np.mean(features[:, i])
        f_range = np.amax(features[:, i]) - np.amin(features[:, i])
        if f_range:
            features[:, i] = (features[:, i] - f_mean) / f_range

    return features


# We will be working with 34 + 1 (bias) features to train this model
def features_selection(features):
    bias = np.ones(shape=(len(features), 1))
    return mv_normalize(np.append(bias, features.iloc[:, 0:34].values, axis=1))


# Prediction function for multivariable logistic regression (linear function closest to predicting the values in
# the data set): predictions (X,K) = features (X,35) . weights (35,K) with K being the number of classes
def predict(features, weights):
    return softmax_activation(np.dot(features, weights))


# We map our predictions into probabilities (values between 0 and 1) using the softmax activation function:
# S(Xji) = e(Xji) / sum{i..K}(z(Xji))) with K being the number of classes, and j being the sample size
# This function is the general case for the sigmoid function used in the binary logistic regression with K classes
def softmax_activation(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1)


# We will use the cost function cross entropy loss instead of mean square error, to optimize our weights for
# better predictions
# cross entropy loss => cel() = -1*(Sum{1..n}(Sum{1..K}((Yji * log(S(Xji))))/(n) with Yji being 1 if the target class
# for Xji is K otherwise it is 0. In order to facilitate this computation we vectorize the Yji values into a vector with
# only 1 in the correct target class and 0s in the rest of the K-1 classes' placement --> one-hot encoded.
# cel (1,1) = np.mean(one_hot_data_values (X, K) * log(predictions) (X, K)) [we multiply row by row]
def cross_entropy_loss(features, one_hot_data_values, weights):
    predictions = predict(features, weights)
    return -1 * np.mean(one_hot_data_values * np.log(predictions))


# Calculating the gradient descent of the softmax activation function with respect to the weights in order to
# minimize the error function by updating the weights:
# partial derivative d(cle()/d(Wi) = (1/n)*(Sum{1..n}((Xji-Yji) * xi) with xi being a feature
# partial derivative d(cle()/d(b) = (1/n)*(Sum{1..n}((Xji-Yji))
def update_weights(features, one_hot_data_values, weights, learning_rate):
    predicted_values = predict(features, weights)
    n = np.shape(one_hot_data_values)[0]

    # gradient (3,K) = features.T (3,414) . ( data_values - predicted_values ) (414,K)
    gradient = (np.dot(features.T,  predicted_values - one_hot_data_values)) / n

    weights = weights - (learning_rate * gradient)

    return weights


# function that maps a class names to a value between [0, ..., 1/(n-1)] with n being the number of classes, in our
# classes, it is n = 7
def class_name_id(name):
    if name == 'BERHI':
        return 0
    if name == 'DEGLET':
        return 1
    if name == 'DOKOL':
        return 2
    if name == 'IRAQI':
        return 3
    if name == 'ROTANA':
        return 4
    if name == 'SAFAVI':
        return 5
    if name == 'SOGAY':
        return 6


# function that applies the function class_name_id onto the list of data values containing the list containing
# class names
def class_name_id_creation(data_values, class_name_id_func):
    class_name_id_func = np.vectorize(class_name_id_func)
    return class_name_id_func(data_values)


# function that maps the predictions onto each class they "should" be mapped to, meaning it will allow us to
# visualize the cross entropy loss by viewing the predictions that are correct and that ones that aren't
def classify(predictions, data_values, types):
    i = 0
    for prediction in predictions:
        if data_values[i] == 'BERHI':
            types[0].append(prediction)
            i += 1
            continue
        if data_values[i] == 'DEGLET':
            types[1].append(prediction)
            i += 1
            continue
        if data_values[i] == 'DOKOL':
            types[2].append(prediction)
            i += 1
            continue
        if data_values[i] == 'IRAQI':
            types[3].append(prediction)
            i += 1
            continue
        if data_values[i] == 'ROTANA':
            types[4].append(prediction)
            i += 1
            continue
        if data_values[i] == 'SAFAVI':
            types[5].append(prediction)
            i += 1
            continue
        if data_values[i] == 'SOGAY':
            types[6].append(prediction)
            i += 1
            continue
        i += 1

    return types


def one_hot(data_values, n):
    one_hot_data_values = np.array([[0.0] * n] * len(data_values))
    for i in range(len(data_values)):
        one_hot_data_values[i][data_values[i]] = 1

    return one_hot_data_values


# Multivariable logistic regression implementation, where we use gradient descent to minimize the cross entropy loss
# In order to reach the closest possible line function that can predict values to the upmost accuracy
def multivariable_logistic_regression():
    train_rate = 0.05
    epoch = 100000
    cel_values = [0] * epoch

    features = features_selection(data)
    data_values = class_name_id_creation(data.iloc[:, 34].values, class_name_id)
    one_hot_matrix = one_hot(data_values, 7)
    weights = np.array([[0.0] * 7] * 35)
    cel = cross_entropy_loss(features, one_hot(data_values, 7), weights)

    for i in range(epoch):
        weights = update_weights(features, one_hot_matrix, weights, train_rate)
        cel = cross_entropy_loss(features, one_hot_matrix, weights)
        cel_values[i] = cel
        if not i%10000:
            print(cel)
    mp.show()
    plot_cross_entropy_loss(cel_values, cel)


def plot_cross_entropy_loss(cel_values, cel):
    mp.ioff()
    fig = mp.figure()
    ax = fig.add_subplot(111)
    ax.plot([i for i in range(len(cel_values))], cel_values, color='red')
    ax.set_title("Cross Entropy Loss %s" % format(cel, '.2f'))
    mp.show(block=True)
    return fig, ax


# Main program entry point
def main():
    multivariable_logistic_regression()


if __name__ == '__main__':
    main()
