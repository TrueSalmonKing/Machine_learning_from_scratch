import Neural_Network as pr
import numpy as np
import pandas as pd
import category_encoders as ce


if __name__ == '__main__':
    """We load the mushroom dataset into a dataframe"""
    data = pd.read_csv('data/agaricus-lepiota.data',
                       header=None)

    # print("stalk-root attribute values' frequencies:\n", data[11].value_counts())

    """Seeing as we have 2480 missing values for the stalk-root attribute, we will replace these values with the most
    frequent value (in our case it is b)"""
    data[11] = data[11].replace('?', data[11].mode()[0])

    """Binary encoding the attributes, in addition to the classes
    This results in two class labels' columns, in our case we choose the second one"""
    be = ce.BinaryEncoder(cols=data.columns)
    data = be.fit_transform(data)

    """We transform the dataframe into an np array and we shuffle the data"""
    data = np.array(data[1:])
    m, n = data.shape
    np.random.shuffle(data)

    """We create the network with 2 hidden layers and one output layer with one node"""
    nn = pr.Network(loss_function=pr.mean_square_error)
    nn.add_layer(pr.Layer(n-1, 10, pr.tanh_activation))
    nn.add_layer(pr.Layer(10, 2, pr.tanh_activation))
    nn.add_layer(pr.Layer(2, 1, pr.sigmoid_activation))

    # nn.hold_out(data[:, 1:n], data[:, 0], 0.1, 2000)
    nn.cross_validation(data[:, 1:n], data[:, 0], 5, 0.05, 1000)
