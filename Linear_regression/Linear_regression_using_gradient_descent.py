import pandas as pd
import matplotlib.pyplot as mp

data = pd.read_csv('../data/Real estate.csv')


# Calculating the sample variance
# variance => E = (Sum{1..n}((Yi - mean(Xi))**2)/(n-1) with mean(Xi) = m*Xi + b
def variance_error(data_points, m_i, b_i):
    variance = int(0)
    for i in range(len(data_points)):
        x = data_points.iloc[i, 2]
        y = data_points.iloc[i, 7]
        variance += (y - (x * m_i + b_i)) ** 2
    return variance / (len(data_points) - 1)


# Calculating the gradient descent of the variance in function of the slope and the constant b
# in order to minimize the error function (sample variance)
# gradient descent for d(E(m))/d(m) = (1/n)*(Sum{1..n}(2*(Yi-(m*Xi+b))*(-Xi)) = (-2/n)*(Sum{1..n}(Xi*(Yi-(m*Xi+b)))
# gradient descent for d(E(b))/d(b) = (1/n)*(Sum{1..n}(2*(Yi-(m*Xi+b))*(-1)) = (-2/n)*(Sum{1..n}(Yi-(m*Xi+b))
def gradient_descent(data_points, m_i, b_i, training_rate):
    m_gradient = 0
    b_gradient = 0
    for i in range(len(data_points)):
        x = data_points.iloc[i, 2]
        y = data_points.iloc[i, 7]
        m_gradient += x * (y - (m_i * x + b_i))
        b_gradient += (y - (m_i * x + b_i))
    m_gradient = (-2 * m_gradient) / len(data_points)
    b_gradient = (-2 * b_gradient) / len(data_points)
    print("m_gradient descent yields : " + str(m_gradient))
    print("b_gradient descent yields : " + str(b_gradient))
    m = m_i - training_rate * m_gradient
    b = b_i - training_rate * b_gradient
    return m, b


# Linear regression implementation, where we use gradient descent to minimize the sample variance
# In order to reach the closest possible line function that can predict values to the upmost accuracy
def linear_regression():
    train_rate = 0.0001
    m = 0
    b = 0
    epoch = 300

    x = data.iloc[:, 2]
    y = m * x + b

# Plotting the value-approximation linear function y = m*x + b, on top of the sample scatter plot data
# in order to showcase the linear function enclosing on the best possible values for the slope m and
# constant b
    mp.ion()
    fig = mp.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data.iloc[:, 2], data.iloc[:, 7], color="black")
    line, = ax.plot(x, y, color="red")

    for i in range(epoch):
        (m, b) = gradient_descent(data, m, b, train_rate)
        line.set_ydata((m * x + b))
        fig.canvas.draw()
        fig.canvas.flush_events()
        print("variance is : " + str(variance_error(data, m, b)))
        print("function is : " + str(m) + " * x + " + str(b))


# Main program entry point
def main():
    linear_regression()


if __name__ == '__main__':
    main()
