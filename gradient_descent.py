import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_points(X, y):
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s=25, color='blue', edgecolor='k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s=25, color='red', edgecolors='k')

def display(m, b, color='g--'):
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    x = np.arange(-10, 10, 0.1)
    plt.plot(x, m*x+b, color)

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

def output_formula(features, weights, bias):
    func_result = weights*features+bias
    y_hat = sigmoid(func_result)
    return y_hat

def error_formula(y, output):
    error = -y*np.log(output)-(1-y)*np.log(1-output)
    return error

def update_weights(x, y, weights, bias, learnrate):
    event = output_formula(x, weights, bias)
    d_error = -(y - event)
    weights -= learnrate * d_error * x
    bias -= learnrate * d_error
    return weights, bias


if __name__ == '__main__':
    data = pd.read_csv('data.csv', header=None)
    X = np.array(data[[0, 1]])
    y = np.array(data[2])

    plot_points(X, y)
    plt.show()