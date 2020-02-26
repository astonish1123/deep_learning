import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

if __name__ == '__main__':
    learnrate = 0.5
    x = np.array([1, 2, 3, 4])
    y = np.array([0.5])

    w = np.array([0.5, -0.5, 0.3, 0.1])

    h = np.dot(x, w)
    nn_output = sigmoid(h)
    error = y - nn_output
    error_term = error * sigmoid_prime(h)
    del_w = [learnrate * error_term * x[0], learnrate * error_term * x[1], 
             learnrate * error_term * x[2], learnrate * error_term * x[2]]
    
    print('Neural Network output: ')
    print(nn_output)
    print('Amount of Error: ')
    print(error)
    print('Change in Weigts: ')
    print(del_w)