#This script is not working, just for understanding. 
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

learnrate = 0.5

#Network size
N_input = 4
N_hidden = 3
N_output = 2

np.random.seed(42)
#Forward calculation 
#Generate fake inputs
X = np.random.randn(4)
target = None # not defined

weights_input_to_hidden = np.random.normal(0, scale=1/N_input**.5, size=(N_input, N_hidden))
weights_hidden_to_output = np.random.normal(0, scale=1/N_input**.5, size=(N_hidden, N_output))

hidden_layer_in = np.dot(X, weights_input_to_hidden)
hidden_layer_out = sigmoid(hidden_layer_in)

output_layer_in = np.dot(hidden_layer_out, weights_input_to_hidden)
output_layer_out = sigmoid(output_layer_in)

#Backward calculation
error = target - output_layer_out

output_grad = sigmoid_prime(output_layer_out)
output_error_term = error * output_grad

hidden_layer_backin = np.dot(output_error_term, weights_hidden_to_output)
hidden_layer_backout = sigmoid_prime(hidden_layer_out)
hidden_error_term = hidden_layer_backin * hidden_layer_backout

delta_w_h_o = learnrate * output_error_term * hidden_layer_out
delta_w_i_h = learnrate * hidden_error_term * X[:, None]


