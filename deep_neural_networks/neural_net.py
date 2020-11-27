import numpy as np
import h5py
import matplotlib.pyplot as plt
from deep_neural_networks.activations import relu, sigmoid, sigmoid_derivative, relu_derivative


def load_dataset():
    training_set = h5py.File('dataset/train_catvnoncat.h5', 'r')
    training_set_X = np.array(training_set['train_set_x'][:])
    training_set_Y = np.array(training_set['train_set_y'][:])

    test_set = h5py.File('dataset/test_catvnoncat.h5', 'r')
    test_set_X = np.array(test_set['test_set_x'][:])
    test_set_Y = np.array(test_set['test_set_y'][:])

    classes = np.array(test_set['list_classes'][:])

    training_set_Y = training_set_Y.reshape((1, training_set_Y.shape[0]))
    test_set_Y = test_set_Y.reshape((1, test_set_Y.shape[0]))

    return training_set_X, training_set_Y, test_set_X, test_set_Y


def initialize_parameters(layer_dims):
    np.random.seed(1)
    params = {}
    L = len(layer_dims)

    for l in range(1, L):
        params["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])  # * 0.01
        params["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return params


def activation_forward(A_prev, W, b, activation):
    Z = W.dot(A_prev) + b
    linear_cache = (A_prev, W, b)
    assert (Z.shape == (W.shape[0], A_prev.shape[1]))

    if activation == 'relu':
        A, activation_cache = relu(Z)
    elif activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))

    caches = (linear_cache, activation_cache)

    return A, caches


def forward_propagation(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], 'relu')
        caches.append(cache)

    AL, cache = activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], 'sigmoid')

    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches


def compute_cost(A, Y):
    # number of examples
    m = Y.shape[1]

    cost = (-1 / m) * np.sum((Y * np.log(A) + ((1 - Y) * np.log(1 - A))))

    # To make sure the cost is a number and not a vector
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    return cost


def activation_backward(dA, activation_cache, activation):
    if activation == 'sigmoid':
        return sigmoid_derivative(dA, activation_cache)
    elif activation == 'relu':
        return relu_derivative(dA, activation_cache)


def compute_gradients(dA, cache, activation):
    linear_cache, activation_cache = cache
    A_prev, W, b = linear_cache

    m = A_prev.shape[1]

    dZ = activation_backward(dA, activation_cache, activation)

    dW = 1. / m * np.dot(dZ, A_prev.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def backward_propagation(A, Y, caches):
    gradients = {}
    L = len(caches)
    Y = Y.reshape(A.shape)

    dAL = -(np.divide(Y, A) - np.divide(1 - Y, 1 - A))

    current_cache = caches[L - 1]

    gradients["dA" + str(L - 1)], gradients["dW" + str(L)], gradients["db" + str(L)] = compute_gradients(dAL,
                                                                                                         current_cache,
                                                                                                         'sigmoid')

    # from L-2 to 0
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = compute_gradients(gradients["dA" + str(l + 1)], current_cache, 'relu')
        gradients["dA" + str(l)] = dA_prev_temp
        gradients["dW" + str(l + 1)] = dW_temp
        gradients["db" + str(l + 1)] = db_temp

    return gradients


def update_parameters(params, gradients, learning_rate):
    L = len(params) // 2

    for l in range(L):
        params["W" + str(l + 1)] = params["W" + str(l + 1)] - (learning_rate * gradients["dW" + str(l + 1)])
        params["b" + str(l + 1)] = params["b" + str(l + 1)] - (learning_rate * gradients["db" + str(l + 1)])

    return params


def predict(X, y, parameters):
    m = X.shape[1]
    n = len(parameters) // 2
    p = np.zeros((1, m))

    probs, caches = forward_propagation(X, parameters)

    for i in range(0, probs.shape[1]):
        if probs[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print("Accuracy: " + str(np.sum((p == y) / m)*100) + "%")

    return p


def run_model(X_train, Y_train, layer_dims, num_iterations, learning_rate=0.0075):
    np.random.seed(1)
    costs = []

    # Weights (W) and intercepts (b) for each layer of Neural Network
    params = initialize_parameters(layer_dims)

    for i in range(0, num_iterations):

        AL, caches = forward_propagation(X_train, params)

        cost = compute_cost(AL, Y_train)

        gradients = backward_propagation(AL, Y_train, caches)

        params = update_parameters(params, gradients, learning_rate)

        if i % 100 == 0:
            print("Cost after {j} iterations is {c}".format(j=i, c=cost))
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return params


if __name__ == '__main__':
    X_train_orig, Y_train, X_test_orig, Y_test = load_dataset()

    # Reshape training and test examples
    X_train_flatten = X_train_orig.reshape((X_train_orig.shape[0], -1)).T
    X_test_flatten = X_test_orig.reshape((X_test_orig.shape[0], -1)).T

    # Standardize data to have feature values between 0 and 1
    X_train = X_train_flatten / 255
    X_test = X_test_flatten / 255

    # Number of hidden nodes in each layer of Neural Network.
    # layer_dims[0] being the features of each training example.
    # layer_dims[-1] being the output layer.
    layer_dims = [X_train.shape[0], 20, 7, 5, 1]

    parameters = run_model(X_train, Y_train, layer_dims, 2500, learning_rate=0.0075)
    print("Training Set:")
    predict(X_train, Y_train, parameters)
    print("Test Set:")
    predict(X_test, Y_test, parameters)
