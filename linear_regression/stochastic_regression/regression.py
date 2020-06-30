"""
Author: Prateek Kumar Oraon
Copyright Prateek Kumar Oraon, free to use under MIT License
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class LinearRegression(object):
    def __init__(self, x_train, y_train, x_test, y_test, learning_rate=0.001):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.learning_rate = learning_rate
        self.optimal_w = None
        self.optimal_b = None
        self.cost_train = None
        self.cost_test = None

    def cost_function(self, b, m, features, target):
        totalError = 0
        for i in range(0, len(features)):
            x = features
            y = target
            totalError += (y[:, i] - (np.dot(x[i], m) + b)) ** 2
        return totalError / len(x)

    def r_sq_score(self, b, m, features, target):
        for i in range(0, len(features)):
            x = features
            y = target
            mean_y = np.mean(y)
            ss_tot = sum((y[:, i] - mean_y) ** 2)
            ss_res = sum(((y[:, i]) - (np.dot(x[i], m) + b)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
        return r2

    def gradient_decent(self, w0, b0, train_data, x_test, y_test, learning_rate):
        n_iter = 500
        partial_deriv_m = 0
        partial_deriv_b = 0
        cost_train = []
        cost_test = []
        for j in range(1, n_iter):

            train_sample = train_data.sample(160)
            y = np.asmatrix(train_sample["PRICE"])
            x = np.asmatrix(train_sample.drop("PRICE", axis=1))
            for i in range(len(x)):
                partial_deriv_m += np.dot(-2 * x[i].T, (y[:, i] - np.dot(x[i], w0) + b0))
                partial_deriv_b += -2 * (y[:, i] - (np.dot(x[i], w0) + b0))

            w1 = w0 - learning_rate * partial_deriv_m
            b1 = b0 - learning_rate * partial_deriv_b

            if (w0 == w1).all():
                break
            else:
                w0 = w1
                b0 = b1
                learning_rate = learning_rate / 2

            error_train = self.cost_function(b0, w0, x, y)
            cost_train.append(error_train)
            error_test = self.cost_function(b0, w0, np.asmatrix(x_test), np.asmatrix(y_test))
            cost_test.append(error_test)

        return w0, b0, cost_train, cost_test

    def fit(self):
        w0_random = np.random.rand(13)
        w0 = np.asmatrix(w0_random).T
        b0 = np.random.rand()

        self.optimal_w, self.optimal_b, self.cost_train, self.cost_test = self.gradient_decent(w0, b0, self.x_train,
                                                                                               self.x_test, self.y_test,
                                                                                               self.learning_rate)
        print("Coefficient: {} \n y_intercept: {}".format(self.optimal_w, self.optimal_b))

    def plot_error(self):
        # Plot train and test error in each iteration
        plt.figure()
        plt.plot(range(len(self.cost_train)), np.reshape(self.cost_train, [len(self.cost_train), 1]),
                 label="Train Cost")
        plt.plot(range(len(self.cost_test)), np.reshape(self.cost_test, [len(self.cost_test), 1]), label="Test Cost")
        plt.title("Cost/loss per iteration")
        plt.xlabel("Number of iterations")
        plt.ylabel("Cost/Loss")
        plt.legend()
        plt.show()


def run():
    boston = load_boston()
    print(boston.feature_names)
    dataset = pd.DataFrame(boston.data)

    print(dataset.describe())

    dataset = (dataset - dataset.mean()) / dataset.std()
    dataset["PRICE"] = boston.target
    print(dataset.head())

    Y = dataset["PRICE"]
    X = dataset.drop("PRICE", axis=1)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    x_train["PRICE"] = y_train

    model = LinearRegression(x_train, y_train, x_test, y_test)
    model.fit()

    model.plot_error()


if __name__ == '__main__':
    run()
