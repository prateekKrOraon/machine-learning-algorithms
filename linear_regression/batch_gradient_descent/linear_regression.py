"""
Author: Prateek Kumar Oraon
Copyright Prateek Kumar Oraon, free to use under MIT License
"""

import pandas as pd
import matplotlib.pyplot as plt


class LinearRegression(object):
    def __init__(self, x, y, epochs, learning_rate=0.0001):
        self.x = x
        self.y = y
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.m = 0
        self.c = 0

    def fit(self):
        m = 0
        c = 0

        n = float(len(self.x))

        for i in range(self.epochs):
            y_pred = m * self.x + c
            d_m = (-2 / n) * sum(self.x * (self.y - y_pred))
            d_c = (-2 / n) * sum(self.y - y_pred)
            m = m - self.learning_rate * d_m
            c = c - self.learning_rate * d_c

        self.m = m
        self.c = c

        print(self.m, self.c)

    def predict(self, x):
        y_pred = []
        for item in x:
            y_pred.append(self.m * item + self.c)
        return y_pred

    def plot_result(self):
        y_pred = self.m * self.x + self.c
        plt.scatter(self.x, self.y)
        plt.plot([min(self.x), max(self.x)], [min(y_pred), max(y_pred)], color='red')
        plt.title('Best fitted Regression Line')
        plt.show()


def run():
    dataset = pd.read_csv('input.csv')
    X = dataset.iloc[:, 0].values
    Y = dataset.iloc[:, 1]
    plt.scatter(X, Y)
    plt.title('Points in Dataset')
    plt.show()

    model = LinearRegression(X, Y, 500)
    model.fit()

    model.plot_result()

    x_test = [33.3, 24.3, 15.5, 100.7, 33.1, 36.4, 36.8]
    print('Test set')
    print(x_test)
    y_pred = model.predict(x_test)
    print('Predicted values')
    print(y_pred)


if __name__ == '__main__':
    run()
