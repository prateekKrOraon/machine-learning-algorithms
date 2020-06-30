"""
Author: Prateek Kumar Oraon
Copyright Prateek Kumar Oraon, free to use under MIT License
"""

import numpy as np
import pandas as pd


def build_var_matrix(data):
    matrix = []
    for item_list in data:
        x = [1]
        for item in item_list:
            x.append(item)
        matrix.append(x)

    return matrix


def matrix_mul(x, y):
    rows = len(x)

    cols = len(y[0])

    result = [[0 for x in range(cols)] for y in range(rows)]

    for i in range(rows):
        for j in range(cols):
            for k in range(len(y)):
                result[i][j] += x[i][k] * y[k][j]

    return result


def transpose(matrix):
    rows, cols = np.shape(matrix)
    new_matrix = [[0 for x in range(rows)] for y in range(cols)]
    for i in range(cols):
        for j in range(rows):
            new_matrix[i][j] = matrix[j][i]

    return new_matrix


class Matrix(object):
    def __init__(self, matrix):
        self.matrix = matrix

    def cofactor(self, mat, p, q, n):
        # mat = self.matrix
        i = 0
        j = 0
        temp = [[0 for x in range(n)] for y in range(n)]
        for row in range(n):
            for col in range(n):
                if row != p and col != q:
                    temp[i][j] = mat[row][col]
                    j += 1
                    if j == n - 1:
                        j = 0
                        i += 1

        return temp

    def determinant(self, mat, n):
        if mat is None:
            mat = self.matrix
            n = len(mat)

        d = 0

        if n == 1:
            return mat[0][0]

        multiplier = 1

        for i in range(n):
            temp = self.cofactor(mat, 0, i, n)
            d += multiplier * mat[0][i] * self.determinant(temp, n - 1)
            multiplier = -multiplier

        return d

    def adjoint(self):
        mat = self.matrix
        n = len(mat)
        if n == 1:
            return [[1]]

        multiplier = 1

        adj = [[0 for x in range(n)] for y in range(n)]

        for i in range(n):
            for j in range(n):
                temp = self.cofactor(mat, i, j, n)
                if (i + j) % 2 == 0:
                    multiplier = 1
                else:
                    multiplier = -1

                adj[j][i] = multiplier * self.determinant(temp, n - 1)

        return adj

    def inverse(self):
        mat = self.matrix
        n = len(mat)
        det = self.determinant(mat, n)
        if det == 0:
            print("Inverse does not exist")
            return False

        adj = self.adjoint()

        inv = [[0 for x in range(n)] for y in range(n)]

        for i in range(n):
            for j in range(n):
                inv[i][j] = adj[i][j] / det

        return inv


class LinearRegressionMul(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.coefficients = []

    def fit(self):
        x = self.x
        y = self.y

        x_mat = build_var_matrix(x)
        x_trans = transpose(x_mat)
        x_mat = build_var_matrix(x)
        x_mult = matrix_mul(x_trans, x_mat)
        mat = Matrix(x_mult)
        x_mult_inv = mat.inverse()
        y_mat = np.reshape(y, (len(y), 1))
        y_mult = matrix_mul(x_trans, y_mat)
        self.coefficients = np.reshape(matrix_mul(x_mult_inv, y_mult),(3))
        print("Coefficients")
        print(self.coefficients)

    def predict(self, x):
        coeff = self.coefficients
        y_pred = []
        val = coeff[0]
        j = 1
        for x_val in x:
            for i in range(len(coeff) - 1):
                val += coeff[i + 1] * x_val[i]
            y_pred.append(val)
            val = coeff[0]
            j += 1
        return y_pred


def run():
    dataset = pd.read_csv('regression_4.csv')
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    model = LinearRegressionMul(x, y)
    model.fit()
    y_pred = model.predict([[35, 250], [40, 250], [40, 300]])
    print("Predictions")
    print(y_pred)


if __name__ == '__main__':
    run()
