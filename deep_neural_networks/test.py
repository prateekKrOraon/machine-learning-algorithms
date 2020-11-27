import unittest
import numpy as np
from deep_neural_networks.neural_net import update_parameters, compute_cost, compute_gradients, \
    backward_propagation, forward_propagation


class ForwardPropagationTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)
        self.X = np.random.randn(4, 2)
        W1 = np.random.randn(3, 4)
        b1 = np.random.randn(3, 1)
        W2 = np.random.randn(1, 3)
        b2 = np.random.randn(1, 1)
        self.parameters = {
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2
        }

        self.expected_AL = [[0.17007265, 0.2524272]]

        self.arr = [
            (
                (
                    np.array(
                        [[1.62434536, -0.61175641],
                         [-0.52817175, -1.07296862],
                         [0.86540763, -2.3015387],
                         [1.74481176, -0.7612069]]
                    ),
                    np.array(
                        [[0.3190391, -0.24937038, 1.46210794, -2.06014071],
                         [-0.3224172, -0.38405435, 1.13376944, -1.09989127],
                         [-0.17242821, -0.87785842, 0.04221375, 0.58281521]]
                    ),
                    np.array(
                        [[-1.10061918],
                         [1.14472371],
                         [0.90159072]]
                    )
                ),
                np.array(
                    [[-2.77991749, -2.82513147],
                     [-0.11407702, -0.01812665],
                     [2.13860272, 1.40818979]]
                )
            ),
            (
                (
                    np.array(
                        [[0., 0.],
                         [0., 0.],
                         [2.13860272, 1.40818979]]
                    ),
                    np.array(
                        [[0.50249434, 0.90085595, -0.68372786]]
                    ),
                    np.array(
                        [[-0.12289023]]
                    )
                ),
                np.array(
                    [[-1.58511248, -1.08570881]]
                )
            )
        ]

    def test_func(self):
        AL, caches = forward_propagation(self.X, self.parameters)

        for i in range(len(caches)):
            expected_cache_one, expected_cache_two = self.arr[i]
            cache_one, cache_two = caches[i]

            assert (len(cache_one) == len(expected_cache_one) == 3)
            np.testing.assert_array_almost_equal(cache_one[0], expected_cache_one[0], decimal=8)
            np.testing.assert_array_almost_equal(cache_one[1], expected_cache_one[1], decimal=8)
            np.testing.assert_array_almost_equal(cache_one[2], expected_cache_one[2], decimal=8)

            assert (expected_cache_two.shape == cache_two.shape)
            np.testing.assert_array_almost_equal(cache_two[0], expected_cache_two[0], decimal=8)


class UpdateParametersTest(unittest.TestCase):

    def setUp(self):
        self.expected = {
            'W1': np.array(
                [
                    [-0.59562069, -0.09991781, -2.14584584, 1.82662008],
                    [-1.76569676, -0.80627147, 0.51115557, -1.18258802],
                    [-1.0535704, -0.86128581, 0.68284052, 2.20374577]
                ]
            ),
            'b1': np.array(
                [
                    [-0.04659241],
                    [-1.28888275],
                    [0.53405496]
                ]
            ),
            'W2': np.array(
                [
                    [-0.55569196, 0.0354055, 1.32964895]
                ]
            ),
            'b2': np.array(
                [
                    [-0.84610769]
                ]
            )
        }

        np.random.seed(2)
        W1 = np.random.randn(3, 4)
        b1 = np.random.randn(3, 1)
        W2 = np.random.randn(1, 3)
        b2 = np.random.randn(1, 1)
        self.parameters = {
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2
        }
        np.random.seed(3)
        dW1 = np.random.randn(3, 4)
        db1 = np.random.randn(3, 1)
        dW2 = np.random.randn(1, 3)
        db2 = np.random.randn(1, 1)
        self.grads = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2
        }

        self.learning_rate = 0.1

    def test_func(self):
        result = update_parameters(self.parameters, self.grads, 0.1)
        self.assertListEqual(list(result.keys()), list(self.expected.keys()))
        np.testing.assert_array_almost_equal(result['W1'], self.expected['W1'], decimal=8)
        np.testing.assert_array_almost_equal(result['W2'], self.expected['W2'], decimal=8)
        np.testing.assert_array_almost_equal(result['b1'], self.expected['b1'], decimal=8)
        np.testing.assert_array_almost_equal(result['b2'], self.expected['b2'], decimal=8)


class ComputeCostTest(unittest.TestCase):

    def setUp(self):
        self.Y = np.asarray([[1, 1, 0]])
        self.AL = np.array([[.8, .9, 0.4]])
        self.expected = 0.2797765635793422

    def test_func(self):
        result = compute_cost(self.AL, self.Y)
        self.assertAlmostEqual(result, self.expected, places=10)


class ComputeGradientTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(2)
        self.dAL = np.random.randn(1, 2)
        A = np.random.randn(3, 2)
        W = np.random.randn(1, 3)
        b = np.random.randn(1, 1)
        Z = np.random.randn(1, 2)
        linear_cache = (A, W, b)
        activation_cache = Z
        self.cache = (linear_cache, activation_cache)

        self.expected_dA_prev_sigmoid = [
            [0.11017994, 0.01105339],
            [0.09466817, 0.00949723],
            [-0.05743092, -0.00576154]
        ]
        self.expected_dW_sigmoid = [[0.10266786, 0.09778551, -0.01968084]]
        self.expected_db_sigmoid = [[-0.05729622]]

        self.expected_dA_prev_relu = [
            [0.44090989, 0],
            [0.37883606, 0],
            [-0.2298228, 0]
        ]
        self.expected_dW_relu = [[0.44513824, 0.37371418, -0.10478989]]
        self.expected_db_relu = [[-0.20837892]]

    def test_sigmoid(self):
        dA_prev, dW, db = compute_gradients(self.dAL, self.cache, 'sigmoid')
        np.testing.assert_array_almost_equal(dA_prev, self.expected_dA_prev_sigmoid, decimal=8)
        np.testing.assert_array_almost_equal(dW, self.expected_dW_sigmoid, decimal=8)
        np.testing.assert_array_almost_equal(db, self.expected_db_sigmoid, decimal=8)

    def test_relu(self):
        dA_prev, dW, db = compute_gradients(self.dAL, self.cache, 'relu')
        np.testing.assert_array_almost_equal(dA_prev, self.expected_dA_prev_relu, decimal=8)
        np.testing.assert_array_almost_equal(dW, self.expected_dW_relu, decimal=8)
        np.testing.assert_array_almost_equal(db, self.expected_db_relu, decimal=8)


class BackwardPropagationTest(unittest.TestCase):

    def setUp(self):
        self.expected_gradients = {
            'dW1': [
                [0.41010002, 0.07807203, 0.13798444, 0.10502167],
                [0, 0, 0, 0.],
                [0.05283652, 0.01005865, 0.01777766, 0.0135308]
            ],
            'db1': [
                [-0.22007063],
                [0],
                [-0.02835349]
            ],
            'dA1': [
                [0.12913162, -0.44014127],
                [-0.14175655, 0.48317296],
                [0.01663708, -0.05670698]
            ]
        }

        np.random.seed(3)
        self.AL = np.random.randn(1, 2)
        self.Y = np.array([[1, 0]])

        A1 = np.random.randn(4, 2)
        W1 = np.random.randn(3, 4)
        b1 = np.random.randn(3, 1)
        Z1 = np.random.randn(3, 2)
        linear_cache_activation_1 = ((A1, W1, b1), Z1)

        A2 = np.random.randn(3, 2)
        W2 = np.random.randn(1, 3)
        b2 = np.random.randn(1, 1)
        Z2 = np.random.randn(1, 2)
        linear_cache_activation_2 = ((A2, W2, b2), Z2)

        self.caches = (linear_cache_activation_1, linear_cache_activation_2)

    def test_func(self):
        gradients = backward_propagation(self.AL, self.Y, self.caches)

        assert (set(self.expected_gradients.keys()).issubset(gradients.keys()))
        np.testing.assert_array_almost_equal(gradients['dA1'], self.expected_gradients['dA1'], decimal=8)
        np.testing.assert_array_almost_equal(gradients['dW1'], self.expected_gradients['dW1'], decimal=8)
        np.testing.assert_array_almost_equal(gradients['db1'], self.expected_gradients['db1'], decimal=8)


if __name__ == '__main__':
    unittest.main()
