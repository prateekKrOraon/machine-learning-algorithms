"""
Author: Prateek Kumar Oraon
Copyright Prateek Kumar Oraon, free to use under MIT License
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


class ExpectationMaximization(object):
    def __init__(self, x_list, steps, clusters):
        self.x_list = x_list
        self.steps = steps
        self.clusters = clusters
        self.labels = []
        self.means = []
        self.std_dev = []

    def get_mean(self, x, y):
        return (x + y) / 2

    def standard_dev(self, x_list, mean):
        return np.sqrt(np.sum((x_list - mean) ** 2) / len(x_list))

    def initialize_means(self, x_list, clusters):
        mean_list = []
        for i in range(clusters):
            num_list = []
            for j in range(5):
                n = random.randint(0, 500)
                num_list.append(x_list[n])
            mean_list.append(np.sum(num_list) / len(num_list))

        return mean_list

    def expectation(self, x_list, mean_list, std_dev):
        labels = []
        probs = {}

        for x in x_list:
            for i in range(len(std_dev)):
                p = (1 / np.sqrt(2 * np.pi * (std_dev[i] ** 2))) * np.exp(
                    -((x - mean_list[i]) ** 2) / 2 * (std_dev[i] ** 2))
                probs[i] = p
            max_p = max(probs, key=probs.get)
            labels.append(max_p)
            mean_list[max_p] = self.get_mean(mean_list[max_p], x)

        return [labels, mean_list]

    def maximization(self, x_list, mean_list):
        std_dev = []
        for mean in mean_list:
            std_dev.append(self.standard_dev(x_list, mean))

        return std_dev

    def fit(self):
        x_list = self.x_list
        steps = self.steps

        means = self.initialize_means(x_list, self.clusters)

        colors = ['red', 'green', 'blue', 'cyan', 'pink']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x_list, np.zeros_like(x_list) + 0, color='black', label='points')
        for i in range(self.clusters):
            ax.scatter(means[i], [0], marker='x', color=colors[i%len(colors)], label='centroid {}'.format(i + 1))

        plt.title('Initial Centroids')
        plt.legend()
        plt.show()

        labels = []
        std_dev = None
        for i in range(steps):
            std_dev = self.maximization(x_list, means)
            labels, means = self.expectation(x_list, means, std_dev)

        self.labels = labels
        self.means = means
        self.std_dev = std_dev


def plot_results(x_list, means, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = ['red', 'green', 'blue', 'cyan', 'brown']
    for i in range(len(x_list)):
        ax.scatter(x_list[i], 0, color=colors[labels[i] % len(colors)])

    for i in range(len(means)):
        ax.scatter(means[i], [1], marker='x', color=colors[i % len(colors)], label='centroid {}'.format(i + 1))

    ax.scatter(0, 10, color='white')
    ax.scatter(0, -10, color='white')
    plt.title("Final Centroids")
    plt.legend()
    plt.show()


def run():
    dataset = pd.read_csv('em_sample.csv')
    dataset.head()

    x_list = dataset.iloc[:, 0].values

    model = ExpectationMaximization(x_list, steps=100, clusters=4)
    model.fit()

    plot_results(x_list, model.means, model.labels)


if __name__ == '__main__':
    run()
