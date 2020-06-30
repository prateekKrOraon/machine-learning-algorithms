"""
Author: Prateek Kumar Oraon
Copyright Prateek Kumar Oraon, free to use under MIT License
"""

import pandas as pd
import matplotlib.pyplot as plt
import random


class KMeans(object):
    def __init__(self, x_list, clusters):
        self.x_list = x_list
        self.clusters = clusters
        self.labels = None
        self.centroids = None
        self.initialize_centroids(x_list, clusters)
        print("Initial Centroids")
        print(self.centroids)

    def get_mean(self, x, y):
        return (x + y) / 2

    def initialize_centroids(self, x_list, clusters):
        max_el = max(x_list)
        min_el = min(x_list)

        centroid_list = []
        i = 0
        while i < clusters:
            n = random.uniform(min_el, max_el)
            if n not in centroid_list:
                centroid_list.append(n)
                i += 1
            else:
                i -= 1

        self.centroids = centroid_list

    def get_centroid(self, distances):
        return min(distances, key=distances.get)

    def fit(self):
        distances = {}
        self.labels = [0] * len(self.x_list)
        changed = True
        k = 0
        while changed:
            changed = False
            k += 1

            for i in range(len(self.x_list)):
                for j in range(len(self.centroids)):
                    distance = self.centroids[j] - self.x_list[i]
                    distance = abs(distance)
                    distances[j] = distance

                centroid = self.get_centroid(distances)

                if centroid != self.labels[i]:
                    changed = True
                    self.labels[i] = centroid

                self.centroids[centroid] = self.get_mean(self.centroids[centroid], self.x_list[i])

        print('Iterations = ', k)
        print('Final Centroids')
        print(self.centroids)


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
    # plt.legend()
    plt.show()


def run():
    dataset = pd.read_csv('em_sample.csv')

    x_list = dataset.iloc[:, 0].values

    model = KMeans(x_list, 3)
    model.fit()

    plot_results(x_list, model.centroids, model.labels)


if __name__ == '__main__':
    run()
