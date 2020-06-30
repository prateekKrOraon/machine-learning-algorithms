"""
Author: Prateek Kumar Oraon
Copyright Prateek Kumar Oraon, free to use under MIT License
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class KNN(object):
    def __init__(self, points, epochs):
        self.points = points
        self.epochs = epochs
        self.centroids = None
        self.labels = None

        self.create_centroids()

    def create_centroids(self):
        centroids = [
            [5.0, 0.0],
            [35.0, 55.0],
            [25.0, 40.0],
        ]

        self.centroids = centroids

    def compute_distance(self, x, y):
        return np.sqrt((y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2)

    def compute_new_centroid(self, centroids, index, point):
        x = [
            (centroids[index][0] + point[0]) / 2,
            (centroids[index][1] + point[1]) / 2
        ]
        return x

    def get_cluster_centroid(self, distances):
        return min(distances, key=distances.get)

    def fit(self):
        points_len = len(self.points)
        cluster_labels = [0] * points_len
        centroids_len = len(self.centroids)

        for i in range(self.epochs):
            for point_index in range(points_len):
                distances = {}
                for centroid_index in range(centroids_len):
                    distances[centroid_index] = self.compute_distance(self.centroids[centroid_index],
                                                                      self.points[point_index])

                index = self.get_cluster_centroid(distances)
                self.centroids[index] = self.compute_new_centroid(self.centroids, index, self.points[point_index])
                cluster_labels[point_index] = index

        self.labels = cluster_labels

    def plot_result(self):
        for i in range(len(self.points)):
            plt.scatter(self.points[i][0], self.points[i][1], color='blue')
        for i in range(len(self.centroids)):
            plt.scatter(self.centroids[i][0], self.centroids[i][1], marker='x', color='red')
        plt.show()


def run():
    dataset = pd.read_csv('knn.csv')
    print(dataset.head())

    points = dataset.values

    model = KNN(points, 10)
    model.fit()

    model.plot_result()


if __name__ == '__main__':
    run()
