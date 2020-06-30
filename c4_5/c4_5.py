"""
Author: Prateek Kumar Oraon
Copyright Prateek Kumar Oraon, free to use under MIT License
"""

import numpy as np
import pandas as pd
from pprint import pprint


class C45(object):
    def __init__(self, dataset, features, target_attribute_name='play', parent_node_class=None):
        self.dataset = dataset
        self.features = features
        self.target_attribute_name = target_attribute_name
        self.parent_node_class = parent_node_class
        self.tree = None

    def entropy(self, target_col):
        elements, counts = np.unique(target_col, return_counts=True)
        entropy = np.sum(
            [(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
        return entropy

    def info_gain(self, data, split_attribute_name, target_name="play"):
        total_entropy = self.entropy(data[target_name])
        vals, counts = np.unique(data[split_attribute_name], return_counts=True)
        weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * self.entropy(
            data.where(data[split_attribute_name] == vals[i]).dropna()[target_name]) for i in range(len(vals))])
        information_gain = (total_entropy - weighted_entropy)

        return information_gain

    def C4(self, data, features, target_attribute_name="play", parent_node_class=None):
        if len(np.unique(data[target_attribute_name])) <= 1:
            return np.unique(data[target_attribute_name])[0]
        elif len(data) == 0:
            return
        elif len(features) == 0:
            return parent_node_class
        else:
            parent_node_class = np.unique(data[target_attribute_name])[
                np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
            item_values = [self.info_gain(data, feature, target_attribute_name) for feature in features]
            best_feature_index = np.argmax(item_values)
            best_feature = features[best_feature_index]
            tree = {best_feature: {}}
            features = [i for i in features if i != best_feature]

            for value in np.unique(data[best_feature]):
                value = value
                if value != 'outlook':
                    sub_data = data.where(data[best_feature] == value).dropna()
                    subtree = self.C4(sub_data, features, target_attribute_name, parent_node_class)
                    tree[best_feature][value] = subtree

            return (tree)

    def fit(self):
        self.tree = self.C4(self.dataset, self.features)


def run():
    dataset = pd.read_csv('weather.csv')
    model = C45(dataset, dataset.columns[:-1])
    model.fit()

    pprint(model.tree)


if __name__ == '__main__':
    run()
