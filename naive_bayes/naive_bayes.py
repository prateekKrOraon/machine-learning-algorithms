"""
Author: Prateek Kumar Oraon
Copyright Prateek Kumar Oraon, free to use under MIT License
"""

import pandas as pd


class NaiveBayes(object):
    def __init__(self, dataset, target):
        self.dataset = dataset
        self.target = target
        self.probs = {}
        self.accuracy = None

    def calculate_probability(self, val, attr):
        freq_y = 0
        freq_n = 0
        total = 0
        for rows, cols in self.dataset.iterrows():
            if str(cols[attr]) == val and str(cols[self.target]) == 'Yes':
                freq_y += 1
            elif str(cols[attr]) == val and cols[self.target] == 'No':
                freq_n += 1
            total += 1

        p_y = freq_y / total
        p_n = freq_n / total

        if p_y == 0 and p_n == 0:
            return [0, 0]

        return [p_y / (p_y + p_n), p_n / (p_y + p_n)]

    def predict(self, x):
        prob_y = 1
        prob_n = 1
        for item in x:
            prob_y *= self.probs[str(item)][0]
            prob_n *= self.probs[str(item)][1]

        prob_y = prob_y * self.probs[self.target][0]
        prob_n = prob_n * self.probs[self.target][1]

        final_prob_y = prob_y / (prob_y + prob_n)
        final_prob_n = prob_n / (prob_y + prob_n)

        y_pred = {'Yes': final_prob_y, 'No': final_prob_n}
        return y_pred

    def fit(self):
        keys = self.dataset.keys()
        keys.delete(-1)

        t_yes = 0
        t_no = 0
        total = 0

        for row, cols in self.dataset.items():
            if row != self.target:
                for item in cols:
                    if item not in self.probs.keys():
                        p = self.calculate_probability(str(item), row)
                        self.probs[str(item)] = p
            else:
                for item in cols:
                    total += 1
                    if item == 'Yes':
                        t_yes += 1
                    else:
                        t_no += 1

        p_yes = t_yes / total
        p_no = t_no / total

        self.probs[self.target] = [p_yes, p_no]

        self.calculate_accuracy()

    def calculate_accuracy(self):
        correct = 0
        incorrect = 0
        for row, cols in self.dataset.iterrows():
            n = list(cols)
            y_pred = self.predict(n[:-1])
            pred = max(y_pred, key=y_pred.get)
            if pred == n[-1]:
                correct += 1
            else:
                incorrect += 1

        acc = (correct / len(self.dataset)) * 100
        self.accuracy = acc


def run():
    dataset = pd.read_csv('weather.csv')
    print(dataset.head())

    model = NaiveBayes(dataset, target='Play Golf')
    model.fit()

    print("\nProbabilities")
    print("-----------------------------------------------")
    print(model.probs)
    print("\n")

    print("Accuracy")
    print("-----------------------------------------------")
    print("Accuracy of model = {}".format(model.accuracy))


if __name__ == '__main__':
    run()
