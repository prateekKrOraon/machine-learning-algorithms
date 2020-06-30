"""
Author: Prateek Kumar Oraon
Copyright Prateek Kumar Oraon, free to use under MIT License
"""

import pandas as pd


def find_s(dataset):
    cols = dataset.keys()
    s = ['0']*(len(cols)-1)
    for rows, columns in dataset.iterrows():
        if columns[cols[len(cols)-1]] is 'Y':
            for i in range(len(cols)-1):
                if s[i] is '0':
                    s[i] = columns[cols[i]]
                elif s[i] is not columns[cols[i]]:
                    s[i] = '?'

    return s


def main():
    dataset = pd.read_csv('weather.csv')
    s = find_s(dataset)
    print("Hypothesis")
    print(s)


if __name__ == '__main__':
    main()
