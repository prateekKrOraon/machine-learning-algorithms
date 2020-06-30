"""
Author: Prateek Kumar Oraon
Copyright Prateek Kumar Oraon, free to use under MIT License
"""

import pandas as pd


def get_version_space(s, g):
    vs = []
    for i in range(len(s)):
        if s[i] is not g[i]:
            for j in range(len(g)):
                if g[j] is not '?':
                    a = ['?'] * len(s)
                    a[i] = s[i]
                    a[j] = g[j]
                    vs.append(a)

    vs.append(s)
    vs.append(g)

    return vs


def candidate_elimination(dataset):
    cols = dataset.keys()
    s = ['0'] * (len(cols) - 1)
    g = ['?'] * (len(cols) - 1)
    for row, column in dataset.iterrows():
        if column[cols[len(cols) - 1]] == 'Y':
            for i in range(len(cols) - 1):
                if s[i] == '0':
                    s[i] = column[cols[i]]
                elif s[i] is not column[cols[i]]:
                    if g[i] is s[i]:
                        g[i] = '?'
                    s[i] = '?'
        else:
            for i in range(len(cols) - 1):
                if s[i] is not column[cols[i]]:
                    g[i] = s[i]

    vs = get_version_space(s, g)
    return [s, g, vs]


def main():
    dataset = pd.read_csv('wsce.csv')
    s, g, vs = candidate_elimination(dataset)
    print('Most specific hypothesis')
    print(s)
    print('Most general hypothesis')
    print(g)
    print('Version Space')
    for hypo in vs:
        print(hypo)


if __name__ == '__main__':
    main()