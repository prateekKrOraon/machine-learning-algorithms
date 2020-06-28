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


    def entropy(freq_y, freq_n, total):
        if freq_n is 0 and freq_y is 0:
            return 0
        elif freq_n is 0:
            return -(freq_y / total) * np.log2(freq_y / total)
        elif freq_y is 0:
            return -(freq_n / total) * np.log2(freq_n / total)
        else:
            return -(freq_y / total) * np.log2(freq_y / total) - (freq_n / total) * np.log2(freq_n / total)

        def entropy_set():
            cols = dataset.keys()
            length = len(cols)
            freq_n = 0
            freq_y = 0
            total = 0
            for rows, columns in dataset.iterrows():
                total += 1
                if columns[cols[length - 1]] == 'no':
                    freq_n += 1
                elif columns[cols[length - 1]] == 'yes':
                    freq_y += 1

            return entropy(freq_y, freq_n, total)

        def entropy_attr(attr):
            cols = dataset.keys()
            length = len(cols)
            vals = set()
            entr = 0
            for rows, columns in dataset.iterrows():
                vals.add(columns[attr])

            for val in vals:
                freq_y = 0
                freq_n = 0
                total = 0
                set_total = 0
                for rows, columns in dataset.iterrows():
                    set_total += 1
                    if columns[attr] == val and columns[cols[len(cols) - 1]] == 'yes':
                        freq_y += 1
                        total += 1
                    elif columns[attr] == val and columns[cols[len(cols) - 1]] == 'no':
                        freq_n += 1
                        total += 1

                entr += (total / set_total) * entropy(freq_y, freq_n, freq_y + freq_n)

            return entr

        def info_gain(attr):
            return entropy_set() - entropy_attr(attr)

        class Node(object):
            def __init__(self, children=[]):
                self.children = children
                self.order = None
                self.label = None

        def run_decision_tree():
            cols = dataset.keys()
            for rows, columns in dataset.iterrows():

