import numpy as np
import random
import csv
import matplotlib.pyplot as plt


def take_data(database):

    with open(database, newline='') as f:
        reader = csv.reader(f)
        data = [tuple(row) for row in reader]

        x = []
        y = []
        for reg in data:
             reg = tuple(map(float, reg))
             x.append(reg[:-1])
             y.append(reg[-1])
             
    return x, y


def dot_product(t1, t2):
        if len(t1) == len(t2):
            return sum([t1[i]*t2[i] for i in range(len(t1))])
        else:
            raise ValueError(f"Doct Product: {t1} and {t2} do not have the same length.")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def apply_CLog_MGB(w0, eta):
    x, y = take_data(database)
    t = 0
    N = len(y)
    w = w0
    aux = []


    while t < 500:
        p = [sigmoid(dot_product(w, (1.0,) + xn)) for xn in x]
        for i in range(N):
            aux.append(tuple((p[i] - y[i]) * comp for comp in ((1.0,) + x[i])))
        sums = tuple(sum(tuplos) for tuplos in zip(*aux))
        s = tuple((1/N) *  comp for comp in sums)
        w = tuple(val1 - val2 for val1, val2 in zip(w, tuple(eta * comp for comp in s))) #there
        t+=1

    return w


def plot():
    x, y = take_data(database)


    for i in range(len(x)):
        if y[i] == 0:
            plt.scatter(x[i][0], x[i][1], color='blue')
        else:
            plt.scatter(x[i][0], x[i][1], color='red')


    x_values = [0, 1]
    y_values = [-(w[0] + w[1]*x)/w[2] for x in x_values]
    plt.plot(x_values, y_values, color='green')


    plt.xlabel('X')
    plt.ylabel('Y')

    plt.show()


database = "databases/ex5_D.csv"
w = apply_CLog_MGB((0.0, 0.0, 0.0), 0.5)
print(w)
plot()

