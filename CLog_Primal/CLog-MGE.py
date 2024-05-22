import numpy as np
import random
import csv
import matplotlib.pyplot as plt
from math import log



def error(ypred, ytrue):

    aux = []
    N = len(ypred)

    if N == 0:
        return 1
    
    delta = 1*(10**(-6))

    for i in range(N):
        if ypred[i] == 1:
            ypred[i] = ypred[i] - delta

        aux.append(-ytrue[i]*log(ypred[i]) - (1-ytrue[i])*log(1-ypred[i]))
    return (1/N)*sum(aux)


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


def plot_error_graph(error_vals, t):
    plt.figure(figsize=(10, 6))
    plt.plot(list(range(1,t+1)), error_vals, marker='o', linestyle='-', color='b')

    plt.title('Erro ao longo das iterações')
    plt.xlabel('Iterações (t)')
    plt.ylabel('Erro')


    plt.grid(True)

    plt.show()


def apply_CLog_MGE(w0, eta, error_graph=True):
    x, y = take_data(database)
    t = 0
    N = len(y)
    w = w0
    error_vals = []
    p_for_error = []

    while t < 2000 and error(p_for_error, y) > 0.025:
        n = random.randint(0, N-1)
        p_for_error = [sigmoid(dot_product(w, (1.0,) + xn)) for xn in x]
        p = p_for_error[n]
        s = tuple((p - y[n]) * comp for comp in ((1.0,) + x[n]))
        w = tuple(val1 - val2 for val1, val2 in zip(w, tuple(eta * comp for comp in s)))
        if error_graph:
            error_vals.append(error(p_for_error, y))
        t+=1
    
    if error_graph:
        plot_error_graph(error_vals, t)

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
w = apply_CLog_MGE((0.0, 0.0, 0.0), 0.5)
print(w)
plot()







