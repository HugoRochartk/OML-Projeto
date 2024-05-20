import numpy as np
import random
import csv
import matplotlib.pyplot as plt
from math import log

def error(ypred, ytrue):

    aux = []
    N = len(ypred)
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


def build_dp_matrix(x, N):

    matrix = []

    for i in range(N):
        line = []
        for j in range(N):
            line.append(dot_product((1.0,) + x[i], (1.0,) + x[j]))
        matrix.append(line)

    return matrix



def apply_CLogD_MGE(eta, d, error_graph=True):
    x, y = take_data(database)
    t = 0
    N = len(y)
    alpha = tuple(0 for i in range(N))
    error_vals = []
    dp_matrix = build_dp_matrix(x, N)

    while t < 500:

        p_for_error = []
        for i in range(N):
            p_for_error.append(sigmoid(sum([alpha[l]*dp_matrix[l][i] for l in range(N)])))
        
        n = random.randint(0, N-1)
        p = sigmoid(sum([alpha[l]*dp_matrix[l][n] for l in range(N)]))
          
     
        aux = []
        for j in range(N):
            for i in range(d):
                aux.append((dp_matrix[j][n]))
        s = tuple((p - y[n]) * comp for comp in aux)
 

        alpha = tuple(val1 - val2 for val1, val2 in zip(alpha, tuple(eta * comp for comp in s)))

        if error_graph:
            error_vals.append(error(p_for_error, y))

        t+=1
    



    if error_graph:
        plot_error_graph(error_vals, t)


    return alpha


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




database = "databases/ex6_D.csv"
w = apply_CLogD_MGE(0.5, 3)
print(w)
plot()