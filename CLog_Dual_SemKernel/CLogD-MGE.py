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


def build_dp_matrix(x, N):

    matrix = []

    for i in range(N):
        line = []
        for j in range(N):
            line.append(dot_product((1.0,) + x[i], (1.0,) + x[j]))
        matrix.append(line)

    return matrix

def get_accuracy(y_pred, y_true):
    c = 0
    N = len(y_pred)

    for i in range(N):
        if (y_pred[i] > 0.5 and y_true[i] == 1) or (y_pred[i] <= 0.5 and y_true[i] == 0):
            c+=1

    return c/N

def apply_CLogD_MGE(eta, error_graph=True, accuracy=True):
    x, y = take_data(database)
    t = 0
    N = len(y)
    alpha = tuple(0 for i in range(N))
    error_vals = []
    dp_matrix = build_dp_matrix(x, N)
    p_for_error = []

    while t < 2000 and error(p_for_error, y) > 0.025:

        p_for_error = []
        for i in range(N):
            p_for_error.append(sigmoid(sum([alpha[l]*dp_matrix[l][i] for l in range(N)])))
        
        n = random.randint(0, N-1)
        p = sigmoid(sum([alpha[l]*dp_matrix[l][n] for l in range(N)]))
          
     
        aux = []
        for j in range(N):
            aux.append(dp_matrix[j][n])
        s = tuple((p - y[n]) * comp for comp in aux)
 

        alpha = tuple(val1 - val2 for val1, val2 in zip(alpha, tuple(eta * comp for comp in s)))

        if error_graph:
            error_vals.append(error(p_for_error, y))

        t+=1
    

    w_to_sum = []
    for i in range(N):
        w_to_sum.append(tuple(alpha[i] * comp for comp in ((1.0,) + x[i])))
    w = tuple(map(sum, zip(*w_to_sum)))

    if accuracy:
        print(f"Accuracy: {get_accuracy(p_for_error, y)}")

    if error_graph:
        plot_error_graph(error_vals, t)


    return w, alpha, x


def plot_decision_boundary(alpha, x):
    x_min, x_max = min([xi[0] for xi in x]) - 1, max([xi[0] for xi in x]) + 1
    y_min, y_max = min([xi[1] for xi in x]) - 1, max([xi[1] for xi in x]) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    dp_matrix_grid = np.array([[(1 + dot_product(xi, xj)) for xj in x] for xi in grid])
    decision_values = np.dot(dp_matrix_grid, alpha)
    decision_values = decision_values.reshape(xx.shape)
    
    plt.contourf(xx, yy, decision_values, levels=[-float('inf'), 0, float('inf')], colors=['blue', 'red'], alpha=0.3)
    plt.contour(xx, yy, decision_values, levels=[0], colors='blue')
    plot_data_points()
    

def plot_data_points():
    x_data, y_data = take_data(database)
    for i in range(len(x_data)):
        color = 'blue' if y_data[i] == 0 else 'red'
        plt.scatter(x_data[i][0], x_data[i][1], color=color)
    plt.show()




database = "databases/ex6_D.csv"
w, alpha, x = apply_CLogD_MGE(0.5)
print(f"w = {w}")

plot_decision_boundary(alpha, x)