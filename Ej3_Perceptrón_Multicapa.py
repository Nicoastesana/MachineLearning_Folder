import csv
from numpy import array
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


def csv_save(file_name_training):
    with open(
            "C:/Users/095/Desktop/Machine Learning/Machine_Learning_Folder/Práctica/Clase_2/Ej3/" + file_name_training) as File:
        datos = csv.reader(File, delimiter=',')
        w = []
        for row in datos:
            w.append(row)
    return (w)


def string_to_float(data):
    num_x = len(data[0]) - 1
    X = []
    Y = []
    for linea in data:
        Xaux = []
        for i in range(0, num_x):
            Xaux.append(float(linea[i]))
        X.append(Xaux)
        Y.append(float(linea[i + 1]))
    X_numpy = array(X)  # creo una matriz numpy
    Y_numpy = array(Y)
    W = vector_de_pesos(len(X[0]), neuronas_x_capa)  # crea un vector de pesos inicializado al azar
    return X_numpy, Y_numpy, W


def vector_de_pesos(len_x, neuronas_x_capa):
    W = []
    W.append(np.random.uniform(-1, 1, [len_x + 1, neuronas_x_capa[0]]))
    for i in range(0, len(neuronas_x_capa) - 1):
        W.append(np.random.uniform(-1, 1, [neuronas_x_capa[i] + 1, neuronas_x_capa[i + 1]]))
    return W


def multiply_numpy(A, B):  # agrega el sesgo al multiplicar
    A = np.append(-1, A)
    C = np.matmul(A, B)
    return C


def multiply_not_byass(A, B):  # multiplicacion comun
    C = np.matmul(A, B)
    return C


def activation_function(A, b):
    for i in range(0, len(A)):
        A[i] = 2 / (1 + (2.71 ** (-b * A[i]))) - 1
    return A


def y_result(a):
    if a >= 0:
        return 1
    else:
        return -1


def zeros(A):
    for i in range(0, len(A)):
        for j in range(0, len(A[i])):
            A[i][j] = 0.0
    return A


def copy_W_wo_W0(A):
    for i in range(0, len(A)):
        A[i] = np.delete(A[i], 0, 0)

    return A


def sum_list_of_array(A, B):
    C = []
    for i in range(0, len(A)):
        C.append(A[i] + B[i])
    return C


def train_1_epoch(X, Y, W, u, b):
    for i in range(0, len(X)):
        propagacion = []
        propagacion.append(activation_function(multiply_numpy(X[i], W[0]), b))
        for j in range(1, len(W)):
            propagacion.append(activation_function(multiply_numpy(propagacion[j - 1], W[j]), b))
        y_calculated = propagacion[len(W) - 1]

        ####RETROPROPAGACION###
        error_prom = zeros(deepcopy(propagacion))
        error_prom[len(error_prom) - 1] = (
                (Y[i] - y_calculated) * 0.5 * (1 + propagacion[len(W) - 1]) * (1 - propagacion[len(W) - 1]))
        W_wo_W0 = copy_W_wo_W0(deepcopy(W))

        for j in range(len(W_wo_W0) - 2, -1, -1):
            for k in range(0, len(W_wo_W0[j + 1])):
                error_prom[j][k] = np.sum((multiply_not_byass(W_wo_W0[j + 1][k], error_prom[j + 1]))) * 0.5 * (
                        1 + propagacion[j][k]) * (1 - propagacion[j][k])

        ####DELTA PESOS###
        delta_W = zeros(deepcopy(W))
        delta_W[0][0] = u * error_prom[0] * (-1)
        for j in range(1, len(X[0]) + 1):
            delta_W[0][j] = u * error_prom[0] * X[i][j - 1]

        for k in range(1, len(W)):
            delta_W[k][0] = u * error_prom[k] * (-1)
            for j in range(0, len(propagacion[k - 1])):
                delta_W[k][j + 1] = u * error_prom[k] * propagacion[k - 1][j]

        ####CORRRIJO PESOS###
        W = sum_list_of_array(W, delta_W)

    return W


def evaluar(W, X, Yd, b):
    Tp = Tn = Fp = Fn = 0

    for i in range(0, len(X)):
        propagacion = []
        propagacion.append(activation_function(multiply_numpy(X[i], W[0]), b))
        for j in range(1, len(W)):
            propagacion.append(activation_function(multiply_numpy(propagacion[j - 1], W[j]), b))
        y_calculated = y_result(propagacion[len(W) - 1])

        if y_calculated == Yd[i]:
            if y_calculated >= 0:
                Tp += 1
                plt.plot(X[i][0], X[i][1], 'ro')

            else:
                Tn = Tn + 1
                plt.plot(X[i][0], X[i][1], 'ro')
        else:
            if y_calculated >= 0:
                Fp = Fp + 1
                plt.plot(X[i][0], X[i][1], 'ob')
            else:
                Fn = Fn + 1
                plt.plot(X[i][0], X[i][1], 'ob')

    Exactitud = (Tp + Tn) / len(Yd)
    # F1_Score = 2 * Tp / (2 * Tp + Fp + Fn)  # Sensibilidad de los +
    print('True +:', Tp, 'True -:', Tn, 'False +:', Fp, 'False -:', Fn)
    # print('Exactitud :', Exactitud, 'F1_Score', F1_Score)
    plt.show()
    return Exactitud


# MAIN#
# Inputs#
neuronas_x_capa = [2, 1]  # numero de neuronas en cada capa
u = 0.05  # velocidad de aprendizaje
b = 1  # parámetro de la funcion de activacion

# Entrenamiento#
X, Y, W = string_to_float(
    csv_save("concentlite.csv"))  # cargo datos de entrenamiento y creo lista de matrices de pesos
for i in range(0, 2000):
    W = train_1_epoch(X, Y, W, u, b)
    print("train number :", i)
    if i % 99 == 0:
        print(W)
        evaluar(W, X, Y, b)
