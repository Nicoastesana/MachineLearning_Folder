import csv
from numpy import array
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


def csv_save(file_name_training):
    with open(
            "C:/Users/095/Desktop/Machine Learning/Machine_Learning_Folder/Práctica/Clase_2/Ej4/" + file_name_training) as File:
        datos = csv.reader(File, delimiter=',')
        w = []
        for row in datos:
            w.append(row)
    return (w)


def string_to_float(data, num_x, num_y):
    X = []
    Y = []
    for linea in data:
        Xaux = []
        Yaux = []
        for i in range(0, num_x):
            Xaux.append(float(linea[i]))
        X.append(Xaux)
        for i in range(num_x, num_y + num_x):
            Yaux.append(float(linea[i]))
        Y.append(Yaux)
    X = delete_columns(X, borrar_col)
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


def y_result(A):
    for i in range(len(A)):
        if A[i] >= 0:
            A[i] = 1
        else:
            A[i] = -1
    return A


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
        y_calculated = propagacion[len(W) - 1]  # son las salidas de la red nuronal luego de la propagacion

        ####RETROPROPAGACION###
        error_prom = zeros(deepcopy(propagacion))
        for j in range(0, len(y_calculated)):
            error_prom[len(error_prom) - 1][j] = (
                    (Y[i][j] - y_calculated[j]) * 0.5 * (1 + propagacion[len(W) - 1][j]) * (
                    1 - propagacion[len(W) - 1][j]))  # Son los Deltas
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


def multi_train(X, Y, W, u, b, n_trainnings):
    for i in range(n_trainnings):
        W = train_1_epoch(X, Y, W, u, b)
    return W


def evaluar(W, X, Yd, b, dimension_one):  # dimension_one =1 => entra un vector, =0=>entra una matriz
    well_classified = bad_classified = 0
    if dimension_one == 1:
        fin = 1
    else:
        fin = len(X)
    for i in range(0, fin):
        propagacion = []

        if dimension_one == 1:
            propagacion.append(activation_function(multiply_numpy(X, W[0]), b))
        else:
            propagacion.append(activation_function(multiply_numpy(X[i], W[0]), b))
        for j in range(1, len(W)):
            propagacion.append(activation_function(multiply_numpy(propagacion[j - 1], W[j]), b))
        y_calculated = y_result(propagacion[len(W) - 1])

        if fin == 1:
            if np.array_equal(y_calculated, Yd):
                well_classified += 1
            #               plt.plot(X[i][0], X[i][1], 'ro')

            else:
                bad_classified += 1
        #               plt.plot(X[i][0], X[i][1], 'ob')
        else:
            if np.array_equal(y_calculated, Yd):
                well_classified += 1
            #               plt.plot(X[i][0], X[i][1], 'ro')

            else:
                bad_classified += 1
        #               plt.plot(X[i][0], X[i][1], 'ob')

    Exactitud = well_classified / (well_classified + bad_classified)
    # print('well_classified: ', well_classified, 'bad_classified', bad_classified)
    # plt.show()
    return Exactitud


def delete_columns(A, B):
    B.reverse()
    for i in range(len(A)):
        for j in B:
            A[i].pop(j)
    return A


def quit_row(A, n):
    B = np.delete(A, n, 0)
    return B


def leave_one_out(X, Y, W, Pcen_acc, n_trainnings, u, b):
    num_x = len(X[0])
    num_y = len(Y[0])
    for i in range(0, n_trainnings):
        hits = 0
        aux_x = []
        aux_y = []
        X_for_test = X[i]
        Y_for_test = Y[i]
        X_less_one = quit_row(X, i)
        Y_less_one = quit_row(Y, i)
        Waux = multi_train(X_less_one, Y_less_one, W, u, b, n_trainnings)
        hits += evaluar(Waux, X_for_test, Y_for_test, b, 1)
        print("train number :", i, "hits", hits)
    print(hits / len(X))

    return W


# MAIN#
# Inputs#
entradas = 4  # Cantidad de entradas "X" que tiene el archivo
salidas = 3
borrar_col = [0, 3]  # las columnas de los datos de entrada que no necesito
neuronas_x_capa = [2, 3]  # numero de neuronas en cada capa
u = 0.05  # velocidad de aprendizaje
b = 1  # parámetro de la funcion de activacion
Pcen_acc = 0.95
n_trainnings = 50

# Carga_de_datos#
X, Y, W = string_to_float(csv_save("irisbin.csv"), entradas,
                          salidas)  # cargo datos de entrenamiento y creo lista de matrices de pesos
# Entrenamiento#
leave_one_out(X, Y, W, Pcen_acc, n_trainnings, u, b)
