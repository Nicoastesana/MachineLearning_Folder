import csv
from numpy import array
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import random
import srt


def csv_save(file_name_training):
    with open(
            "C:/Users/095/Desktop/Machine Learning/Machine_Learning_Folder/Práctica/Clase_2/Ej5/" + file_name_training) as File:
        datos = csv.reader(File, delimiter=',')
        w = []
        for row in datos:
            w.append(row)
    return (w)


def delete_columns(A, B):
    B.reverse()
    for i in range(len(A)):
        for j in B:
            A[i].pop(j)
    return A


def string_to_float(data):
    num_x = len(data[0])
    X = []
    for linea in data:
        Xaux = []
        for i in range(0, num_x):
            Xaux.append(float(linea[i]))
        X.append(Xaux)
    X = delete_columns(X, borrar_col)
    X_numpy = array(X)  # creo una matriz numpy

    return X_numpy


def vector_de_pesos(len_x, neuronas_x_capa):
    W = []
    W.append(np.random.uniform(-1, 1, [len_x + 1, neuronas_x_capa[0]]))
    for i in range(0, len(neuronas_x_capa) - 1):
        W.append(np.random.uniform(-1, 1, [neuronas_x_capa[i] + 1, neuronas_x_capa[i + 1]]))
    return W


def disorder_index(A):
    index = []
    for i in range(len(A)):
        index.append(i)
    return random.sample(index, len(index))


def make_groups(A, n_k):
    index = disorder_index(A)
    disorder_A = A[index]
    return np.split(disorder_A, n_k)


def centroide(A, size):
    if len(A) == 0:
        aux = []
        for i in range(size):
            aux.append(-9999)
        return aux
    return np.sum(A, axis=0) / len(A)


def distancia(A, C):
    return np.linalg.norm(A - C)


def group_with_less_distance(a, C):
    index = 0
    dist = 99999
    for i in range(len(C)):
        if distancia(a, C[i]) < dist:
            index = i
            dist = distancia(a, C[i])
    return index


def classify(A, C):
    index = []
    for i in range(len(A)):
        index.append(group_with_less_distance(A[i], C))

    return index


def zerolistmaker(n):
    listofzeros = []
    for i in range(n):
        listofzeros.append(array([]))
    return listofzeros


def are_eq(A, B):
    for i in range(len(B)):
        if (np.array_equal(A[i], B[i])) == False:
            return 1
    return 0


def k_means(A, n_k):
    groups = make_groups(A, n_k)  # n_k grupos con elemento clasificados al azar
    C = array([])
    for i in range(n_k):  # Calculo los centroides de los K grupos
        aux = centroide(groups[i], len(A[0]))
        C = np.append(aux, C)
    C = np.split(C, n_k)
    index = classify(A, C)  # grupo al que pertence cada entrada

    new_groups = zerolistmaker(n_k)

    while are_eq(groups, new_groups) == 1:  # mientras los grupos sigan cambiando
        groups = new_groups.copy()
        new_groups = zerolistmaker(n_k)
        for i in range(len(A)):
            new_groups[index[i]] = np.append(A[i], new_groups[index[i]])
        for i in range(len(new_groups)):
            if len(new_groups[i] != 0):
                new_groups[i] = np.split(new_groups[i],
                                         len(new_groups[i]) / len(A[0]))  # lleno los grupos con as entradas

        C = array([])
        for i in range(n_k - 1, -1, -1):  # Calculo los centroides de los K grupos
            aux = centroide(new_groups[i], len(A[0]))
            C = np.append(aux, C)
        C = np.split(C, len(C) / len(aux))
        index = classify(A, C)  # grupo al que pertence cada entrada

    return groups, C


def show_data(A, C, colA, colB):
    market = ['b.', 'gx', 'ro', 'ob', 'm.']
    for i in range(len(A)):
        for j in range(len(A[i])):
            plt.plot(A[i][j][colA], A[i][j][colB], market[i % len(market)])

    for i in range(len(C)):
        if C[i][0] != -9999:
            plt.text(C[i][colA], C[i][colB], 'X', color="black", fontsize=16)
    plt.show()
    return 1


def mean_distance(groups, C):
    gloabal_mean = 0
    counter = 0
    for i in range(len(groups)):
        if C[i][0] != -9999:
            counter += 1
            suma = 0
            for j in range(len(groups[i])):
                suma += distancia(groups[i][j], C[i])
            print(i, suma / len(groups[i]))
            plt.bar(i, suma / len(groups[i]))
            gloabal_mean += suma / len(groups[i])
        else:
            print(i, 'sin elementos')

    plt.show()
    print(gloabal_mean / counter)
    return 1


# MAIN#
# Inputs#
borrar_col = []  # elijo las columnas de datos que no quiero comparar o cargar
K = 10  # tiene que ser divisor de 150 . ej:1,2,3,5,10,15,30...
colA = 0
colB = 1

# Carga_de_datos#
X = string_to_float(csv_save("irisclu.csv"))  # cargo datos de entrenamiento y creo lista de matrices de pesos
groups, C = k_means(X, K)
show_data(groups, C, colA, colB)
mean_distance(groups, C)
