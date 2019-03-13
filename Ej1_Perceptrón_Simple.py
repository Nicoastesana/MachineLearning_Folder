import csv
import matplotlib.pyplot as plt
from random import shuffle
import math


###########################
###########################
def mix(File_name):
    data = csv_save(File_name)
    shuffle(data)
    return data


def validacion_cruzada(data, n, n_part, Pcent):
    X, Y, num_x = string_to_float(data)
    Xt = []
    Yt = []

    pos_part = (n_part - 1) * int(len(Y) * (1 - (1 - Pcent)) / (n - 1))
    if n_part == n:
        pos_part = math.ceil(len(Y) - len(Y) * (1 - Pcent))
    for i in range(pos_part, math.ceil(pos_part + int(len(Y) * (1 - Pcent)) + 1)):
        for j in range(0, num_x):
            Xt.append(X.pop(j + pos_part*num_x))
            print(pos_part)
        Yt.append(Y.pop(pos_part))

    return X, Y, Xt, Yt, num_x


def leave_some_out(data, Pcent, particiones, u, lim_acc,W,n):
    for i in range(1, 2):
        X, Y, Xt, Yt, num_x = validacion_cruzada(data, particiones, i + 1, Pcent)
        index, WW, a = multi_entrenamiento(W, X, Y, num_x, u, n, lim_acc, X, Y)

    acc=[]
    for i in range(0,index):
        Waux=[WW[i*len(W)],WW[i*len(W)+1],WW[i*len(W)+2],WW[i*len(W)+3],]
        acc.append(evaluar(Waux, Xt, Yt, num_x))
    graph(index, a)
    graph(index, acc)
    return



###########################
###########################


###FUNCIONES###
# Convierto array de strings X2,X1,Yd a arrays de flotantes
def string_to_float(data):
    num_x = len(data[0]) - 1
    X = []
    Yd = []
    for linea in data:
        for i in range(0, num_x):
            X.append(float(linea[i]))
        Yd.append(float(linea[i + 1]))
    return X, Yd, num_x


# Obtengo los datos en un array de strings X2,X1,Yd
def csv_save(file_name_training):
    with open("Excel/" + file_name_training) as File:
        datos = csv.reader(File, delimiter=',')
        w = []
        for row in datos:
            w.append(row)
    return (w)


# redondeo
def redondeo(num):
    if num >= 0:
        num = 1
    else:
        num = -1
    return num


# Funcion de entrenamiento para 1 epoca
def entrenamiento(W, X, Yd, num_x, u):
    for i in range(0, len(Yd)):
        Y = -W[0]
        for j in range(0, num_x):
            Y = W[j + 1] * X[j + i * num_x] + Y
        Y = redondeo(Y)  # funcion signo

        if Y != Yd[i]:
            W[0] = W[0] + u * (Yd[i] - Y) * (-1)
            for j in range(0, num_x):
                W[j + 1] = W[j + 1] + u * (Yd[i] - Y) * X[j + i * num_x]
    return W


# Funcion Evaluar: Obtiene medidas de desempeño
def evaluar(W, Xt, Ydt, num_xt):
    Tp = Tn = Fp = Fn = 0
    for i in range(0, len(Ydt)):
        Y = -W[0]
        for j in range(0, num_xt):
            Y = W[j + 1] * Xt[j + i * num_xt] + Y
        Y = redondeo(Y)  # funcion signo

        if Y == Ydt[i]:
            if Y >= 0:
                Tp += 1
            else:
                Tn = Tn + 1
        else:
            if Y >= 0:
                Fp = Fp + 1
            else:
                Fn = Fn + 1

    Exactitud = (Tp + Tn) / len(Ydt)
    # F1_Score = 2 * Tp / (2 * Tp + Fp + Fn)  # Sensibilidad de los +
    print('True +:', Tp, 'True -:', Tn, 'False +:', Fp, 'False -:', Fn)
    # print('Exactitud :', Exactitud, 'F1_Score', F1_Score)
    return Exactitud


def multi_entrenamiento(W, X, Yd, num_x, u, n_lim_trainings, lim_accuaracy, Xt, Ydt):
    a = []
    WW = []
    i = 0
    for i in range(0, n_lim_trainings):
        W = entrenamiento(W, X, Yd, num_x, u)
        for j in range(0, len(W)):
            WW.append(W[j])
        a.append(evaluar(W, Xt, Ydt, num_x))
        # if a[i] > lim_acc:
        # break

    return i + 1, WW, a


def graph(index, accuaracy):

    epochs = []
    for i in range(1, index + 1):
        epochs.append(i)
    plt.plot(epochs, accuaracy, label='Accuaracy')
    plt.show()


###VARIABLES###
# index = 0
# W = [1.0, 1.0, 1.0, 1.0]  # [W0...Wn]
# u = 0.01  # velocidad de aprendizaje
# n = 5  # n° límite de entrenamientos
# lim_acc = 0.9

# particiones = 10
# Pcent_train = 0.5
# n_part = 5
###OPERACIONES###
# X, Yd, num_x = string_to_float(csv_save("spheres2d10.csv"))  # cargo datos de entrenamiento
# Xt, Ydt, num_xt = string_to_float(csv_save("spheres2d10.csv"))  # cargo datos de entrenamiento
# index, WW, a = multi_entrenamiento(W, X, Yd, num_x, u, n, lim_acc, Xt, Ydt)
# graph(index, a)

# data = mix("spheres2d10.csv")
# X, Yd, Xt, Ydt, num_x = validacion_cruzada(data, particiones, n_part, Pcent_train)

# leave_k_out("spheres2d70.csv",n, Pcent_train,u,lim_acc)

##EJ1##
# index = 0
# W = [1,-1,1]  # [W0...Wn]
# u = 5 # velocidad de aprendizaje
n = 50  # n° límite de entrenamientos
lim_acc = 0.99

# X, Yd, num_x = string_to_float(csv_save("XORRRR.csv"))  # cargo datos de entrenamiento
# Xt, Ydt, num_xt = string_to_float(csv_save("XORRRR.csv"))  # cargo datos de entrenamiento
# index, WW, a = multi_entrenamiento(W, X, Yd, num_x, u, n, lim_acc, Xt, Ydt)
# graph(index, a)

##EJ2##
particiones = 10
Pcent_train = 0.8  # la cantidad de datos que se entrenan por corrida
W2 = [0.5, -.25, .69, 1.0]  # [W0...Wn]
# n_part = 9
u = 0.0001 # velocidad de aprendizaje
index = 0

data = mix("spheres2d10.csv")
#X, Yd, Xt, Ydt, num_x = validacion_cruzada(data, particiones, n_part, Pcent_train)
leave_some_out(data, Pcent_train, particiones, u, lim_acc,W2,n)
