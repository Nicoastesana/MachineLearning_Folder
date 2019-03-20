from numpy import exp, array, random, dot, matmul, reshape, insert
import csv
import matplotlib.pyplot as plt
import Tkinter, tkFileDialog

root = Tkinter.Tk().withdraw()  # Para poder usar el fileSelector


def csv_open():
    file_path = tkFileDialog.askopenfilename()
    with open(file_path) as File:
        datos = csv.reader(File, delimiter=',')
        rows = []
        for row in datos:
            rows.append(row)

    # El largo de la lista de entrada - 1 (El ultimo elemento de la lista es el Yd) es la cantidad de entradas (Xn)
    cant_entradas = len(rows[0]) - 1

    entradas = []  # Matrix con entradas X1, X2, ..., Xn
    salidas_deseadas = []  # Vector de salidas - Y deseadas
    for linea_csv in rows:
        x_aux = []
        for i in range(0, cant_entradas):
            x_aux.append(float(linea_csv[i]))
        entradas.append(x_aux)
        salidas_deseadas.append(float(linea_csv[i + 1]))
    matrix_entradas = array(entradas)  # Creo una matriz numpy
    matrix_salidas = array(salidas_deseadas)

    return matrix_entradas, matrix_salidas


def add_byass(array, axis):
    return insert(array, 0, -1, axis=axis)


class NeuronLayer():
    def __init__(self, numero_neuronas, numero_entradas_por_neurona):
        self.numero_entradas_por_neurona = numero_entradas_por_neurona
        self.pesos_sinapticos = 2 * random.random((numero_entradas_por_neurona, numero_neuronas)) - 1


class NeuralNetwork():
    def __init__(self, capa1, capa2):
        self.capa1 = capa1
        self.capa2 = capa2

    # La funcion sigmoidea, que describe una curva en forma de S.
    # Pasamos la suma ponderada de las entradas a traves de esta funcion para
    # normalizarlos entre -1 y 1.
    def __sigmoid(self, x):
        return (2 / (1 + exp(-x))) - 1

    # La derivada de la funcion sigmoidea.
    # Este es el gradiente de la curva sigmoidea.
    # Indica cuanta confianza tenemos sobre el peso existente.
    def __sigmoid_derivative(self, x):
        return ((1 + x) * (1 - x)) / 2

    def make_positive_or_negative(self, value):
        return 1 if value > 0 else -1

    # Entrenamos la red neuronal a traves de un proceso de prueba y error.
    # Ajustando los pesos sinapticos cada vez.
    def train(self, entradas_entrenamiento, salidas_entrenamiento, number_of_training_iterations, vel_aprendizaje):
        for iteration in xrange(number_of_training_iterations):
            for index_entradas in xrange(len(entradas_entrenamiento)):
                # Pasamos el set de training a traves de la red
                salida_capa_1, salida_capa_2 = self.think(add_byass(entradas_entrenamiento[index_entradas], 0))

                # Calculamos el error para la capa 2
                capa2_error = salidas_entrenamiento[index_entradas] - salida_capa_2
                capa2_delta = capa2_error * self.__sigmoid_derivative(salida_capa_2)

                # Calculamos el error de la capa 1. Mirando los pesos de la capa 2, podemos determinar cuantro contribuyen al error en la capa 2
                capa1_error = capa2_delta.dot(self.capa2.pesos_sinapticos.T)
                capa1_delta = capa1_error * self.__sigmoid_derivative(add_byass(salida_capa_1, 0))

                # Calculamos cuanto hay que ajustar los pesos
                capa1_ajuste = matmul(reshape(
                    add_byass(entradas_entrenamiento[index_entradas], 0),
                    (self.capa1.numero_entradas_por_neurona, 1)  # Nueva dimension - cant_Entradas x 1
                ),
                    reshape(capa1_delta[1:self.capa1.numero_entradas_por_neurona + 1],
                            (1, self.capa1.numero_entradas_por_neurona)))  # Nueva dimension - 1 x cant_Entradas
                capa2_ajuste = reshape(add_byass(salida_capa_1, 0),
                                       (self.capa2.numero_entradas_por_neurona, 1)) * capa2_delta

                # Ajustamos los pesos
                self.capa1.pesos_sinapticos += vel_aprendizaje * capa1_ajuste
                self.capa2.pesos_sinapticos += vel_aprendizaje * capa2_ajuste

            print iteration

    # Funcion para que la neurona se activen
    def think(self, entradas, axis=0):
        salida_capa1 = self.__sigmoid(dot(entradas, self.capa1.pesos_sinapticos))
        salida_capa2 = self.__sigmoid(dot(add_byass(salida_capa1, axis), self.capa2.pesos_sinapticos))
        return salida_capa1, salida_capa2

    # Muestra los pesos de cada capa
    def print_weights(self):
        print "    Capa 1 (2 neuronas, con 2 entradas cada una): "
        print self.capa1.pesos_sinapticos
        print "    Capa 2 (1 neurona, con 2 entradas):"
        print self.capa2.pesos_sinapticos

    def graph_data(self, entradas, salidas_deseadas):
        Tp = Tn = Fp = Fn = 0
        # True Positive - True Negative - False Positive - False Negative

        # Pasamos las entradas a traves de la red
        salidas_capa_oculta, salidas = self.think(add_byass(entradas, 1), 1)

        for index in xrange(len(salidas)):
            if self.make_positive_or_negative(salidas[index]) == salidas_deseadas[index]:
                if salidas[index] > 0:
                    Tp += 1
                    plt.plot(entradas[index][0], entradas[index][1], 'rx')
                else:
                    Tn = Tn + 1
                    plt.plot(entradas[index][0], entradas[index][1], 'gx')
            else:
                if salidas[index] > 0:
                    Fp = Fp + 1
                    plt.plot(entradas[index][0], entradas[index][1], 'ob')
                else:
                    Fn = Fn + 1
                    plt.plot(entradas[index][0], entradas[index][1], 'ob')

        exactitud = (Tp + Tn) / len(salidas_deseadas)
        # F1_Score = 2 * Tp / (2 * Tp + Fp + Fn)  # Sensibilidad de los +
        print('True +:', Tp, 'True -:', Tn, 'False +:', Fp, 'False -:', Fn)
        # print('Exactitud :', Exactitud, 'F1_Score', F1_Score)
        plt.show()
        return exactitud


if __name__ == "__main__":
    # Seed para generar los randoms
    random.seed(1)

    # Creo la Capa 1 (2 neuronas, con 2 entradas)
    capa1 = NeuronLayer(3, 3)

    # Creo la Capa 2 (1 neurona, con 2 entradas)
    capa2 = NeuronLayer(1, 4)

    # Combino las capas para crear la red
    neural_network = NeuralNetwork(capa1, capa2)

    print "Paso 1) Inicializo los pesos de forma aleatoria: "
    neural_network.print_weights()

    # Cargo el archivo de entrenamiento y lo devido en entradas y salidas
    entradas, salidas = csv_open()
    entradas_entrenamiento = entradas
    salidas_entrenamiento = salidas.T  # .T genera la transpuesta

    # Entrenamos la red con el set de entrenamiento
    # Se entrena 1000 vueltas con un ajuste chico en cada una
    neural_network.train(entradas_entrenamiento, salidas_entrenamiento, 1500, 0.05)

    print "Paso 2) Nuevos pesos luego del entrenamiento: "
    neural_network.print_weights()

    # Probamos la red con una siatuacion totalmente nueva
    print "Paso 3) Probamos la red con datos de prueba "
    exactitud = neural_network.graph_data(entradas_entrenamiento, salidas_entrenamiento)
    print exactitud
