from qiskit import *
from qiskit.circuit import Gate

from qiskit.circuit.library import RYGate
import numpy as np

def int_to_x(n, i):
    qc = QuantumCircuit(n)
    qc.name = "B"+str(i)
    for j in range(n):
        if (i>>j)&1 == 1:
            qc.x(j)
    return qc

def obtenerSecuenciaXMas1(n_size, n_actual, n_meta):
    sec_final = ['-' for i in range(n_size)]
    for i in range(n_actual+1, n_meta+1):

        # secuencia = []
        for j in range(n_size):

            # if i%(2**j) == 0 and sec_final[n_size - j - 1] == '-':
            #     sec_final[n_size - j - 1] = 'x'
            # elif i%(2**j) == 0 and sec_final[n_size - j - 1] == 'x':
            #     sec_final[n_size - j - 1] = '-'

            if i%(2**j) == 0 and sec_final[j] == '-':
                sec_final[j] = 'x'
            elif i%(2**j) == 0 and sec_final[j] == 'x':
                sec_final[j] = '-'

            # if i%(2**j) == 0:
            #     secuencia.insert(0, 'x')
            # else:
            #     secuencia.insert(0, '-')

        # print(secuencia)
    # print("Sec Final:",sec_final)

    qc = QuantumCircuit(n_size)
    qc.name = "Sec_"+str(n_actual)+"-"+str(n_meta)
    for j in range(n_size):
        if sec_final[j] == 'x':
            qc.x(j)
    return qc


def QRAM(datos, n_data):
    n = int(np.round(np.log2(len(datos))))

    x = QuantumRegister(n, "x")
    data = QuantumRegister(n_data, "d")

    qc = QuantumCircuit(data,x)
    qc.name = "QRAM"

    qc.h(x)
    for i in range(len(datos)):
        qc.append(int_to_x(n, ~i), x)
        qc.append(int_to_x(n_data,datos[i]).control(n), x[:] + data[:])
        qc.append(int_to_x(n, ~i), x)
    return qc

def QRAM_image(imagen):
    n_x = int(np.round(np.log2(imagen.shape[0])))
    n_y = int(np.round(np.log2(imagen.shape[1])))

    channels = 1
    if len(imagen.shape)>2:
        channels = imagen.shape[2]
    n_data = channels*8

    x = QuantumRegister(n_x, "x")
    y = QuantumRegister(n_y, "y")
    data = QuantumRegister(n_data, "d")

    qc = QuantumCircuit(data,y,x)

    qc.h(x)
    qc.h(y)

    for yi in range(imagen.shape[1]):
        qc.append(int_to_x(n_y, ~yi), y)
        for xi in range(imagen.shape[0]):
            qc.append(int_to_x(n_x, ~xi), x)
            if channels != 1:
                for a in range(channels):
                    qc.append(int_to_x(8, imagen[yi][xi][a]).control(n_x + n_y), x[:] + y[:] + data[a*8:(a+1)*8])
                    
            else:
                qc.append(int_to_x(8, imagen[yi][xi]).control(n_x + n_y), x[:] + y[:] + data[:])

            qc.append(int_to_x(n_x, ~xi), x)
        qc.append(int_to_x(n_y, ~yi), y)

    return qc


class QRAM(QuantumCircuit):
    def __init__(self, n_qbts_posicion, n_qbts_datos, initialize = True):
        super().__init__()
        self.name = "QRAM"

        self.Posicion = QuantumRegister(n_qbts_posicion, name="Posicion")
        self.Datos = QuantumRegister(n_qbts_datos, name="Datos")

        self.add_register(self.Posicion)
        self.add_register(self.Datos)
        
        if initialize:
            self.inicializarRegistrDePosicion()


    def inicializarRegistrDePosicion(self):
        self.h(self.Posicion)

    def agregarDatoBin(self, posicion, dato):
        self.append(int_to_x(self.Posicion.size, ~posicion), self.Posicion)
        self.append(int_to_x(self.Datos.size, ~dato).control(self.Posicion.size), self.Posicion[:] + self.Datos[:])
        self.append(int_to_x(self.Posicion.size, ~posicion), self.Posicion)

    def agregarDatoRy(self, posicion, dato, dato_max, angulo_max = np.pi):
        self.append(int_to_x(self.Posicion.size, ~posicion), self.Posicion)
        self.append(RYGate(angulo_max * (dato/dato_max)).control(self.Posicion.size), self.Posicion[:] + self.Datos[:])
        self.append(int_to_x(self.Posicion.size, ~posicion), self.Posicion)

    def copiarDatosBin(self, inicio, lista):
        for i in range(len(lista)):
            self.append(obtenerSecuenciaXMas1(self.Posicion.size, i-1, i), self.Posicion)
            self.append(int_to_x(self.Datos.size, ~lista[i]).control(self.Posicion.size), self.Posicion[:] + self.Datos[:])

    def copiarDatosRy(self, inicio, lista, dato_max, angulo_max = np.pi):
        for i in range(len(lista)):
            self.append(obtenerSecuenciaXMas1(self.Posicion.size, i-1, i), self.Posicion)
            self.append(RYGate(angulo_max * (lista[i]/dato_max)).control(self.Posicion.size), self.Posicion[:] + self.Datos[:])


def obtenerProbabilidadEsperada_Full(u, v, max_val):

    pr = 0

    for i in range(len(u)):
        theta = np.pi*u[i]/max_val
        phi = np.pi*v[i]/max_val

        pr += (np.cos((theta-phi)/2))
    
    return 1/2 + (1/2)*((pr/len(u))**2)

def obtenerProbabilidadEsperada_Parcial(u, v, max_val):

    pr = 0

    for i in range(len(u)):
        theta = np.pi*u[i]/max_val
        phi = np.pi*v[i]/max_val

        pr += (np.cos(theta-phi)+1)
    
    return 1/2 + (1/2)*((pr/2)/len(u))

def obtenerProbabilidadesEsperadaMultivector_Full(u, vs, max_val):

    prbs = []

    M = len(vs)
    N = len(u)

    for v in vs:
        pr = 0

        for i in range(N):
            theta = np.pi*u[i]/max_val
            phi = np.pi*v[i]/max_val

            pr += (np.cos((theta-phi)/2))

        prbs.append(1/(2*M) + (1/(2*M))*(     (   pr/N   )**2     ))
    
    return np.array(prbs)

def obtenerProbabilidadesEsperadaMultivector_Parcial(u, vs, max_val):

    prbs = []

    M = len(vs)
    N = len(u)

    for v in vs:
        pr = 0

        for i in range(N):
            theta = np.pi*u[i]/max_val
            phi = np.pi*v[i]/max_val

            pr += (np.cos(theta-phi) + 1)

        prbs.append(1/(2*M) + (1/(2*M))*(     (   pr/2   )/N     ))
    
    return np.array(prbs)

def prueba_1_1_Full():

    p_img_1 = QuantumRegister(2, "p_img_1")
    p_img_2 = QuantumRegister(2, "p_img_2")
    d_img_1 = QuantumRegister(1, "d_img_1")
    d_img_2 = QuantumRegister(1 , "d_img_2")
    st_qbit = QuantumRegister(1, "st_qbit")

    st_meas = ClassicalRegister(1, "st_meas")

    u = [255, 170, 85, 0]
    v = [0, 85, 170, 255]

    prb = obtenerProbabilidadEsperada_Full(u, v, 255)

    print("Probabilidad esperada: " + str(prb))
    print("Similitud: " + str(2*prb - 1))


    ram1 = QRAM(n_qbts_posicion=2, n_qbts_datos=1)
    ram1.copiarDatosRy(0, u, 255)

    ram2 = QRAM(n_qbts_posicion=2, n_qbts_datos=1)
    ram2.copiarDatosRy(0, v, 255)

    qc = QuantumCircuit(p_img_1, d_img_1, p_img_2, d_img_2, st_qbit, st_meas)

    qc.append(ram1, p_img_1[:] + d_img_1[:])
    qc.append(ram2, p_img_2[:] + d_img_2[:])

    qc.h(st_qbit)
    for i in range(int(np.ceil(np.log2(len(u))))):
        qc.cswap(st_qbit, p_img_1[i], p_img_2[i])
    qc.cswap(st_qbit, d_img_1, d_img_2)
    qc.h(st_qbit)

    qc.measure(st_qbit, st_meas)

    qc.draw("mpl")
    plt.waitforbuttonpress()

    be = Aer.get_backend("statevector_simulator")
    trn = transpile(qc, be)
    job = be.run(trn, shots=1024)
    result = job.result()

    counts = result.get_counts()

    plot_histogram(counts)


    # ram.draw("mpl")
    plt.waitforbuttonpress()

def prueba_1_1_Partial():

    p_img = QuantumRegister(2, "p_img")
    d_img_1 = QuantumRegister(1, "d_img_1")
    d_img_2 = QuantumRegister(1 , "d_img_2")
    st_qbit = QuantumRegister(1, "st_qbit")

    st_meas = ClassicalRegister(1, "st_meas")

    u = [255, 170, 85, 0]
    v = [0, 85, 170, 255]

    prb = obtenerProbabilidadEsperada_Parcial(u, v, 255)

    print("Probabilidad esperada: " + str(prb))
    print("Similitud: " + str(2*prb - 1))


    ram1 = QRAM(n_qbts_posicion=2, n_qbts_datos=1)
    ram1.copiarDatosRy(0, u, 255)

    ram2 = QRAM(n_qbts_posicion=2, n_qbts_datos=1, initialize=False)
    ram2.copiarDatosRy(0, v, 255)

    qc = QuantumCircuit(p_img, d_img_1, d_img_2, st_qbit, st_meas)

    qc.append(ram1, p_img[:] + d_img_1[:])
    qc.append(ram2, p_img[:] + d_img_2[:])

    qc.h(st_qbit)
    qc.cswap(st_qbit, d_img_1, d_img_2)
    qc.h(st_qbit)

    qc.measure(st_qbit, st_meas)

    qc.draw("mpl")
    plt.waitforbuttonpress()

    be = Aer.get_backend("statevector_simulator")
    trn = transpile(qc, be)
    job = be.run(trn, shots=2**16)
    result = job.result()

    counts = result.get_counts()

    plot_histogram(counts)


    # ram.draw("mpl")
    plt.waitforbuttonpress()

def prueba_1_m_Full():

    y       = QuantumRegister(2, "y")
    p_img_1 = QuantumRegister(2, "p_img_1")
    p_img_2 = QuantumRegister(2, "p_img_2")
    d_img_1 = QuantumRegister(1, "d_img_1")
    d_img_2 = QuantumRegister(1 , "d_img_2")
    st_qbit = QuantumRegister(1, "st_qbit")

    y_meas = ClassicalRegister(2, "y_meas")
    st_meas = ClassicalRegister(1, "st_meas")

    u = [255, 170, 85, 0]
    vs = [
        [0, 85, 170, 255],
        [0, 170, 85, 255],
        [170, 0, 85, 255],
        [255, 170, 85, 0]
          ]

    pbs = obtenerProbabilidadesEsperadaMultivector_Full(u, vs, 255)

    print("Probabilidad esperada: ", pbs)
    print("Similitud: ", 2*(len(vs))*pbs - 1)

    ram1 = QRAM(n_qbts_posicion=2, n_qbts_datos=1)
    ram1.copiarDatosRy(0, u, 255)

    ram2 = QRAM(n_qbts_posicion=4, n_qbts_datos=1)
    ram2.copiarDatosRy(0, np.array(vs).reshape(( (len(vs)*len(vs[0])) )), 255)

    qc = QuantumCircuit(p_img_1, d_img_1, y, p_img_2, d_img_2, st_qbit, st_meas, y_meas)

    qc.append(ram1, p_img_1[:] + d_img_1[:])
    qc.append(ram2, p_img_2[:] + y[:] + d_img_2[:])

    qc.h(st_qbit)
    for i in range(int(np.ceil(np.log2(len(u))))):
        qc.cswap(st_qbit, p_img_1[i], p_img_2[i])
    qc.cswap(st_qbit, d_img_1, d_img_2)
    qc.h(st_qbit)


    qc.barrier()
    qc.measure( st_qbit, st_meas)
    qc.measure( y, y_meas)

    plt.figure(3)
    qc.draw("mpl")
    plt.waitforbuttonpress()

    be = Aer.get_backend("statevector_simulator")
    trn = transpile(qc, be)
    job = be.run(trn, shots=2**16)
    result = job.result()

    counts = result.get_counts()

    plt.figure(4)
    plot_histogram(counts)


    # ram.draw("mpl")
    plt.waitforbuttonpress()

def prueba_1_m_Parcial():

    y       = QuantumRegister(2, "y")
    p_img = QuantumRegister(2, "p_img")
    d_img_1 = QuantumRegister(1, "d_img_1")
    d_img_2 = QuantumRegister(1 , "d_img_2")
    st_qbit = QuantumRegister(1, "st_qbit")

    y_meas = ClassicalRegister(2, "y_meas")
    st_meas = ClassicalRegister(1, "st_meas")

    u = [255, 170, 85, 0]
    vs = [
        [0, 85, 170, 255],
        [0, 170, 85, 255],
        [170, 0, 85, 255],
        [255, 170, 85, 0]
          ]

    pbs = obtenerProbabilidadesEsperadaMultivector_Parcial(u, vs, 255)

    print("Probabilidad esperada: ", pbs)
    print("Similitud: ", 2*(len(vs))*pbs - 1)

    ram1 = QRAM(n_qbts_posicion=2, n_qbts_datos=1)
    ram1.copiarDatosRy(0, u, 255)

    ram2 = QRAM(n_qbts_posicion=4, n_qbts_datos=1, initialize=False)
    ram2.h(2)
    ram2.h(3)
    ram2.copiarDatosRy(0, np.array(vs).reshape(( (len(vs)*len(vs[0])) )), 255)

    qc = QuantumCircuit(y, p_img, d_img_1, d_img_2, st_qbit, st_meas, y_meas)

    qc.append(ram1, p_img[:] + d_img_1[:])
    qc.append(ram2, p_img[:] + y[:] + d_img_2[:])

    qc.h(st_qbit)
    qc.cswap(st_qbit, d_img_1, d_img_2)
    qc.h(st_qbit)


    qc.barrier()
    qc.measure( st_qbit, st_meas)
    qc.measure( y, y_meas)

    plt.figure(3)
    qc.draw("mpl")
    plt.waitforbuttonpress()

    be = Aer.get_backend("statevector_simulator")
    trn = transpile(qc, be)
    job = be.run(trn, shots=2**16)
    result = job.result()

    counts = result.get_counts()

    plt.figure(4)
    plot_histogram(counts)

    plt.waitforbuttonpress()

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from qiskit_aer import Aer
    from qiskit.visualization import plot_histogram

    # prueba_1_1_Full()
    # prueba_1_1_Partial()
    # prueba_1_m_Full()
    # prueba_1_m_Parcial()



