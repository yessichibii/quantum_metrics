import numpy as np
import pennylane as qml
from quantum_metrics import (
    gray_code_inverso,gray_code,binario_code_inverso,binario_code,
    qram_initialize,encode_data,
    mba_distance,state_mba_distance
)


# Ejemplo: dataset de entrenamiento con 4 patrones de 2 características cada uno
train = np.array([
    [1, 0],
    [0, 1],
    [1, 1],
    [0.5, 0.5]
])
test = [0.5,0.5]

# =============================================================
# ------ Codificación de las direcciones de la QRAM -----------
# =============================================================

print("=== Ejemplo de codificaciones ===")
n_bits = 3
# Codificación binaria
binarios = binario_code(n_bits)
inv_bin = binario_code_inverso(n_bits)

# Codificación Gray
grays = gray_code(n_bits)
inv_gray = gray_code_inverso(n_bits)
print("Binario:     ", binarios)
print("Bin→direcciones de control:  ", inv_bin)
print("Gray:        ", grays)
print("Gray→direcciones de control: ", inv_gray)

# =============================================================
# ------ Construcción del circuito QRAM -----------
# =============================================================

def QRAM(codigo,train):
    # Codificación para QRAM
    state_vector, direcciones, qubits_qram, qubits_data = qram_initialize(train,codigo=codigo)
    print("Vector de los estados equiprobables de las direcciones de la QRAM: ", state_vector)
    print("Direcciones de la QRAM: ", direcciones)
    print("Qubits necesarios para la QRAM: ", qubits_qram)
    print("Qubits necesarios para codificar los datos por angulo: ", qubits_data)

    n_total = qubits_qram + qubits_data
    dev = qml.device("default.qubit", wires=n_total)
    @qml.qnode(dev)
    def circuit():
        qml.AmplitudeEmbedding(state_vector, wires=range(qubits_qram), normalize=True)
        return qml.state()

    return circuit()

def QRAM_data(codigo,train):
    # Codificación para QRAM
    state_vector, direcciones, qubits_qram, qubits_data = qram_initialize(train,codigo=codigo)

    n_total = qubits_qram + qubits_data
    dev = qml.device("default.qubit", wires=n_total)
    @qml.qnode(dev)
    def circuit():
        qml.AmplitudeEmbedding(state_vector, wires=range(qubits_qram), normalize=True)
        encode_data(direcciones, qubits_qram, train, codigo=codigo)
        return qml.state()

    return circuit()

# print("=== Ejemplos de QRAM ===")

print("====== Binario =====")
binario = QRAM("binario",train)
print("Binario: ",binario)
binario = QRAM_data("binario",train)
print("Binario inicializado: ",binario)

print("====== Gray =====")
gray = QRAM("gray",train)
print("Gray: ",gray)
gray = QRAM_data("gray",train)
print("Gray inicializado: ",gray)

print("====== Diagonal =====")
diag = QRAM("diag",train)
print("Diagonal: ",diag)
diag = QRAM_data("diag",train)
print("Diagonal inicializado: ",diag)

# # =============================================================
# # ------------ Calcula de distancias cuánticas ----------------
# # =============================================================

def distancia(codigo,train,test):
    # Codificación para QRAM
    _, _, qubits_qram, qubits_data = qram_initialize(train,codigo=codigo)
    
    n_total = qubits_qram + qubits_data
    dev = qml.device("default.qubit", wires=n_total)
    @qml.qnode(dev)
    def circuit():
        mba_distance(train, test, codigo=codigo)
        return qml.state()

    return circuit()

print("====== Binario =====")
binario = distancia("binario",train,test)
print("Binario distancia: ",binario)
binario = state_mba_distance(train, test, codigo="binario", result = "state")
print("Binario distancia mba: ",binario)

print("====== Gray =====")
gray = distancia("gray",train,test)
print("Gray distancia: ",gray)
gray = state_mba_distance(train, test, codigo="gray", result = "state")
print("Gray distancia mba: ",gray)

print("====== Diagonal =====")
diag = distancia("diag",train,test)
print("Diagonal distancia: ",diag)
diag = state_mba_distance(train, test, codigo="diag", result = "state")
print("Diagonal distancia mba: ",diag)
