
from quantum_metric.quantum.distances import state_mba_distance, mba_distance
import numpy as np
import pennylane as qml

x_train = np.array([[0.0], [1.0]])
y_train = np.array([1, 0])
test = [0.0]

state = state_mba_distance(x_train, test, labels=y_train)
print("Estado cuántico:", state)
print("Norma:", np.sum(np.abs(state) ** 2))

m, qubits_dato = x_train.shape
qubits_qram= int(np.ceil(np.log2(m)))
n_totales = qubits_qram + qubits_dato
n_totales += 2 #1labels + 1 aux

dev = qml.device("default.qubit", wires=n_totales)

@qml.qnode(dev)
def biclase():
    mba_distance(x_train, test, labels=y_train, codigo="gray")

    for i in range(qubits_dato):
        qml.ctrl(qml.RY, control=qubits_qram+i)(np.pi/qubits_dato, wires=n_totales-1)
    qml.CNOT(wires=[n_totales-1, qubits_qram+qubits_dato])
    return qml.probs(wires=range(qubits_qram+qubits_dato,n_totales-1))

print(biclase())