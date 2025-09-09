
from quantum_metrics import h_amplitude_distance, amplitude_distance
import numpy as np
import pennylane as qml

x_train = np.array([[0.0], [1.0]])
y_train = np.array([[1,0], [0,1]])
test = [0.0]

# state = h_amplitude_distance(x_train, test, labels=y_train)
# print("Estado cuántico:", state)
# print("Norma:", np.sum(np.abs(state) ** 2))


m, qubits_dato = x_train.shape
_, qubits_label = y_train.shape
qubits_qram= int(np.ceil(np.log2(m)))
n_totales = qubits_qram + qubits_dato
n_totales += qubits_label
n_totales += 1 # aux
print(n_totales)

dev = qml.device("default.qubit", wires=n_totales)

@qml.qnode(dev)
def circuit():
    amplitude_distance(x_train, test, tipo="multilabel", labels=y_train)

    for i in range(qubits_dato):
        qml.ctrl(qml.RY, control=qubits_qram+i)(np.pi/qubits_dato, wires=n_totales-1)
    
    for j in range(qubits_label):
        qml.CNOT(wires=[n_totales-1, qubits_qram+qubits_dato+j])
    return qml.probs(wires=range(qubits_qram+qubits_dato,n_totales-1))

print(circuit())