
from quantum_metric.quantum.distances import  mba_distance
import numpy as np
import pennylane as qml

x_train = np.array([[0.0,0.0], [1.0,0.0], [0.0,1.0], [1.0,1.0]])
y_train = np.array([1, 0, 1, 2])
test = [0.0,0.0]

m, qubits_dato = x_train.shape
qubits_qram= int(np.ceil(np.log2(m)))
qubits_label = int(np.ceil(np.log2(len(np.unique(y_train)))))
n_totales = qubits_qram + qubits_dato + qubits_label
n_totales += 1
print(n_totales)

dev = qml.device("default.qubit", wires=n_totales)

@qml.qnode(dev)
def circuit():
    
    mba_distance(x_train, test, tipo="multiclase", labels=y_train)
    for i in range(qubits_dato):
        qml.ctrl(qml.RY, control=qubits_qram+i)(np.pi/qubits_dato, wires=n_totales-1)
    return qml.probs(wires=range(qubits_qram+qubits_dato,n_totales-1))

print(circuit())