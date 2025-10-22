import pennylane as qml
import numpy as np
from qiskit_aer.noise import NoiseModel
from .distances import mba_distance, distance

def make_device(n_wires: int, backend=None, shots: int = 1024, noise_model=None):
    if backend is not None:
        nm = NoiseModel.from_backend(backend) if noise_model is None else noise_model
        return qml.device("qiskit.aer", wires=n_wires, backend="qasm_simulator", noise_model=nm, shots=shots)
    return qml.device("default.qubit", wires=n_wires)

def build_biclase_qnode(train, y_train, n_totales, qubits_qram, qubits_dato, device, codigo="gray", noise=0.0, result="probs"):
    @qml.qnode(device, interface="autograd")
    def circuit(test, params):
        mba_distance(train, test, labels = y_train, codigo=codigo, noise=noise)
        if params is not None:
            distance(qubits_qram, params)

        for i in range(qubits_dato):
            qml.ctrl(qml.RY, control=qubits_qram+i)(np.pi/qubits_dato, wires=n_totales-1)
            if noise > 0:
                qml.DepolarizingChannel(noise, wires = n_totales-1)

        qml.CNOT(wires=[n_totales-1, qubits_qram+qubits_dato])
        if noise > 0:
            qml.DepolarizingChannel(noise, wires = qubits_qram+qubits_dato)

        if result == "probs":
            if noise > 0:
                qml.AmplitudeDamping(2*noise, wires=0)
            return qml.probs(wires=range(qubits_qram+qubits_dato,n_totales-1))
        return qml.state()

    return circuit
