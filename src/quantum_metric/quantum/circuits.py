import pennylane as qml
import numpy as np
from qiskit_aer.noise import NoiseModel
from .distances import mba_distance, distance

def make_device(n_wires: int, backend=None, shots: int = 1024, noise_model=None, sample_mode: bool = False):
    """
    Crea un dispositivo de simulación cuántica.

    Args:
        n_wires (int): Número de qubits (wires)
        backend: Backend opcional de Qiskit
        shots (int): Número de mediciones por muestreo
        noise_model: Modelo de ruido (si se desea simular decoherencia)
        sample_mode (bool): Si True, simula muestreo sin ruido

    Returns:
        qml.Device: Dispositivo de PennyLane configurado
    """
    
    if backend is not None:
        nm = NoiseModel.from_backend(backend) if noise_model is None else noise_model
        return qml.device("qiskit.aer", wires=n_wires, backend="qasm_simulator", noise_model=nm, shots=shots)
    
    # ✅ Caso: simulación sin ruido pero con muestreo
    if sample_mode:
        return qml.device("default.qubit", wires=n_wires, shots=shots)
    
    # ✅ Caso: simulación ideal (sin muestreo)
    return qml.device("default.qubit", wires=n_wires)

def build_biclase_qnode(train, y_train, n_totales, qubits_qram, qubits_dato, device, codigo="gray", noise=0.0, result="probs"):
    @qml.qnode(device, interface="autograd")
    def circuit(test, params):
        # print(f"params: {params}")
        # print(f"n_totales: {n_totales}")
        # print(f"qubits_qram: {qubits_qram}")
        # print(f"qubits_dato: {qubits_dato}")
        mba_distance(train, test, labels = y_train, codigo=codigo, noise=noise)
        # print(f"test: {test}")
        if params is not None:
            distance(qubits_qram, params)
            # print(f"distance: {qubits_qram}")

        for i in range(qubits_dato):
            qml.ctrl(qml.RY, control=qubits_qram+i)(np.pi/qubits_dato, wires=n_totales-1)
            # print(f"{qubits_qram+i} : {n_totales-1}")
            # Only add noise if using a noise-compatible device
            if noise > 0 and hasattr(device, 'capabilities') and device.capabilities().get('supports_noise', False):
                qml.DepolarizingChannel(noise, wires = n_totales-1)

        qml.CNOT(wires=[n_totales-1, qubits_qram+qubits_dato])
        # print(f"{n_totales-1} : {qubits_qram+qubits_dato}")
        # Only add noise if using a noise-compatible device
        if noise > 0 and hasattr(device, 'capabilities') and device.capabilities().get('supports_noise', False):
            qml.DepolarizingChannel(noise, wires = qubits_qram+qubits_dato)

        if result == "probs":
            # Only add noise if using a noise-compatible device
            if noise > 0 and hasattr(device, 'capabilities') and device.capabilities().get('supports_noise', False):
                qml.AmplitudeDamping(2*noise, wires=0)
            # print(f"{n_totales-1} : {qubits_qram+qubits_dato}")
            return qml.probs(wires=range(qubits_qram+qubits_dato,n_totales-1))
        return qml.state()

    return circuit

def build_multiclase_qnode(train, y_train, n_totales, qubits_qram, qubits_dato, qubits_label, device, codigo="gray", noise=0.0, result="probs"):
    @qml.qnode(device, interface="autograd")
    def circuit(test, params):
        # print(f"params: {params}")
        # print(f"n_totales: {n_totales}")
        # print(f"qubits_qram: {qubits_qram}")
        # print(f"qubits_dato: {qubits_dato}")
        # print(f"qubits_label: {qubits_label}")
        mba_distance(train, test, tipo="multiclase", labels=y_train, codigo=codigo, noise=noise)
        # print(f"test: {test}")
        if params is not None:
            distance(qubits_qram, params)
            # print(f"distance: {qubits_qram}")

        for i in range(qubits_dato):
            qml.ctrl(qml.RY, control=qubits_qram+i)(np.pi/qubits_dato, wires=n_totales-1)
            # print(f"{qubits_qram+i} : {n_totales-1}")
            # Only add noise if using a noise-compatible device
            if noise > 0 and hasattr(device, 'capabilities') and device.capabilities().get('supports_noise', False):
                qml.DepolarizingChannel(noise, wires = n_totales-1)

        # Note: No CNOT gate after the rotation loop for multiclass
        # This is the key difference from the binary classifier

        if result == "probs":
            # Only add noise if using a noise-compatible device
            if noise > 0 and hasattr(device, 'capabilities') and device.capabilities().get('supports_noise', False):
                qml.AmplitudeDamping(2*noise, wires=0)
            # print(f"{n_totales-1} : {qubits_qram+qubits_dato}")
            return qml.probs(wires=range(qubits_qram+qubits_dato,n_totales-1))
        return qml.state()

    return circuit