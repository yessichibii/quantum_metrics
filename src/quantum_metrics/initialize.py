import pennylane as qml
import numpy as np
from .utils import gray_code_inverso, binario_code_inverso


# =============================================================
# ------------------ INICIALIZACIÓN DE QRAM --------------------
# =============================================================

def qram_initialize(dataset, codigo="gray"):
    """
    Inicializa un vector de amplitudes y el código Gray para construir el QRAM.

    Parámetros
    ----------
    dataset : np.ndarray
        Conjunto de datos de forma (m, n), donde m = número de muestras.
    gray_mode : str, default gray
        'gray', 'diag' o 'binario' para definir el tipo de salida.

    Retorna
    -------
    state_vector : np.ndarray
        Vector de amplitudes normalizado.
    direcciones : list[str]
        Lista de direcciones en el codigo seleccionado.
    qubits_qram : int
        Número de qubits necesarios para direccionar las muestras.
    qubits_dato : int
        Número de qubits que representan los datos.
    """

    m, qubits_dato = dataset.shape
    qubits_qram = int(np.ceil(np.log2(m)))

    state_vector = np.zeros(2**qubits_qram)
    data = []
    direcciones = []

    # Generamos el codigo de num_qubits bits
    match codigo:
        case "binario":
            data = binario_code_inverso(qubits_qram)
        case "diag":
            qubits_qram = m
            state_vector = np.zeros(2**m)
            for i in range(2**m):
                if bin(i).count('1') == 1:
                    data.append(format(i, f"0{qubits_qram}b"))
        case "gray":
            data = gray_code_inverso(qubits_qram)
        case _:
            raise ValueError(f'Codigo "{codigo}" no válido')
            
    for i in range(m):
        idx = int(data[i], 2)
        direcciones.append(data[i])
        state_vector[idx] = 1

    norm_factor = np.linalg.norm(state_vector)
    if norm_factor > 0:
        state_vector /= norm_factor
    return state_vector, direcciones, qubits_qram, qubits_dato


def encode_data(direcciones, qubits_qram, dataset, tipo="biclase", labels=None, noise=0.0, codigo='gray'):
    """
    Codifica los datos y etiquetas en un circuito cuántico QRAM.

    Parámetros
    ----------
    direcciones : list[str]
        Direcciones en código Gray, binario o diagonal.
    qubits_qram : int
        Número de qubits de dirección (QRAM).
    dataset : np.ndarray
        Datos normalizados en [0, 1].
    tipo : str, default biclase
        'biclase', 'multiclase' o 'multilabel' para definir el tipo de salida.
    labels : list, int o None, opcional
        Etiquetas asociadas a cada muestra, biclase y multiclase int multilabel list.
    noise : float, opcional
        Probabilidad del canal de ruido (0.0 desactiva el ruido).
    codigo: str, default gray
        'gray', 'diag' o 'binario' para definir el tipo de salida.
    """

    g_prev = 2**qubits_qram -1
    recuperar_estado = []
    controls = range(qubits_qram)

    for i, datos in enumerate(dataset):

        if codigo == "diag":
            controls = range(i,i+1)
        else:
            g_curr = int(direcciones[i],2)
            mask = g_prev ^ g_curr
            lsb_pos = [p for p in range(len(direcciones[i])) if mask & (1 << (len(direcciones[i])-1-p))]
            g_prev = g_curr
            
            for pos in lsb_pos:
                if pos in recuperar_estado:
                    recuperar_estado.remove(pos)
                else:
                    recuperar_estado.append(pos)
                qml.PauliX(wires=pos)
                if noise > 0:
                    qml.DepolarizingChannel(noise, wires=pos)
        
        # Codificación de datos
        for j, val in enumerate(datos):
            theta = val * np.pi
            qml.ctrl(qml.RY, control=controls)(theta, wires=qubits_qram + j)
            if noise > 0:
                qml.DepolarizingChannel(noise, wires=qubits_qram + j)

        # Codificación de etiquetas
        if labels is not None:
            start_wire = qubits_qram + len(datos)
            label_addr = []
            match tipo:
                case "multiclase":
                    qubits_label = int(np.ceil(np.log2(max(labels))))
                    label_binario = format(labels[i], f"0{qubits_label}b")
                    label_addr = [int(b) for b in label_binario]
                    
                case "multilabel":
                    label_addr = labels[i]
                            
                case _:
                    label_addr = [labels[i]]
            for j, val in enumerate(label_addr):
                if val == 1:
                    qml.ctrl(qml.RY, control=controls)(np.pi, wires = start_wire + j)
                    if noise > 0:
                        qml.DepolarizingChannel(noise, wires = start_wire + j)

        # Restauración del control 
        for pos in recuperar_estado:
            qml.PauliX(wires=pos)
            if noise > 0:
                qml.DepolarizingChannel(noise, wires=pos)

