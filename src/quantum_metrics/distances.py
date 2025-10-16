import pennylane as qml
import numpy as np
from scipy.linalg import sqrtm

def trace_distance(rho, sigma):
    """Calcula la distancia traza entre dos matrices densidad."""
    diff = rho - sigma
    eigvals = np.linalg.eigvals(diff @ diff.conj().T)
    return 0.5 * np.sum(np.sqrt(np.abs(eigvals)))

def fidelity(rho, sigma):
    """Calcula la fidelidad cuántica entre dos matrices densidad."""
    sqrt_rho = sqrtm(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho
    return np.real(np.trace(sqrtm(inner)) ** 2)


def amplitude_distance(train, test, tipo="biclase", labels=None):
    state_vector, gray, qubits_qram, _ = init_qram(train)
    qml.AmplitudeEmbedding(state_vector, wires=range(qubits_qram), normalize=True)
    init_data(gray, qubits_qram, train, tipo, labels)
    distance(qubits_qram, test)
    

def amplitude_distance2(train, test, tipo="biclase", labels=None):
    state_vector, gray, qubits_qram, _ = init_qram(train)
    qml.AmplitudeEmbedding(state_vector, wires=range(qubits_qram), normalize=True)
    init_data2(gray, qubits_qram, train, tipo, labels)
    distance(qubits_qram, test)

def h_amplitude_distance(train, test, tipo="biclase", labels=None):
    state_vector, gray, qubits_qram, qubits_dato = init_qram(train)
    n_totales = qubits_qram + qubits_dato

    if labels is not None:
        if tipo == "biclase":
            n_totales += 1
        elif tipo == "multilabel":
            n_totales += len(labels[0])
    
    
    dev = qml.device("default.qubit", wires=n_totales)

    @qml.qnode(dev)
    def circuit():
        qml.AmplitudeEmbedding(state_vector, wires=range(qubits_qram), normalize=True)
        init_data(gray, qubits_qram, train, tipo, labels)
        distance(qubits_qram,test)
        return qml.state()

    return circuit()

# Generador de código Gray de n bits
def gray_code_inverso(n):
    if n == 0:
        return ["0"]
    if n == 1:
        return ["0", "1"]
    prev = gray_code_inverso(n-1)
    normal = ["0" + x for x in prev] + ["1" + x for x in reversed(prev)]
    invertido = ["".join("1" if b == "0" else "0" for b in code) for code in normal]
    return invertido

def gray_code(n):
    if n == 0:
        return ["0"]
    if n == 1:
        return ["0", "1"]
    prev = gray_code(n-1)
    return ["0" + x for x in prev] + ["1" + x for x in reversed(prev)]

# Inicializar QRAM
def init_qram(dataset):
    m, qubits_dato = dataset.shape
    qubits_qram= int(np.ceil(np.log2(m)))
    
    # Generamos Gray code de num_qubits bits
    gray = gray_code(qubits_qram)
    # Vector de amplitudes de dimensión 2**n
    state_vector = np.zeros(2**qubits_qram)

    # Ponemos 1 en las primeras m posiciones de Gray code
    for i in range(m):
        idx = int(gray[i], 2)   # posición en decimal
        state_vector[idx] = 1
    # Normalizamos
    state_vector = state_vector / np.linalg.norm(state_vector)

    return state_vector, gray, qubits_qram, qubits_dato

# Inicializar QRAM con 1 en las diagonales
def init_qram_mod(qubits_qram):
    m = 2**qubits_qram
    # Vector de amplitudes de dimensión 2**n
    state_vector = np.zeros(m)

    # Ponemos 1 en los binarios con un solo 1 (diagonal)
    for i in range(m):
        if bin(i).count('1') == 1:
            state_vector[i] = 1
    # Normalizamos
    norm_factor = np.linalg.norm(state_vector)
    if norm_factor > 0:
        state_vector /= norm_factor
    return state_vector

def init_data(gray, qubits_qram, dataset, tipo, labels):
    for i, datos in enumerate(dataset):
        addr = [int(b) for b in gray[i]]
        controls = []
        
        # Control en 0 → flip antes y después
        for k, a in enumerate(addr):
            if a == 1:
                controls.append(k)
            else:
                qml.PauliX(wires=k)
                controls.append(k)
        

        for j, val in enumerate(datos):
            # Escalamos valor [0,1] a ángulo [0,π]
            theta = val * np.pi
            qml.ctrl(qml.RY, control=controls)(theta, wires=qubits_qram + j)

        if labels is not None:
            if tipo == "biclase":
                # Un qubit de clase por patrón
                label = labels[i]  # 0 o 1
                wire_clase = qubits_qram + len(datos)  # siguiente qubit después de los datos
                if label == 1:
                    qml.ctrl(qml.RY, control=controls)(np.pi, wires=wire_clase) # |1> si label=1
            elif tipo == "multilabel":
                # Cada elemento del label corresponde a un qubit
                label_vector = labels[i]  # arreglo de 0 y 1
                start_wire = qubits_qram + len(datos)
                for j, val in enumerate(label_vector):
                    wire_clase = start_wire + j
                    if val == 1:
                        qml.ctrl(qml.RY, control=controls)(np.pi, wires=wire_clase) # |1> si label=1


        # Control en 0 → flip antes y después
        for k, a in enumerate(addr):
            if a == 0:
                qml.PauliX(wires=k)


def init_data2(gray, qubits_qram, dataset, tipo, labels):
    for k in range(qubits_qram):
        qml.PauliX(wires=k)
    g_prev = 0
    for i, datos in enumerate(dataset):
        addr = [int(b) for b in gray[i]]
        controls = range(qubits_qram)
        g_curr = i ^ (i >> 1)
        mask = g_prev ^ g_curr
        lsb_pos = (mask.bit_length() - 1)
        g_prev = g_curr
        # Control en 0 → flip antes y después
        if lsb_pos >= 0:
            qml.PauliX(wires=lsb_pos)

        for j, val in enumerate(datos):
            # Escalamos valor [0,1] a ángulo [0,π]
            theta = val * np.pi
            qml.ctrl(qml.RY, control=controls)(theta, wires=qubits_qram + j)

        if labels is not None:
            if tipo == "biclase":
                # Un qubit de clase por patrón
                label = labels[i]  # 0 o 1
                wire_clase = qubits_qram + len(datos)  # siguiente qubit después de los datos
                if label == 1:
                    qml.ctrl(qml.RY, control=controls)(np.pi, wires=wire_clase) # |1> si label=1
            elif tipo == "multilabel":
                # Cada elemento del label corresponde a un qubit
                label_vector = labels[i]  # arreglo de 0 y 1
                start_wire = qubits_qram + len(datos)
                for j, val in enumerate(label_vector):
                    wire_clase = start_wire + j
                    if val == 1:
                        qml.ctrl(qml.RY, control=controls)(np.pi, wires=wire_clase) # |1> si label=1


    # # Control en 0 → flip antes y después
    addr = [int(b) for b in gray[len(dataset)-1]]
    for k, a in enumerate(addr):
        if a == 0:
            qml.PauliX(wires=len(addr)-k-1)



def init_data_mod(dataset):
    q_qram = len(dataset)
    for i, datos in enumerate(dataset):
        for j, val in enumerate(datos):
            # Escalamos valor [0,1] a ángulo [0,π]
            theta = val * np.pi
            qml.ctrl(qml.RY, control = i)(theta, wires = q_qram + j)


def distance(qubits_qram,test):
    for j, val in enumerate(test):
        # Escalamos valor [0,1] a ángulo [0,π]
        theta = -1 * val * np.pi
        # Aplicamos RY inverso en los qubits de dato
        qml.RY(theta, wires=qubits_qram + j)

def distance_noise(qubits_qram,test):
    for j, val in enumerate(test):
        # Escalamos valor [0,1] a ángulo [0,π]
        theta = -1 * val * np.pi
        # Aplicamos RY inverso en los qubits de dato
        qml.RY(theta, wires=qubits_qram + j)
        qml.DepolarizingChannel(0.02, wires=qubits_qram + j)

def init_data_noise(gray, qubits_qram, dataset, tipo, labels):
    for i, datos in enumerate(dataset):
        addr = [int(b) for b in gray[i]]
        controls = []
        
        # Control en 0 → flip antes y después
        for k, a in enumerate(addr):
            if a == 1:
                controls.append(k)
            else:
                qml.PauliX(wires=k)
                qml.DepolarizingChannel(0.02, wires=k)
                controls.append(k)
        

        for j, val in enumerate(datos):
            # Escalamos valor [0,1] a ángulo [0,π]
            theta = val * np.pi
            qml.ctrl(qml.RY, control=controls)(theta, wires=qubits_qram + j)
            qml.DepolarizingChannel(0.02, wires=qubits_qram + j)

        if labels is not None:
            if tipo == "biclase":
                # Un qubit de clase por patrón
                label = labels[i]  # 0 o 1
                wire_clase = qubits_qram + len(datos)  # siguiente qubit después de los datos
                if label == 1:
                    qml.ctrl(qml.RY, control=controls)(np.pi, wires=wire_clase)
                    qml.DepolarizingChannel(0.02, wires=wire_clase)# |1> si label=1
            elif tipo == "multilabel":
                # Cada elemento del label corresponde a un qubit
                label_vector = labels[i]  # arreglo de 0 y 1
                start_wire = qubits_qram + len(datos)
                for j, val in enumerate(label_vector):
                    wire_clase = start_wire + j
                    if val == 1:
                        qml.ctrl(qml.RY, control=controls)(np.pi, wires=wire_clase) # |1> si label=1
                        qml.DepolarizingChannel(0.02, wires=wire_clase)

        # Control en 0 → flip antes y después
        for k, a in enumerate(addr):
            if a == 0:
                qml.PauliX(wires=k)
                qml.DepolarizingChannel(0.02, wires=k)

def init_data_mod_noise(dataset):
    q_qram = len(dataset)
    for i, datos in enumerate(dataset):
        for j, val in enumerate(datos):
            # Escalamos valor [0,1] a ángulo [0,π]
            theta = val * np.pi
            qml.ctrl(qml.RY, control = i)(theta, wires = q_qram + j)
            qml.DepolarizingChannel(0.02, wires=q_qram + j)


def init_data2_noise(gray, qubits_qram, dataset, tipo, labels):
    for k in range(qubits_qram):
        qml.PauliX(wires=k)
        qml.DepolarizingChannel(0.02, wires=k)
    g_prev = 0
    for i, datos in enumerate(dataset):
        addr = [int(b) for b in gray[i]]
        controls = range(qubits_qram)
        g_curr = i ^ (i >> 1)
        mask = g_prev ^ g_curr
        lsb_pos = (mask.bit_length() - 1)
        g_prev = g_curr
        # Control en 0 → flip antes y después
        if lsb_pos >= 0:
            qml.PauliX(wires=lsb_pos)
            qml.DepolarizingChannel(0.02, wires=lsb_pos)

        for j, val in enumerate(datos):
            # Escalamos valor [0,1] a ángulo [0,π]
            theta = val * np.pi
            qml.ctrl(qml.RY, control=controls)(theta, wires=qubits_qram + j)
            qml.DepolarizingChannel(0.02, wires=qubits_qram + j)

        if labels is not None:
            if tipo == "biclase":
                # Un qubit de clase por patrón
                label = labels[i]  # 0 o 1
                wire_clase = qubits_qram + len(datos)  # siguiente qubit después de los datos
                if label == 1:
                    qml.ctrl(qml.RY, control=controls)(np.pi, wires=wire_clase) # |1> si label=1
                    qml.DepolarizingChannel(0.02, wires=wire_clase)
            elif tipo == "multilabel":
                # Cada elemento del label corresponde a un qubit
                label_vector = labels[i]  # arreglo de 0 y 1
                start_wire = qubits_qram + len(datos)
                for j, val in enumerate(label_vector):
                    wire_clase = start_wire + j
                    if val == 1:
                        qml.ctrl(qml.RY, control=controls)(np.pi, wires=wire_clase) # |1> si label=1
                        qml.DepolarizingChannel(0.02, wires=wire_clase)


    # # Control en 0 → flip antes y después
    addr = [int(b) for b in gray[len(dataset)-1]]
    for k, a in enumerate(addr):
        if a == 0:
            qml.PauliX(wires=len(addr)-k-1)
            qml.DepolarizingChannel(0.02, wires=len(addr)-k-1)

