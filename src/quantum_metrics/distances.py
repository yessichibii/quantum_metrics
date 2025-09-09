import pennylane as qml
import numpy as np

def trace_distance(rho, sigma):
    """Calcula la distancia traza entre dos matrices densidad."""
    diff = rho - sigma
    eigvals = np.linalg.eigvals(diff @ diff.conj().T)
    return 0.5 * np.sum(np.sqrt(np.abs(eigvals)))

def fidelity(rho, sigma):
    """Calcula la fidelidad cuántica entre dos matrices densidad."""
    sqrt_rho = scipy.linalg.sqrtm(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho
    return np.real(np.trace(scipy.linalg.sqrtm(inner)) ** 2)


def amplitude_distance(train, test):
    state_vector, gray, qubits_qram, qubits_dato = init_qram(train)

    qml.AmplitudeEmbedding(state_vector, wires=range(qubits_qram), normalize=True)
    init_data(gray,qubits_qram,dataset)
    distance(qubits_qram,test)
    

def h_amplitude_distance(train, test):
    state_vector, gray, qubits_qram, qubits_dato = init_qram(train)
    n_totales = qubits_qram + qubits_dato

    dev = qml.device("default.qubit", wires=n_totales)

    @qml.qnode(dev)
    def circuit():
        qml.AmplitudeEmbedding(state_vector, wires=range(qubits_qram), normalize=True)
        init_data(gray,qubits_qram,dataset)
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

# Inicializar QRAM
def init_qram(dataset):
    m, qubits_dato = dataset.shape
    qubits_qram= int(np.ceil(np.log2(m)))
    
    # Generamos Gray code de num_qubits bits
    gray = gray_code_inverso(qubits_qram)
    # Vector de amplitudes de dimensión 2**n
    state_vector = np.zeros(2**qubits_qram)

    # Ponemos 1 en las primeras m posiciones de Gray code
    for i in range(m):
        idx = int(gray[i], 2)   # posición en decimal
        state_vector[idx] = 1
    # Normalizamos
    state_vector = state_vector / np.linalg.norm(state_vector)
    return state_vector, gray, qubits_qram, qubits_dato

def init_data(gray,qubits_qram,dataset):
    for i, datos in enumerate(dataset):
        addr = [int(b) for b in gray[i]]
        for j, val in enumerate(datos):
            # Escalamos valor [0,1] a ángulo [0,π]
            theta = val * np.pi
            controls = []
            for k, a in enumerate(addr):
                if a == 1:
                    controls.append(k)
                else:
                    # Control en 0 → flip antes y después
                    qml.PauliX(wires=k)
                    controls.append(k)

            qml.ctrl(qml.RY, control=controls)(theta, wires=qubits_qram + j)

            for k, a in enumerate(addr):
                if a == 0:
                    qml.PauliX(wires=k)

def distance(qubits_qram,test):
    for j, val in enumerate(test):
        # Escalamos valor [0,1] a ángulo [0,π]
        theta = -1 * val * np.pi
        # Aplicamos RY inverso en los qubits de dato
        qml.RY(theta, wires=qubits_qram + j)



# Probamos

# Ejemplo: dataset con 4 patrones de 2 características cada uno
dataset = np.array([
    [1, 0],
    [0, 1],
    [1, 1],
    [0.5, 0.5]
], requires_grad=False)
test = [0.5,0.5]


state = penny_amplitude_distance(dataset,test)
print("Estado QRAM:\n", state)
