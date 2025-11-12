import pennylane as qml
from qiskit_aer.noise import NoiseModel
import numpy as np

from quantum_information.metrics.distances import mba_distance

def biclase(train, test, labels=None, codigo="gray", noise = 0.0, backend = None, result = "probs", shots = 1024):
    """
    Ejecuta un circuito QRAM completo y retorna el estado final del sistema.

    Parámetros
    ----------
    train : array-like
        Conjunto de datos de entrenamiento, cada fila representa un patrón.
    test : array-like
        Vector de prueba para calcular la distancia.
    labels : list, int o None, opcional
        Etiquetas asociadas a los datos de entrenamiento. Puede ser None si no se usan.
        En caso de biclase y multiclase retorna int
    codigo : str, opcional default gray
        Tipo de codificación para las direcciones en el QRAM:
        - "binario" : direcciones codificadas en binario
        - "gray" : drecciones codificadoas en codigo gray.
        - "diag" : solo las diagonales.
    noise : float, opcional
        Nivel de ruido para aplicar canales de depolarización. Por defecto 0.0.
    backend : backend o None, opcional
        Backend de Qiskit para simular el circuito con ruido. Si es None, se usa simulación ideal.
    result : str, default probs
        Tipo de respuesta a retornar: probs, state
    shots : int, opcional
        Numero de ejecuaciones, por defecto 1024, solo si se usa backend con ruido.

    Retorno
    -------
    np.ndarray
        - Probabilidades de clase si result es "probs"
        - Vector de estado final del sistema cuántico después de aplicar el clasificador biclase.

    Descripción
    -----------
    1. Inicializa el QRAM y calcula el número total de qubits necesarios
       (qubits de dirección + qubits de datos + qubits de etiquetas, si existen).
    2. Crea un dispositivo cuántico simulado (`default.qubit`) con el número total de qubits.
    3. Define un QNode que:
       - Aplica codificación de amplitud de los datos de entrenamiento.
       - Inserta los datos y etiquetas en el circuito (`encode_data`), con posible ruido.
       - Calcula la distancia con el patrón de prueba (`distance`).
       - Retorna el estado final del sistema.
    4. Ejecuta el QNode y retorna el vector de estado resultante.
    """
    m, qubits_dato = train.shape
    qubits_qram= int(np.ceil(np.log2(m)))
    n_totales = qubits_qram + qubits_dato
    n_totales += 2 
            
    if backend is not None:
        backend = backend
        noise_model = NoiseModel.from_backend(backend)
        dev = qml.device(
            "qiskit.aer",
            wires=n_totales,
            backend="qasm_simulator",
            noise_model=noise_model,
            shots=shots 
        )
    else:
        dev = qml.device("default.qubit", wires=n_totales)

    @qml.qnode(dev)
    def circuit():
        mba_distance(train, test, labels=labels, codigo=codigo, noise=noise)

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

    return circuit()

def asociativo(train, test, labels=None, noise = 0.0, backend = None, result = "probs", shots = 1024):
    """
    Ejecuta un circuito QRAM completo y retorna el estado final del sistema.

    Parámetros
    ----------
    train : array-like
        Conjunto de datos de entrenamiento, cada fila representa un patrón.
    test : array-like
        Vector de prueba para calcular la distancia.
    labels : list, int o None, opcional
        Etiquetas asociadas a los datos de entrenamiento. Puede ser None si no se usan.
        En caso de biclase y multiclase retorna int
    noise : float, opcional
        Nivel de ruido para aplicar canales de depolarización. Por defecto 0.0.
    backend : backend o None, opcional
        Backend de Qiskit para simular el circuito con ruido. Si es None, se usa simulación ideal.
    result : str, default probs
        Tipo de respuesta a retornar: probs, state
    shots : int, opcional
        Numero de ejecuaciones, por defecto 1024, solo si se usa backend con ruido.

    Retorno
    -------
    np.ndarray
        - Probabilidades de clase si result es "probs"
        - Vector de estado final del sistema cuántico después de aplicar el clasificador biclase.

    Descripción
    -----------
    1. Inicializa el QRAM y calcula el número total de qubits necesarios
       (qubits de dirección + qubits de datos + qubits de etiquetas, si existen).
    2. Crea un dispositivo cuántico simulado (`default.qubit`) con el número total de qubits.
    3. Define un QNode que:
       - Aplica codificación de amplitud de los datos de entrenamiento.
       - Inserta los datos y etiquetas en el circuito (`encode_data`), con posible ruido.
       - Calcula la distancia con el patrón de prueba (`distance`).
       - Retorna el estado final del sistema.
    4. Ejecuta el QNode y retorna el vector de estado resultante.
    """
    qubits_qram, qubits_dato = train.shape
    n_totales = qubits_qram + qubits_dato
            
    if backend is not None:
        backend = backend
        noise_model = NoiseModel.from_backend(backend)
        dev = qml.device(
            "qiskit.aer",
            wires=n_totales,
            backend="qasm_simulator",
            noise_model=noise_model,
            shots=shots
        )
    else:
        dev = qml.device("default.qubit", wires=n_totales)

    @qml.qnode(dev)
    def circuit():
        mba_distance(train, test, tipo="multiclase", labels=labels, codigo="diag", noise=noise)

        for i in range(qubits_dato):
            for j in range(qubits_qram):
                qml.ctrl(qml.RY, control=qubits_qram+i)(np.pi/qubits_dato, wires=j)
                if noise > 0:
                    qml.DepolarizingChannel(noise, wires = j)

        if result == "probs":
            if noise > 0:
                qml.AmplitudeDamping(2*noise, wires=0)
            return qml.probs(wires=range(qubits_qram))
        return qml.state()

    return circuit()

def multilabel(train, test, labels=None, codigo="gray", noise = 0.0, backend = None, result = "probs", shots = 1024):
    """
    Ejecuta un circuito QRAM completo y retorna el estado final del sistema.

    Parámetros
    ----------
    train : array-like
        Conjunto de datos de entrenamiento, cada fila representa un patrón.
    test : array-like
        Vector de prueba para calcular la distancia.
    labels : list, int o None, opcional
        Etiquetas asociadas a los datos de entrenamiento. Puede ser None si no se usan.
        En caso de biclase y multiclase retorna int
    codigo : str, opcional default gray
        Tipo de codificación para las direcciones en el QRAM:
        - "binario" : direcciones codificadas en binario
        - "gray" : drecciones codificadoas en codigo gray.
        - "diag" : solo las diagonales.
    noise : float, opcional
        Nivel de ruido para aplicar canales de depolarización. Por defecto 0.0.
    backend : backend o None, opcional
        Backend de Qiskit para simular el circuito con ruido. Si es None, se usa simulación ideal.
    result : str, default probs
        Tipo de respuesta a retornar: probs, state
    shots : int, opcional
        Numero de ejecuaciones, por defecto 1024, solo si se usa backend con ruido.

    Retorno
    -------
    np.ndarray
        - Probabilidades de clase si result es "probs"
        - Vector de estado final del sistema cuántico después de aplicar el clasificador biclase.

    Descripción
    -----------
    1. Inicializa el QRAM y calcula el número total de qubits necesarios
       (qubits de dirección + qubits de datos + qubits de etiquetas, si existen).
    2. Crea un dispositivo cuántico simulado (`default.qubit`) con el número total de qubits.
    3. Define un QNode que:
       - Aplica codificación de amplitud de los datos de entrenamiento.
       - Inserta los datos y etiquetas en el circuito (`encode_data`), con posible ruido.
       - Calcula la distancia con el patrón de prueba (`distance`).
       - Retorna el estado final del sistema.
    4. Ejecuta el QNode y retorna el vector de estado resultante.
    """
    m, qubits_dato = train.shape
    qubits_qram= int(np.ceil(np.log2(m)))
    n_totales = qubits_qram + qubits_dato
    n_totales += 1

    if labels is not None:
        n_totales += len(labels[0])
            
    if backend is not None:
        backend = backend
        noise_model = NoiseModel.from_backend(backend)
        # Dispositivo PennyLane usando Qiskit Aer con ruido
        dev = qml.device(
            "qiskit.aer",
            wires=n_totales,
            backend="qasm_simulator",
            noise_model=noise_model,
            shots=shots  # usar shots para evitar matrices de densidad enormes
        )
    else:
        dev = qml.device("default.qubit", wires=n_totales)

    @qml.qnode(dev)
    def circuit():
        mba_distance(train, test, tipo="multilabel", labels=labels, codigo=codigo, noise=noise)

        for i in range(qubits_dato):
            qml.ctrl(qml.RY, control=qubits_qram+i)(np.pi/qubits_dato, wires=n_totales-1)
            if noise > 0:
                qml.DepolarizingChannel(noise, wires = n_totales-1)

        if labels is not None:
            for j in range(len(labels[0])):
                qml.CNOT(wires=[n_totales-1, qubits_qram+qubits_dato+j])
                if noise > 0:
                    qml.DepolarizingChannel(noise, wires = qubits_qram+qubits_dato+j)


        if result == "probs":
            if noise > 0:
                qml.AmplitudeDamping(2*noise, wires=0)
            return qml.probs(wires=range(qubits_qram+qubits_dato,n_totales-1))
        return qml.state()

    return circuit()