import pennylane as qml
from pennylane import numpy as pnp
from qiskit_aer.noise import NoiseModel
import numpy as np
from .initialize import qram_initialize, encode_data

# def state_mba_distance(train, test, tipo="biclase", labels=None, codigo="gray", noise = 0.0, backend = None, result = "probs", shots = 1024):
#     """
#     Ejecuta un circuito QRAM completo y retorna el estado final del sistema.

#     Parámetros
#     ----------
#     train : array-like
#         Conjunto de datos de entrenamiento, cada fila representa un patrón.
#     test : array-like
#         Vector de prueba para calcular la distancia.
#     tipo : str, opcional default biclase
#         Tipo de problema:
#         - "biclase" : problema con solo dos clases (una etiqueta por patrón).
#         - "multiclase" : problema con n clases (una etiqueta por patrón).
#         - "multilabel" : problema con múltiples etiquetas binarias por patrón.
#     labels : list, int o None, opcional
#         Etiquetas asociadas a los datos de entrenamiento. Puede ser None si no se usan.
#         En caso de biclase y multiclase retorna int
#     codigo : str, opcional default gray
#         Tipo de codificación para las direcciones en el QRAM:
#         - "binario" : direcciones codificadas en binario
#         - "gray" : drecciones codificadoas en codigo gray.
#         - "diag" : solo las diagonales.
#     noise : float, opcional
#         Nivel de ruido para aplicar canales de depolarización. Por defecto 0.0.
#     backend : backend o None, opcional
#         Backend de Qiskit para simular el circuito con ruido. Si es None, se usa simulación ideal.
#     result : str, default probs
#         Tipo de respuesta a retornar: probs, state
#     shots : int, opcional
#         Numero de ejecuaciones, por defecto 1024, solo si se usa backend con ruido.

#     Retorno
#     -------
#     np.ndarray
#         - Probabilidades de medida de las etiquetas si result es "probs"
#         - Vector de estado final del sistema cuántico después de aplicar la codificación
#         y el cálculo de distancia.

#     Descripción
#     -----------
#     1. Inicializa el QRAM y calcula el número total de qubits necesarios
#        (qubits de dirección + qubits de datos + qubits de etiquetas, si existen).
#     2. Crea un dispositivo cuántico simulado (`default.qubit`) con el número total de qubits.
#     3. Define un QNode que:
#        - Aplica codificación de amplitud de los datos de entrenamiento.
#        - Inserta los datos y etiquetas en el circuito (`encode_data`), con posible ruido.
#        - Calcula la distancia con el patrón de prueba (`distance`).
#        - Retorna el estado final del sistema.
#     4. Ejecuta el QNode y retorna el vector de estado resultante.
#     """
#     state_vector, direcciones, qubits_qram, qubits_data = qram_initialize(train, codigo=codigo)
#     n_total = qubits_qram + qubits_data
    
#     if labels is not None:
#         match tipo:
#             case "multiclase":
#                 n_total += int(np.ceil(np.log2(max(labels))))
#             case "multilabel":
#                 n_total += len(labels[0])
#             case "biclase":
#                 n_total += 1
#             case _:
#                 raise ValueError(f'Tipo "{tipo}" no válido')
            
#     if backend is not None:
#         backend = backend
#         noise_model = NoiseModel.from_backend(backend)
#         # Dispositivo PennyLane usando Qiskit Aer con ruido
#         dev = qml.device(
#             "qiskit.aer",
#             wires=n_total,
#             backend="qasm_simulator",
#             noise_model=noise_model,
#             shots=shots  # usar shots para evitar matrices de densidad enormes
#         )
#     else:
#         dev = qml.device("default.qubit", wires=n_total)

#     @qml.qnode(dev)
#     def circuit():
#         qml.AmplitudeEmbedding(state_vector, wires=range(qubits_qram), normalize=True)
#         encode_data(direcciones, qubits_qram, train, tipo, labels, noise=noise, codigo=codigo)
#         distance(qubits_qram, test, noise=noise)
#         if result == "probs":
#             if noise > 0:
#                 qml.AmplitudeDamping(2*noise, wires=0)
#             return qml.probs(wires=range(qubits_qram+qubits_data, n_total-1))
#         return qml.state()

#     return circuit()

def mba_distance(train, test, tipo="biclase", labels=None, codigo="gray", noise=0.0):
    """
    Construye un circuito QRAM con la inicializacion de datos y el calculo de la distancia, retorna el circuito para seguir operando

    Parámetros
    ----------
    train : array-like
        Conjunto de datos de entrenamiento, cada fila representa un patrón.
    test : array-like
        Vector de prueba para el cálculo de distancia.
    tipo : str, opcional default biclase
        Tipo de problema:
        - "biclase" : problema con solo dos clases (una etiqueta por patrón).
        - "multiclase" : problema con n clases (una etiqueta por patrón).
        - "multilabel" : problema con múltiples etiquetas binarias por patrón.
    labels : list, int o None, opcional
        Etiquetas asociadas a los datos de entrenamiento. Puede ser None si no se usan.
        En caso de biclase y multiclase retorna int
    noise : float, opcional
        Nivel de ruido para aplicar canales de depolarización en el circuito.
        Por defecto 0.0 (sin ruido).
    codigo : str, opcional default gray
        Tipo de codificación para las direcciones en el QRAM:
        - "binario" : direcciones codificadas en binario
        - "gray" : drecciones codificadoas en codigo gray.
        - "diag" : solo las diagonales.

    Descripción
    -----------
    1. Inicializa un QRAM para el número de patrones del conjunto de entrenamiento (`qram_initialize`).
    2. Aplica codificación de amplitud sobre los qubits de dirección para generar las direcciones equiprobables.
    3. Inserta los datos de entrenamiento y/o etiquetas en el circuito (`encode_data`),
       opcionalmente aplicando ruido.
    4. Aplica rotaciones inversas sobre los qubits de datos para calcular la distancia
       con el vector de prueba (`distance`).

    Nota
    ----
    Esta función construye el circuito y aplica las operaciones cuánticas, pero no
    retorna el estado final del sistema.
    """
    state_vector, direcciones, qubits_qram, _ = qram_initialize(train, codigo=codigo)
    # print(f"n_totales: {direcciones}")
    # print(f"qubits_qram: {qubits_qram}")
    qml.AmplitudeEmbedding(state_vector, wires=range(qubits_qram), normalize=True)
    encode_data(direcciones, qubits_qram, train, tipo, labels, noise=noise, codigo=codigo)
    distance(qubits_qram, test, noise=noise)

def distance(qubits_qram, test, noise=0.0):
    """
    Aplica la codificación inversa (distancia cuántica) sobre los qubits de datos.

    Parámetros
    ----------
    qubits_qram : int
        Número de qubits de dirección.
    test : np.ndarray or autograd array
        Vector de entrada de prueba, con valores en [0, 1] o parámetros entrenables.
        Puede ser un array de numpy regular o un array de autograd para diferenciación.
    noise : float, opcional
        Intensidad del canal de ruido (0.0 desactiva el ruido).
    """

    for j, val in enumerate(test):
        # Escalamos valor [0,1] a ángulo [0,π]
        # CRÍTICO: Usar anp.pi en lugar de np.pi para mantener diferenciación
        # cuando val es un array de autograd
        theta = -1 * val * np.pi
        # Aplicamos RY inverso en los qubits de dato
        # qml.RY con interface="autograd" debería manejar arrays de autograd automáticamente
        qml.RY(theta, wires=qubits_qram + j)
        if noise > 0:
            qml.DepolarizingChannel(noise, wires = qubits_qram + j)



def weight_params(qubits_qram, params, noise=0.0):
    """
    Aplica la codificación inversa (distancia cuántica) sobre los qubits de datos.

    Parámetros
    ----------
    qubits_qram : int
        Número de qubits de dirección.
    test : np.ndarray or autograd array
        Vector de entrada de prueba, con valores en [0, 1] o parámetros entrenables.
        Puede ser un array de numpy regular o un array de autograd para diferenciación.
    noise : float, opcional
        Intensidad del canal de ruido (0.0 desactiva el ruido).
    """
    for j, val in enumerate(params):
        # Escalamos valor [0,1] a ángulo [0,π]
        # CRÍTICO: Usar anp.pi en lugar de np.pi para mantener diferenciación
        # cuando val es un array de autograd
        theta = val * pnp.pi
        # Aplicamos RY inverso en los qubits de dato
        # qml.RY con interface="autograd" debería manejar arrays de autograd automáticamente
        print(f"theta: {theta} - qubits_qram + j: {qubits_qram + j}")
        qml.RY(theta, wires=qubits_qram + j)
        if noise > 0:
            qml.DepolarizingChannel(noise, wires = qubits_qram + j)

