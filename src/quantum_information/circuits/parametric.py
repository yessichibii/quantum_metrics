import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding, AngleEmbedding
from typing import Sequence


def convolucionRot(tipRot, params, qubits):
    """
    Aplica una etapa de convolución sobre el conjunto completo de qubits.

    Cada iteración recorre las rotaciones y compuertas controladas indicadas en
    ``tipRot``. Para cada tipo de rotación se asigna un bloque de parámetros
    consecutivo en ``params``.

    Parámetros
    ----------
    tipRot:
        Lista de etiquetas que determinan el tipo de rotación a aplicar. Debe
        contener cuatro elementos que combinen rotaciones simples y
        controladas, por ejemplo ``['X', 'Y', 'Z', 'CNOT']``.
    params:
        Vector de parámetros que se asigna secuencialmente a las compuertas.
    qubits:
        Lista con los índices de los qubits que participan en la capa.
    """
    for j in (0, 2):
        for ind, qubit in enumerate(qubits):
            if tipRot[j] == "X":
                qml.RX(params[ind + j * len(qubits)], wires=qubit)
            if tipRot[j] == "Y":
                qml.RY(params[ind + j * len(qubits)], wires=qubit)
            if tipRot[j] == "Z":
                qml.RZ(params[ind + j * len(qubits)], wires=qubit)
        if j == 2:
            qubits.reverse()
        for ind, qubit in enumerate(qubits):
            if tipRot[j + 1] == "X":
                qml.CRX(
                    params[ind + (j + 1) * len(qubits)],
                    wires=[qubit, qubits[(ind + 1) % len(qubits)]],
                )
            if tipRot[j + 1] == "Y":
                qml.CRY(
                    params[ind + (j + 1) * len(qubits)],
                    wires=[qubit, qubits[(ind + 1) % len(qubits)]],
                )
            if tipRot[j + 1] == "Z":
                qml.CRZ(
                    params[ind + (j + 1) * len(qubits)],
                    wires=[qubit, qubits[(ind + 1) % len(qubits)]],
                )
            if tipRot[j + 1] == "CNOT":
                qml.CNOT(
                    params[ind + (j + 1) * len(qubits)],
                    wires=[qubit, qubits[(ind + 1) % len(qubits)]],
                )

def reducirTodos(tipRot, params, qubits):
    """
    Ejecuta una etapa de reducción eliminando gradualmente qubits de control.

    El primer qubit del arreglo actúa como control y finalmente se elimina,
    transfiriendo la información mediante compuertas controladas y una
    compuerta ``U3`` final.

    Parámetros
    ----------
    tipRot:
        Lista que define la secuencia de rotaciones controladas.
    params:
        Vector de parámetros aplicado a las compuertas.
    qubits:
        Lista de qubits afectados. El primer elemento se elimina al finalizar
        la etapa.
    """
    for j in (0, 1):
        if j == 1:
            qml.U3(
                params[len(qubits) - 1],
                params[len(qubits) - 2],
                params[len(qubits) - 3],
                wires=qubits[0],
            )
            qubits.reverse()
        for ind in range(0, len(qubits) - 1):
            if tipRot[j] == "X":
                qml.CRX(
                    params[ind + j * (len(qubits) - 1)],
                    wires=[qubits[j * (len(qubits) - 1)], qubits[ind + 1 - j]],
                )
            if tipRot[j] == "Y":
                qml.CRY(
                    params[ind + j * (len(qubits) - 1)],
                    wires=[qubits[j * (len(qubits) - 1)], qubits[ind + 1 - j]],
                )
            if tipRot[j] == "Z":
                qml.CRZ(
                    params[ind + j * (len(qubits) - 1)],
                    wires=[qubits[j * (len(qubits) - 1)], qubits[ind + 1 - j]],
                )
            if tipRot[j] == "CNOT":
                qml.CNOT(
                    params[ind + j * (len(qubits) - 1)],
                    wires=[qubits[j * (len(qubits) - 1)], qubits[ind + 1 - j]],
                )


def clasificacionVariacional(tipRot, params, qubits):
    """
    Construye una etapa variacional completa orientada a clasificación.

    La secuencia se compone de una primera capa local (``U3`` o ``Rot``),
    seguida de pares de compuertas controladas y simples aplicadas en barrido
    directo e inverso, concluyendo con otra capa local.

    Parámetros
    ----------
    tipRot:
        Lista de tres etiquetas que describen la compuerta inicial, la
        compuerta controlada y la compuerta simple a emplear.
    params:
        Vector de parámetros consumido en el orden de aplicación de compuertas.
    qubits:
        Lista de qubits sobre los que actúa la etapa.
    """
    cont = 0
    for ind, qubit in enumerate(qubits):
        if tipRot[0] == "U3":
            qml.U3(
                params[ind + cont],
                params[ind + cont + 1],
                params[ind + cont + 2],
                wires=qubit,
            )
        if tipRot[0] == "Rot":
            qml.Rot(
                params[ind + cont],
                params[ind + cont + 1],
                params[ind + cont + 2],
                wires=qubit,
            )
        cont += 2
    intervalo = range(0, len(qubits) - 1)
    cont = 3 * len(qubits)
    for j in [1, -1]:
        if j == -1:
            intervalo = range(len(qubits) - 1, 0, -1)
        for ind in intervalo:
            if tipRot[1] == "X":
                qml.CRX(params[cont], wires=[qubits[ind], qubits[ind + 1 * j]])
            if tipRot[1] == "Y":
                qml.CRY(params[cont], wires=[qubits[ind], qubits[ind + 1 * j]])
            if tipRot[1] == "Z":
                qml.CRZ(params[cont], wires=[qubits[ind], qubits[ind + 1 * j]])
            if tipRot[1] == "CNOT":
                qml.CNOT(wires=[qubits[ind], qubits[ind + 1 * j]])
                cont -= 1
            cont += 1
            if tipRot[2] == "X":
                qml.RX(params[cont], wires=qubits[ind + 1 * j])
            if tipRot[2] == "Y":
                qml.RY(params[cont], wires=qubits[ind + 1 * j])
            if tipRot[2] == "Z":
                qml.RZ(params[cont], wires=qubits[ind + 1 * j])
            if tipRot[2] == "NOT":
                qml.PauliX(wires=qubits[ind + 1 * j])
                cont -= 1
            cont += 1

    for ind, qubit in enumerate(qubits):
        if tipRot[0] == "U3":
            qml.U3(
                params[ind + cont],
                params[ind + cont + 1],
                params[ind + cont + 2],
                wires=qubit,
            )
        if tipRot[0] == "Rot":
            qml.Rot(
                params[ind + cont],
                params[ind + cont + 1],
                params[ind + cont + 2],
                wires=qubit,
            )
        cont += 2


def circuito_variacional(params, total_params, qubits, tipRot, layers):
    """
    Ensambla secuencialmente las capas variacionales que conforman la QCNN.

    Parámetros
    ----------
    params:
        Vector plano que contiene todos los parámetros libres del circuito.
    total_params:
        Secuencia que indica cuántos parámetros consume cada capa.
    qubits:
        Lista de listas donde cada elemento describe los qubits utilizados por
        la capa correspondiente.
    tipRot:
        Lista de configuraciones de rotaciones asociadas a cada capa.
    layers:
        Lista de funciones (o callables) que se invocarán para construir cada
        capa.
    """
    inicio = 0
    for ind, layer in enumerate(layers):
        final = inicio + total_params[ind]
        layer(tipRot[ind], params[inicio:final], qubits[ind])
        inicio = final


class QCNNCircuitBuilder:
    """
    Construye circuitos variacionales QCNN basados en configuraciones dinámicas.

    La clase encapsula la creación del dispositivo y la lógica repetitiva de
    incrustación de datos, permitiendo obtener ``qnodes`` listos para evaluar
    probabilidades sobre subconjuntos de qubits.
    """

    def __init__(
        self,
        n_qubits: int,
        target_wires: Sequence[int],
        embedding_type: str = "amplitude",
        rotation_angle: str = "Y",
    ) -> None:
        """
        Inicializa el constructor del circuito.

        Parámetros
        ----------
        n_qubits:
            Número total de qubits disponibles en el circuito.
        target_wires:
            Qubits sobre los que se medirán las probabilidades.
        embedding_type:
            Tipo de codificación de datos. Valores válidos: ``"amplitude"`` o
            ``"angle"``.
        rotation_angle:
            Rotación elemental para ``AngleEmbedding`` cuando se utiliza esa
            codificación (por ejemplo ``"Y"``).
        """
        if embedding_type not in {"amplitude", "angle"}:
            raise ValueError(
                "embedding_type debe ser 'amplitude' o 'angle', "
                f"se recibió '{embedding_type}'."
            )
        self.n_qubits = n_qubits
        self.target_wires = target_wires
        self.embedding_type = embedding_type
        self.rotation_angle = rotation_angle
        self.device = qml.device("default.qubit", wires=n_qubits)

    def _embed(self, features):
        """
        Codifica un vector clásico en el registro cuántico.
        """
        if self.embedding_type == "amplitude":
            AmplitudeEmbedding(features, wires=range(self.n_qubits), normalize=True)
        else:
            AngleEmbedding(
                features,
                wires=range(self.n_qubits),
                rotation=self.rotation_angle,
            )

    def build_qnode(self, layers, tip_rot, qubits):
        """
        Construye un ``qnode`` configurado con las capas provistas.

        Parámetros
        ----------
        layers:
            Lista de funciones que definen cada etapa variacional.
        tip_rot:
            Lista de configuraciones de rotación por capa.
        qubits:
            Lista de listas con los qubits involucrados por capa.

        Retorna
        -------
        Callable
            QNode de PennyLane listo para evaluar el circuito.
        """

        @qml.qnode(self.device)
        def _qcnn(features, params, total_params):
            self._embed(features)
            circuito_variacional(params, total_params, qubits, tip_rot, layers)
            return qml.probs(wires=self.target_wires)

        return _qcnn

    def predict(self, features, params, total_params, layers, tip_rot, qubits):
        """
        Evalúa el circuito para un único vector de características.

        Retorna
        -------
        numpy.ndarray
            Probabilidades sobre ``target_wires``.
        """
        qnode = self.build_qnode(layers, tip_rot, qubits)
        return qnode(features, params, total_params)

    def batch_predict(self, batch, params, total_params, layers, tip_rot, qubits):
        """
        Evalúa el circuito para un lote de vectores de características.

        Retorna
        -------
        list[numpy.ndarray]
            Lista de probabilidades, una por elemento del lote.
        """
        qnode = self.build_qnode(layers, tip_rot, qubits)
        return [qnode(features, params, total_params) for features in batch]


def QCNN(
    X,
    params,
    total_params,
    qubits,
    tipRot,
    layers,
    prob_clases,
    rotAngle,
    embedding_type,
    n_qubit,
):
    """
    Función de compatibilidad que evalúa una QCNN con la configuración dada.

    Parámetros
    ----------
    X:
        Vector de características a codificar.
    params:
        Vector plano de parámetros libres.
    total_params:
        Secuencia que indica la cantidad de parámetros por capa.
    qubits, tipRot, layers:
        Configuración estructural del circuito.
    prob_clases:
        Qubits de lectura para el cálculo de probabilidades.
    rotAngle:
        Angulo elemental utilizado por ``AngleEmbedding`` si aplica.
    embedding_type:
        Tipo de codificación a emplear.
    n_qubit:
        Número total de qubits del circuito.

    Retorna
    -------
    numpy.ndarray
        Probabilidades calculadas sobre ``prob_clases``.
    """
    builder = QCNNCircuitBuilder(
        n_qubits=n_qubit,
        target_wires=prob_clases,
        embedding_type=embedding_type,
        rotation_angle=rotAngle,
    )
    return builder.predict(X, params, total_params, layers, tipRot, qubits)