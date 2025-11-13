"""
Ejemplo mínimo de entrenamiento de una QCNN variacional usando el módulo
`quantum_information.parametrics.qcnn`.

Entrenamos una red sencilla sobre un subconjunto binario del dataset `digits`
de scikit-learn (clases 0 y 1), aplicando una reducción de dimensionalidad a
8 componentes para codificarlas mediante `AmplitudeEmbedding` en 3 qubits.
El circuito consta de
una única capa variacional definida en este mismo archivo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from quantum_information.circuits.parametric import QCNNCircuitBuilder
from quantum_information.parametrics.qcnn import QCNNVariationalModel
from quantum_information.utils.params import init_params
from quantum_information.utils.utils import ensure_directory, save_pickle


def simple_layer(_: Sequence[str] | None, params, qubits):
    """
    Capa variacional mínima utilizada en el ejemplo.

    - Aplica rotaciones RY parametrizadas a cada qubit.
    - Genera entrelazamiento con una cadena de compuertas CNOT.

    ``total_params`` debe reflejar únicamente los parámetros empleados en las
    rotaciones (uno por qubit en este caso).
    """
    for idx, wire in enumerate(qubits):
        qml.RY(params[idx], wires=wire)

    for control, target in zip(qubits, qubits[1:]):
        qml.CNOT(wires=[control, target])
    # Conecta el último con el primero para cerrar el anillo.
    if len(qubits) > 2:
        qml.CNOT(wires=[qubits[-1], qubits[0]])


def prepare_data(num_qubits: int, test_size: float = 0.25, random_state: int = 42):
    """
    Carga un subconjunto del dataset digits y lo adapta para `AmplitudeEmbedding`.
    """
    digits = load_digits()
    # Usamos únicamente las clases 0 y 1 para mantener el ejemplo ligero.
    mask = digits.target < 2
    X = digits.data[mask]
    y = digits.target[mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    components = 2 ** num_qubits
    pca = PCA(n_components=components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    # Normalizamos cada vector para usar AmplitudeEmbedding.
    def normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec if norm == 0 else vec / norm

    X_normalized = np.array([normalize(sample) for sample in X_pca])

    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=test_size, random_state=random_state, stratify=y
        )

    num_classes = len(np.unique(y_train))
    y_train_oh = np.eye(num_classes)[y_train]
    y_test_oh = np.eye(num_classes)[y_test]

    # Convertimos a arreglos de autograd (pnp) para la optimización.
    return (
        pnp.array(X_train, requires_grad=False),
        pnp.array(y_train_oh, requires_grad=False),
        pnp.array(X_test, requires_grad=False),
        pnp.array(y_test_oh, requires_grad=False),
        num_classes,
    )


def main():
    n_qubits = 3
    target_wires = [2]  # Medimos el último qubit para obtener probabilidades.
    layers = [simple_layer]
    tip_rot = [None]
    qubits = [list(range(n_qubits))]
    total_params = [len(qubits[0])]

    X_train, y_train, X_test, y_test, num_classes = prepare_data(n_qubits)

    builder = QCNNCircuitBuilder(
        n_qubits=n_qubits,
        target_wires=target_wires,
        embedding_type="amplitude",
        rotation_angle="Y",
    )

    model = QCNNVariationalModel(
        prob_clases=target_wires,
        rot_angle="Y",
        embedding_type="amplitude",
        n_qubit=n_qubits,
        layers=layers,
        tip_rot=tip_rot,
        qubits=qubits,
    )

    params = init_params(sum(total_params), rng=0.1)
    opt = qml.AdamOptimizer(stepsize=0.3)

    def cost_fn(param_vector):
        # Calcula el costo promedio en el conjunto de entrenamiento.
        losses = []
        for features, label in zip(X_train, y_train):
            # El método loss espera lotes completos; usamos listas de un elemento.
            loss = model.loss(
                [features],
                [label],
                param_vector,
                total_params,
                loss_variant="binary",
            )
            losses.append(loss)
        return pnp.mean(pnp.stack(losses))

    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        params, current_loss = opt.step_and_cost(cost_fn, params)
        print(f"Epoch {epoch:02d} | Loss: {float(current_loss):.4f}")

    # Evaluación simple sobre el conjunto de prueba.
    qnode = builder.build_qnode(layers, tip_rot, qubits)
    correct = 0
    for features, label in zip(X_test, y_test):
        probs = qnode(features, params, total_params)
        pred = int(pnp.argmax(probs))
        gold = int(pnp.argmax(label))
        if pred == gold:
            correct += 1
    accuracy = correct / len(X_test)
    print(f"Test accuracy: {accuracy:.3f}")

    # Guardamos parámetros entrenados como referencia.
    param_path = ensure_directory("examples/variacionales_output", "qcnn_params.pkl")
    save_pickle(params, param_path, create_parents=True)
    print(f"Parámetros guardados en: {param_path}")


if __name__ == "__main__":
    main()

