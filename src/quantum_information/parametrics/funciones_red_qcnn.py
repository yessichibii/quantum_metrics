from __future__ import annotations

import itertools
from pathlib import Path
from typing import Callable, Sequence

from pennylane import numpy as np
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from quantum_information.circuits.parametric import QCNNCircuitBuilder
from quantum_information.utils.utils import (
    binary_cross_entropy_multilabel,
    binary_cross_entropy_positive,
    binary_cross_entropy_with_sigmoid,
    calculate_multilabel_accuracy,
    cargar_metricas,
    cargar_o_inicializar_parametros,
    guardar_objetos,
    guardar_parametros,
    guardar_predicciones,
    load_parametric_config,
)


def acutalizarObjetos(archivo, path_guardar, params):  # pragma: no cover - compatibilidad
    """
    Alias histórico que delega en :func:`guardar_parametros`.
    """
    guardar_parametros(archivo, path_guardar, params)


def cargarObjetos(archivo, path_guardar, total_params):  # pragma: no cover - compatibilidad
    """
    Alias histórico que delega en :func:`cargar_o_inicializar_parametros`.
    """
    return cargar_o_inicializar_parametros(archivo, path_guardar, total_params)


def guardarObjetos(archivo, path_guardar, params):  # pragma: no cover - compatibilidad
    """
    Alias histórico que delega en :func:`guardar_objetos`.
    """
    guardar_objetos(archivo, path_guardar, params)


def guardarPrediciones(path_guardar, algoritmo, archivo, datos, tipo):  # pragma: no cover - compatibilidad
    """
    Alias histórico que delega en :func:`guardar_predicciones`.
    """
    guardar_predicciones(path_guardar, algoritmo, archivo, datos, tipo)


def cargarAcc(archivo, path_guardar):  # pragma: no cover - compatibilidad
    """
    Alias histórico que delega en :func:`cargar_metricas`.
    """
    return cargar_metricas(archivo, path_guardar)


class QCNNVariationalModel:
    """
    Construye y evalúa una QCNN variacional a partir de una configuración específica.

    La clase encapsula la lógica recurrente para instanciar el circuito, obtener
    predicciones en lote y calcular métricas de desempeño usando distintas familias de
    funciones de costo basadas en entropía cruzada.
    """

    _LOSS_FUNCTIONS = {
        "binary": binary_cross_entropy_multilabel,
        "positive": binary_cross_entropy_positive,
        "sigmoid": binary_cross_entropy_with_sigmoid,
    }

    def __init__(
        self,
        prob_clases: Sequence[int],
        rot_angle: str,
        embedding_type: str,
        n_qubit: int,
        layers: Sequence,
        tip_rot: Sequence,
        qubits: Sequence[Sequence[int]],
    ) -> None:
        """
        Parámetros
        ----------
        prob_clases:
            Qubits objetivo sobre los que se calcularán las probabilidades de salida.
        rot_angle:
            Ángulo elemental utilizado en ``AngleEmbedding`` cuando corresponde.
        embedding_type:
            Tipo de codificación de datos (``"amplitude"`` o ``"angle"``).
        n_qubit:
            Número total de qubits disponibles en el circuito.
        layers:
            Secuencia de funciones que describen cada capa del circuito variacional.
        tip_rot:
            Configuraciones de rotaciones asociadas a cada capa.
        qubits:
            Arreglo de qubits por capa, en el mismo orden que ``layers``.
        """
        self.layers = layers
        self.tip_rot = tip_rot
        self.qubits = qubits
        self.builder = QCNNCircuitBuilder(
            n_qubits=n_qubit,
            target_wires=prob_clases,
            embedding_type=embedding_type,
            rotation_angle=rot_angle,
        )

    def predict_batch(
        self,
        batch: Sequence[Sequence[float]],
        params,
        total_params,
    ) -> list[np.ndarray]:
        """
        Evalúa el circuito para un lote de muestras y devuelve las probabilidades estimadas.

        Parámetros
        ----------
        batch:
            Lote de vectores clásicos listos para codificarse en el circuito.
        params:
            Vector plano con los parámetros entrenables del circuito.
        total_params:
            Desglose del número de parámetros por capa, según ``layers``.

        Retorna
        -------
        list[numpy.ndarray]
            Lista con las probabilidades para cada muestra del lote.
        """
        return self.builder.batch_predict(
            batch,
            params,
            total_params,
            self.layers,
            self.tip_rot,
            self.qubits,
        )

    def loss(
        self,
        batch: Sequence[Sequence[float]],
        labels: Sequence[Sequence[float]],
        params,
        total_params,
        loss_variant: str = "binary",
    ) -> float:
        """
        Calcula la pérdida de entrenamiento utilizando la variante de entropía cruzada indicada.

        Parámetros
        ----------
        batch:
            Lote de vectores clásicos para evaluar el circuito.
        labels:
            Etiquetas verdaderas asociadas a las muestras del lote.
        params, total_params:
            Parámetros del circuito en su formato plano y desglosado por capa.
        loss_variant:
            Identificador de la función de pérdida a utilizar. Valores válidos:
            ``"binary"``, ``"positive"`` o ``"sigmoid"``.

        Retorna
        -------
        float
            Valor medio de la pérdida calculada sobre el lote.
        """
        predictions = self.predict_batch(batch, params, total_params)
        loss_fn = self._select_loss_fn(loss_variant)
        return loss_fn(labels, predictions)

    def accuracy(
        self,
        batch: Sequence[Sequence[float]],
        labels: Sequence[Sequence[int]],
        params,
        total_params,
    ) -> tuple[np.ndarray, float]:
        """
        Evalúa la exactitud de la QCNN sobre un lote de datos.

        Parámetros
        ----------
        batch:
            Lote de vectores clásicos.
        labels:
            Etiquetas reales en formato one-hot.
        params, total_params:
            Parámetros del circuito.

        Retorna
        -------
        tuple[numpy.ndarray, float]
            Exactitud por clase y exactitud total para el lote evaluado.
        """
        predictions = self.predict_batch(batch, params, total_params)
        return calculate_multilabel_accuracy(predictions, labels)

    def _select_loss_fn(
        self, key: str
    ) -> Callable[[Sequence[Sequence[float]], Sequence[Sequence[float]]], float]:
        """
        Recupera la función de pérdida registrada para el identificador solicitado.

        Parámetros
        ----------
        key:
            Clave que identifica la función de pérdida requerida.

        Retorna
        -------
        Callable[[Sequence[Sequence[float]], Sequence[Sequence[float]]], float]
            Función de pérdida correspondiente a ``key``.
        """
        try:
            return self._LOSS_FUNCTIONS[key]
        except KeyError as exc:
            opciones = ", ".join(sorted(self._LOSS_FUNCTIONS))
            raise ValueError(
                f"loss_variant '{key}' no es válido. Opciones disponibles: {opciones}."
            ) from exc


def data_load_and_process(
    size: int,
    batch_size: int,
    input_images_path: str | Path,
    banco_train: str,
    banco_test: str,
):
    """
    Carga imágenes desde discos, aplica transformaciones básicas y genera lotes.

    Parámetros
    ----------
    size:
        Tamaño (alto y ancho) al que se escalarán las imágenes.
    batch_size:
        Número de muestras por lote al construir los `DataLoader`.
    input_images_path:
        Ruta base que contiene los bancos de entrenamiento y prueba.
    banco_train:
        Nombre de la carpeta con el conjunto de entrenamiento.
    banco_test:
        Nombre de la carpeta con el conjunto de prueba.

    Retorna
    -------
    tuple[list[numpy.ndarray], list[numpy.ndarray], list[numpy.ndarray], list[numpy.ndarray]]
        Listas con los lotes de imágenes y etiquetas codificadas en one-hot para
        entrenamiento y prueba, respectivamente.
    """
    base_path = Path(input_images_path)
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),  # Cambia el tamaño de la imagen a 64x64
            transforms.Grayscale(
                num_output_channels=1
            ),  # Convertir a escala de grises si no lo están
            transforms.ToTensor(),
        ]
    )
    train_set = datasets.ImageFolder(
        str(base_path / banco_train), transform=transform
    )
    test_set = datasets.ImageFolder(str(base_path / banco_test), transform=transform)
    print(f"Se cargaron {len(train_set)} elementos del conjunto TRAIN")
    print(f"Se cargaron {len(test_set)} elementos del conjunto TEST")

    # Obtener los nombres de las carpetas (clases)
    classes = train_set.classes
    print("Nombres de las clases: ", classes)

    clases_unicas = set()
    for i in classes:
        clases_unicas.update(i.split("+"))
    clases_unicas = list(clases_unicas)
    max_valor = len(clases_unicas)
    print("Nombre de las clases únicas: ", clases_unicas)

    multi_label = []
    for clase in classes:
        indices = [clases_unicas.index(nombre) for nombre in clase.split("+")]
        multi_label.append(indices)
    print("Relación de los índices de multiclase: ", multi_label)
    train_batch = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_batch = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    X_train = []
    Y_train = []
    for images, labels in train_batch:
        X_train.append(images.squeeze(0).numpy())
        one_hot = np.zeros((len(labels), max_valor), dtype=np.uint8)
        label = labels.tolist()
        for i, val in enumerate(label):
            one_hot[i, multi_label[val]] = 1
        Y_train.append(one_hot)

    X_test = []
    Y_test = []
    for images, labels in test_batch:
        X_test.append(images.squeeze(0).numpy())
        one_hot = np.zeros((len(labels), max_valor), dtype=np.uint8)
        label = labels.tolist()
        for i, val in enumerate(label):
            one_hot[i, multi_label[val]] = 1
        Y_test.append(one_hot)

    return X_train, X_test, Y_train, Y_test


def transform_data_load(
    X_train,
    X_test,
    n_qubit: int,
    batch_size: int,
    encoding: str = "amplitude",
):
    """
    Reduce dimensionalidad y normaliza los lotes para ajustarlos al número de qubits.

    Parámetros
    ----------
    X_train, X_test:
        Listas de lotes de imágenes provenientes de ``data_load_and_process``.
    n_qubit:
        Número de qubits disponibles en el circuito cuántico.
    batch_size:
        Tamaño deseado por lote tras la transformación.
    encoding:
        Tipo de codificación de datos (``"amplitude"`` o ``"angle"``).

    Retorna
    -------
    tuple[list[numpy.ndarray], list[numpy.ndarray]]
        Listas transformadas de entrenamiento y prueba, respectivamente.
    """
    pca = PCA(n_components=2**n_qubit) if encoding == "amplitude" else PCA(n_qubit)

    X_train_t = []
    for train in X_train:
        X_train_flat = train.reshape(train.shape[0], -1)
        if encoding == "amplitude":
            if X_train_flat.shape[1] <= 2**n_qubit:
                X_train_t.append(X_train_flat)
                continue
        if X_train_flat.shape[0] < 2**n_qubit:
            break
        train = pca.fit_transform(X_train_flat)
        if encoding == "angle":
            train = (train - train.min()) * (np.pi / (train.max() - train.min()))
        X_train_t.append(train)

    if len(X_train_t) < 1:
        X_train = np.array(list(itertools.chain(*X_train)))
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        train = pca.fit_transform(X_train_flat)
        for i in range(0, len(train), batch_size):
            X_train_t.append(train[i : i + batch_size])

    X_test_t = []
    for test in X_test:
        X_test_flat = test.reshape(test.shape[0], -1)
        if encoding == "amplitude":
            if X_test_flat.shape[1] <= 2**n_qubit:
                X_test_t.append(X_test_flat)
                continue
        if X_test_flat.shape[0] < 2**n_qubit:
            break
        test = pca.transform(X_test_flat)
        if encoding == "angle":
            test = (test - test.min()) * (np.pi / (test.max() - test.min()))
        X_test_t.append(test)

    if len(X_test_t) < 1:
        X_test = np.array(list(itertools.chain(*X_test)))
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        test = pca.fit_transform(X_test_flat)
        for i in range(0, len(test), batch_size):
            X_test_t.append(test[i : i + batch_size])

    return X_train_t, X_test_t


def cost(
    X,
    Y,
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
    Calcula la pérdida principal (entropía cruzada binaria) para una QCNN configurada.

    Parámetros
    ----------
    X:
        Lote de vectores de características listos para codificación cuántica.
    Y:
        Lote de etiquetas verdaderas en formato one-hot.
    params:
        Vector plano con todos los parámetros entrenables del circuito.
    total_params:
        Secuencia con el número de parámetros consumidos por cada capa.
    qubits:
        Lista de capas donde cada elemento describe los qubits involucrados.
    tipRot:
        Configuraciones de rotaciones por capa (secuencia paralela a ``layers``).
    layers:
        Secuencia de funciones que construyen cada etapa del circuito.
    prob_clases:
        Índices de los qubits medidos para obtener las probabilidades de clase.
    rotAngle:
        Ángulo elemental utilizado en ``AngleEmbedding`` cuando aplica.
    embedding_type:
        Tipo de codificación de datos (``"amplitude"`` o ``"angle"``).
    n_qubit:
        Número total de qubits del circuito.

    Retorna
    -------
    float
        Valor medio de la pérdida calculada sobre el lote ``X``.

    Nota
    ----
    La función se conserva como alias histórico y delega la lógica en
    :class:`QCNNVariationalModel` para evitar duplicación de código.
    """
    model = QCNNVariationalModel(
        prob_clases=prob_clases,
        rot_angle=rotAngle,
        embedding_type=embedding_type,
        n_qubit=n_qubit,
        layers=layers,
        tip_rot=tipRot,
        qubits=qubits,
    )
    return model.loss(X, Y, params, total_params, loss_variant="binary")

def cost2(
    X,
    Y,
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
    Calcula una variante de pérdida que solo considera clases positivas del objetivo.

    Parámetros
    ----------
    X, Y, params, total_params, qubits, tipRot, layers, prob_clases, rotAngle,
    embedding_type, n_qubit:
        Coinciden con los descritos en :func:`cost` y se reutilizan para evaluar la
        configuración del circuito.

    Retorna
    -------
    float
        Pérdida promedio basada únicamente en las clases positivas del lote ``Y``.

    Nota
    ----
    Esta función persiste por compatibilidad, pero la implementación real ocurre en
    :class:`QCNNVariationalModel` empleando la variante ``"positive"``.
    """
    model = QCNNVariationalModel(
        prob_clases=prob_clases,
        rot_angle=rotAngle,
        embedding_type=embedding_type,
        n_qubit=n_qubit,
        layers=layers,
        tip_rot=tipRot,
        qubits=qubits,
    )
    return model.loss(X, Y, params, total_params, loss_variant="positive")

def cost3(
    X,
    Y,
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
    Evalúa una pérdida que primero normaliza las salidas mediante una función sigmoide.

    Parámetros
    ----------
    X, Y, params, total_params, qubits, tipRot, layers, prob_clases, rotAngle,
    embedding_type, n_qubit:
        Misma semántica que en :func:`cost`.

    Retorna
    -------
    float
        Valor medio de la pérdida que aplica normalización sigmoide antes de la entropía.

    Nota
    ----
    Para asegurar consistencia, la lógica se delega a :class:`QCNNVariationalModel` con la
    variante ``"sigmoid"``.
    """
    model = QCNNVariationalModel(
        prob_clases=prob_clases,
        rot_angle=rotAngle,
        embedding_type=embedding_type,
        n_qubit=n_qubit,
        layers=layers,
        tip_rot=tipRot,
        qubits=qubits,
    )
    return model.loss(X, Y, params, total_params, loss_variant="sigmoid")

def guardar(
    path_guardar,
    predictions,
    test,
    optimizers,
    encoding,
    learning_rate,
    nombre,
    param,
    modelo,
    error,
    superposicion,
):
    """
    Persiste las predicciones y registra métricas de evaluación en disco.

    Parámetros
    ----------
    path_guardar:
        Directorio base para almacenar resultados y artefactos.
    predictions:
        Probabilidades estimadas por el modelo en el conjunto de prueba.
    test:
        Etiquetas reales corresponentes al conjunto de evaluación.
    optimizers, encoding, learning_rate, nombre, param, modelo, error, superposicion:
        Metadatos asociados al experimento; se conservan por compatibilidad.

    Retorna
    -------
    tuple[numpy.ndarray, float]
        Exactitud por clase y exactitud promedio global.
    """
    accuracy, acc_total = calculate_multilabel_accuracy(predictions, test)
    print("Accuracy por clase:", accuracy)
    print("Accuracy Total:", acc_total)
    fila = [
        str(optimizers),
        encoding,
        str(learning_rate),
        nombre,
        str(param),
        str(modelo),
        str(acc_total),
        str(accuracy),
    ]
    guardar_objetos(f"predictions_test_{nombre}", path_guardar, predictions)
    guardar_predicciones(path_guardar, "multi_qcnn", "resultados_pruebas.csv", fila)
    return accuracy, acc_total


def cargar_parametros_config(path_parametros: str | Path):
    """
    Envuelve :func:`~quantum_information.utils.utils.load_parametric_config` para mantener compatibilidad.

    Parámetros
    ----------
    path_parametros:
        Ruta del archivo de texto con las configuraciones paramétricas.

    Retorna
    -------
    list[list[Any]]
        Configuraciones paramétricas procesadas.
    """
    return load_parametric_config(path_parametros)


def cargarParametros(path_parametros):  # pragma: no cover - compatibilidad
    """
    Mantiene compatibilidad con ``cargar_parametros_config``.
    """
    return cargar_parametros_config(path_parametros)
