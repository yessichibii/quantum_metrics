from __future__ import annotations

import csv
import os
import pickle
from pathlib import Path
from typing import Any, Iterable, Sequence

import autograd.numpy as anp
from pennylane import numpy as pnp


# =============================================================
# ------------------------ UTILIDADES --------------------------
# =============================================================


def ensure_directory(path: str | os.PathLike[str]) -> Path:
    """
    Crea el directorio indicado si no existe y devuelve la ruta como ``Path``.

    Parámetros
    ----------
    path:
        Ruta del directorio que debe existir.

    Retorna
    -------
    Path
        Ruta normalizada del directorio creado o existente.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_pickle(
    obj: Any, file_path: str | os.PathLike[str], create_parents: bool = True
) -> None:
    """
    Serializa un objeto en formato pickle dentro de la ruta especificada.

    Parámetros
    ----------
    obj:
        Objeto serializable que se desea guardar.
    file_path:
        Ruta absoluta o relativa del archivo destino.
    create_parents:
        Si es ``True`` se crean los directorios padre necesarios.
    """
    destination = Path(file_path)
    if create_parents:
        ensure_directory(destination.parent)
    with destination.open("wb") as handler:
        pickle.dump(obj, handler)


def load_pickle(
    file_path: str | os.PathLike[str],
    default: Any | None = None,
) -> Any | None:
    """
    Carga un archivo pickle y devuelve el objeto almacenado.

    Parámetros
    ----------
    file_path:
        Ruta absoluta o relativa del archivo a leer.
    default:
        Valor a devolver si el archivo no existe. Si es ``None`` y tampoco
        existe el archivo, se retorna ``None``.

    Retorna
    -------
    Any | None
        Objeto deserializado o ``default`` si el archivo no está disponible.
    """
    source = Path(file_path)
    if not source.exists():
        return default
    with source.open("rb") as handler:
        return pickle.load(handler)


def append_csv_row(
    directory: str | os.PathLike[str],
    filename: str,
    row: Sequence[Any],
    mode: str = "a+",
    delimiter: str = ",",
) -> None:
    """
    Agrega una fila a un archivo CSV, creando los directorios necesarios.

    Parámetros
    ----------
    directory:
        Directorio donde se almacenará el archivo CSV.
    filename:
        Nombre del archivo CSV.
    row:
        Iterable con los datos a registrar en la fila.
    mode:
        Modo de apertura del archivo (por defecto ``"a+"``).
    delimiter:
        Delimitador empleado para separar las columnas.
    """
    ensure_directory(directory)
    csv_path = Path(directory) / filename
    with csv_path.open(mode, newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter)
        writer.writerow(list(row))


def guardar_parametros(
    archivo: str, directorio: str | os.PathLike[str], parametros: Any
) -> Path:
    """
    Almacena un conjunto arbitrario de parámetros en disco usando formato pickle.

    Parámetros
    ----------
    archivo:
        Nombre del archivo destino (sin la ruta).
    directorio:
        Directorio donde se guardará el archivo. Se crea automáticamente si no
        existe.
    parametros:
        Objeto serializable en pickle, por ejemplo un vector de parámetros
        entrenables.

    Retorna
    -------
    Path
        Ruta completa del archivo generado.
    """
    destino = Path(directorio) / archivo
    save_pickle(parametros, destino, create_parents=True)
    return destino


def cargar_o_inicializar_parametros(
    archivo: str,
    directorio: str | os.PathLike[str],
    total_parametros: Sequence[int],
) -> Any:
    """
    Carga un arreglo de parámetros desde disco o lo inicializa aleatoriamente.

    Parámetros
    ----------
    archivo:
        Nombre del archivo pickle que contiene los parámetros.
    directorio:
        Directorio base donde se espera encontrar el archivo.
    total_parametros:
        Secuencia que indica el número de parámetros de cada bloque; se utiliza
        para calcular el tamaño total a inicializar en caso de no existir el
        archivo.

    Retorna
    -------
    Any
        Objeto con los parámetros listos para ser utilizados en la
        optimización variacional.
    """
    # Importación diferida para evitar dependencias circulares con módulos que
    # también dependen de utilidades genéricas.
    from pennylane import numpy as np  # type: ignore import-not-found

    ensure_directory(directorio)
    ruta = Path(directorio) / archivo
    parametros = load_pickle(ruta)
    if parametros is None:
        print("Creando nuevos parámetros")
        parametros = np.random.randn(sum(total_parametros), requires_grad=True)
    else:
        print("Cargando parámetros desde:", ruta)
    return parametros


def cargar_metricas(
    archivo: str, directorio: str | os.PathLike[str]
) -> Any | None:
    """
    Recupera métricas serializadas previamente en formato pickle.

    Parámetros
    ----------
    archivo:
        Nombre del archivo pickle.
    directorio:
        Directorio donde se localiza el archivo.

    Retorna
    -------
    Any | None
        Objeto deserializado o ``None`` si el archivo no está disponible.
    """
    ensure_directory(directorio)
    ruta = Path(directorio) / archivo
    metricas = load_pickle(ruta)
    if metricas is not None:
        print("Cargando métricas desde:", ruta)
    return metricas


def guardar_objetos(
    archivo: str, directorio: str | os.PathLike[str], objeto: Any
) -> Path:
    """
    Guarda de manera genérica un objeto serializable en un archivo pickle.

    Parámetros
    ----------
    archivo:
        Nombre del archivo destino.
    directorio:
        Directorio donde se escribirá el archivo.
    objeto:
        Objeto serializable.

    Retorna
    -------
    Path
        Ruta completa del archivo guardado.
    """
    return guardar_parametros(archivo, directorio, objeto)


def guardar_predicciones(
    directorio_base: str | os.PathLike[str],
    nombre_algoritmo: str,
    archivo: str,
    datos: Sequence[Any],
    modo: str = "a+",
) -> Path:
    """
    Agrega una fila de resultados a un CSV asociado a un algoritmo determinado.

    Parámetros
    ----------
    directorio_base:
        Ruta raíz donde se almacenarán los resultados.
    nombre_algoritmo:
        Subdirectorio específico que agrupa los experimentos del algoritmo.
    archivo:
        Nombre del archivo CSV.
    datos:
        Secuencia con los valores que se desean registrar como nueva fila.
    modo:
        Modo de apertura del archivo (por defecto ``\"a+\"``).

    Retorna
    -------
    Path
        Ruta completa del archivo actualizado.
    """
    base = ensure_directory(directorio_base)
    subdirectorio = ensure_directory(base / nombre_algoritmo)
    append_csv_row(subdirectorio, archivo, datos, mode=modo)
    return subdirectorio / archivo


def calculate_multilabel_accuracy(
    predictions: Sequence[Sequence[float]],
    labels: Sequence[Sequence[int]],
) -> tuple[pnp.ndarray, float]:
    """
    Calcula la exactitud por clase y global para un problema de clasificación multietiqueta.

    La función identifica el número real de etiquetas activas por muestra y, para cada
    vector de probabilidades, selecciona los índices de mayor probabilidad como predicción.
    Posteriormente compara etiqueta por etiqueta para obtener estadísticas por clase y
    exactitud global (micro-promedio).

    Parámetros
    ----------
    predictions:
        Colección de vectores de probabilidad generados por el modelo. Acepta también
        listas anidadas que contengan un único lote.
    labels:
        Colección de etiquetas binarizadas (one-hot) correspondientes a cada muestra.

    Retorna
    -------
    tuple[numpy.ndarray, float]
        Arreglo con la exactitud por clase y valor escalar con la exactitud global.
    """
    if not labels:
        raise ValueError("Se requiere al menos una etiqueta para evaluar la exactitud.")

    # Normaliza el lote de predicciones por compatibilidad con diferentes flujos de datos.
    if isinstance(predictions, (list, tuple)) and len(predictions) == 1:
        probabilities: Iterable[Sequence[float]] = predictions[0]
    else:
        probabilities = predictions

    num_classes = len(labels[0])
    correct_predictions = pnp.zeros(num_classes, dtype=float)
    acc_total = 0.0

    for etiqueta, prob in zip(labels, probabilities):
        etiqueta_array = pnp.asarray(etiqueta)
        prob_array = pnp.asarray(prob)
        num_activos = int(pnp.sum(etiqueta_array))
        if num_activos == 0:
            raise ValueError(
                "Las etiquetas deben contener al menos una clase positiva por muestra."
            )
        indices_mayores = pnp.argsort(prob_array)[-num_activos:]
        predicha = pnp.zeros(num_classes, dtype=int)
        predicha[indices_mayores] = 1
        correct_predictions += (predicha == etiqueta_array).astype(float)
        if pnp.array_equal(predicha, etiqueta_array):
            acc_total += 1.0

    acc_total /= len(labels)
    accuracy = correct_predictions / len(labels)
    return accuracy, float(acc_total)


def binary_cross_entropy_multilabel(
    labels: Sequence[Sequence[float]], predictions: Sequence[Sequence[float]], epsilon: float = 1e-15
) -> float:
    """
    Calcula la pérdida de entropía cruzada binaria considerando clases positivas y negativas.

    Parámetros
    ----------
    labels:
        Lote de etiquetas en codificación binaria (one-hot).
    predictions:
        Lote de probabilidades estimadas por el modelo.
    epsilon:
        Valor mínimo empleado para evitar evaluaciones de ``log(0)``.

    Retorna
    -------
    float
        Valor medio de la pérdida sobre el lote proporcionado.
    """
    if not labels:
        raise ValueError("Se requiere al menos una etiqueta para calcular la pérdida.")

    num_samples = len(labels)
    loss = 0.0
    for etiqueta, prob in zip(labels, predictions):
        etiqueta_array = anp.asarray(etiqueta)
        prob_array = anp.asarray(prob)
        clipped = anp.clip(prob_array, epsilon, 1 - epsilon)
        loss -= anp.sum(
            etiqueta_array * anp.log(clipped) + (1 - etiqueta_array) * anp.log(1 - clipped)
        )
    return float(loss / num_samples)


def binary_cross_entropy_positive(
    labels: Sequence[Sequence[float]], predictions: Sequence[Sequence[float]], epsilon: float = 1e-15
) -> float:
    """
    Calcula una variante de entropía cruzada que considera únicamente las clases positivas.

    Parámetros
    ----------
    labels:
        Lote de etiquetas binarizadas.
    predictions:
        Lote de probabilidades estimadas por el modelo.
    epsilon:
        Tolerancia mínima para evitar problemas numéricos en los logaritmos.

    Retorna
    -------
    float
        Promedio de la pérdida sobre el lote proporcionado.
    """
    if not labels:
        raise ValueError("Se requiere al menos una etiqueta para calcular la pérdida.")

    num_samples = len(labels)
    loss = 0.0
    for etiqueta, prob in zip(labels, predictions):
        etiqueta_array = anp.asarray(etiqueta)
        prob_array = anp.asarray(prob)
        clipped = anp.clip(prob_array, epsilon, 1 - epsilon)
        loss -= anp.sum(etiqueta_array * anp.log(clipped))
    return float(loss / num_samples)


def binary_cross_entropy_with_sigmoid(
    labels: Sequence[Sequence[float]], predictions: Sequence[Sequence[float]], epsilon: float = 1e-15
) -> float:
    """
    Calcula la entropía cruzada binaria tras normalizar las probabilidades con una sigmoide.

    Parámetros
    ----------
    labels:
        Lote de etiquetas verdaderas en formato binario.
    predictions:
        Lote de puntuaciones estimadas por el modelo (no necesariamente normalizadas).
    epsilon:
        Tolerancia para prevenir ``log(0)`` y ``log(1)``.

    Retorna
    -------
    float
        Promedio de la pérdida sobre el lote.
    """
    if not labels:
        raise ValueError("Se requiere al menos una etiqueta para calcular la pérdida.")

    num_samples = len(labels)
    loss = 0.0
    for etiqueta, prob in zip(labels, predictions):
        etiqueta_array = anp.asarray(etiqueta)
        prob_array = anp.asarray(prob)
        max_val = float(anp.max(prob_array))
        if max_val == 0.0:
            raise ValueError(
                "Las probabilidades no pueden ser todas cero al normalizar con sigmoide."
            )
        unnormalized = prob_array / max_val
        prob_sigmoid = 1 / (1 + anp.exp(-unnormalized))
        clipped = anp.clip(prob_sigmoid, epsilon, 1 - epsilon)
        loss -= anp.sum(
            etiqueta_array * anp.log(clipped) + (1 - etiqueta_array) * anp.log(1 - clipped)
        )
    return float(loss / num_samples)


def load_parametric_config(path_parametros: str | os.PathLike[str]) -> list[list[Any]]:
    """
    Carga y estructura configuraciones paramétricas almacenadas en un archivo de texto.

    El archivo debe contener bloques delimitados por líneas que finalicen con ``]``. Cada
    bloque representa una configuración independiente que se agrega a la lista retornada.

    Parámetros
    ----------
    path_parametros:
        Ruta al archivo de texto que contiene las configuraciones.

    Retorna
    -------
    list[list[Any]]
        Colección de configuraciones evaluadas a partir del contenido textual.
    """
    datos: list[Any] = []
    parametros: list[list[Any]] = []
    with Path(path_parametros).open("r", encoding="utf-8") as archivo:
        for linea in archivo:
            linea = linea.strip()
            if not linea:
                continue
            if linea == "]":
                if datos:
                    parametros.append(datos)
                datos = []
                continue
            if "=" not in linea:
                datos.append(eval(linea[:-1]))
    if datos:
        parametros.append(datos)
    return parametros

def gray_code_inverso(n):
    """
    Genera el código Gray inverso de n bits (bits invertidos).

    Esta función construye primero el código Gray estándar y luego invierte cada bit (0→1, 1→0).

    Parámetros
    ----------
    n : int
        Número de bits para generar el código Gray inverso.

    Retorna
    -------
    list[str]
        Lista de cadenas binarias representando el código Gray con bits invertidos.

    Ejemplo
    --------
    >>> gray_code_inverso(3)
    ['111', '110', '100', '101', '001', '000', '010', '011']
    """
    gray = gray_code(n)
    gray_invertido = ["".join("1" if b == "0" else "0" for b in code) for code in gray]
    return gray_invertido

def gray_code(n):
    """
    Genera el código Gray de n bits.

    El código Gray es una secuencia binaria en la cual dos números consecutivos difieren en un solo bit.

    Parámetros
    ----------
    n : int
        Número de bits para generar el código Gray.

    Retorna
    -------
    list[str]
        Lista de cadenas binarias que representan el código Gray de n bits.

    Ejemplo
    --------
    >>> gray_code(3)
    ['000', '001', '011', '010', '110', '111', '101', '100']
    """
    if n == 0:
        return ["0"]
    if n == 1:
        return ["0", "1"]
    prev = gray_code(n-1)
    return ["0" + x for x in prev] + ["1" + x for x in reversed(prev)]

def binario_code(n):
    """
    Genera todas las combinaciones binarias posibles de n bits.

    Parámetros
    ----------
    n : int
        Número de bits.

    Retorna
    -------
    list[str]
        Lista de cadenas binarias de longitud n.

    Ejemplo
    --------
    >>> generar_binario(3)
    ['000', '001', '010', '011', '100', '101', '110', '111']
    """
    return [format(i, f"0{n}b") for i in range(2 ** n)]

def binario_code_inverso(n):
    """
    Genera todas las combinaciones binarias posibles de n bits con los bits invertidos. Devuelve su complemento binario, es decir, cada bit se invierte:
    - 0 → 1  
    - 1 → 0  

    Parámetros
    ----------
    n : int
        Número de bits.

    Retorna
    -------
    list[str]
        Lista de cadenas binarias de longitud n con bits invertidos.

    Ejemplo
    --------
    >>> binario_code_inverso(3)
    ['111', '110', '101', '100', '011', '010', '001', '000']
    """

    # Genera las combinaciones binarias normales
    binario = binario_code(n)

    # Invierte cada bit (0→1, 1→0) para obtener el complemento binario
    binario_invertido = [
        "".join("1" if bit == "0" else "0" for bit in code)
        for code in binario
    ]

    return binario_invertido