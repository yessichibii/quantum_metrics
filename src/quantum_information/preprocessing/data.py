import pandas as pd
import numpy as np
import csv,os
import pickle
import math

# =============================================================
# --------------- PREPROCESAMIENTO DE DATOS -------------------
# =============================================================


def normalizar(numericos,decimales):
    """
    Normaliza los datos numéricos al rango [0, 1].

    Parámetros
    ----------
    numericos : pandas.DataFrame o pandas.Series
        Conjunto de columnas numéricas a normalizar.
    decimales : int
        Número de decimales al que se redondearán los valores normalizados.

    Retorna
    -------
    pandas.DataFrame o pandas.Series
        Datos normalizados en el rango [0, 1], redondeados al número de decimales especificado.
    """
    if not isinstance(numericos, pd.DataFrame):
        raise TypeError("El parámetro 'numericos' debe ser un DataFrame.")
    numericos = (numericos - numericos.min()) / ( numericos.max() - numericos.min())
    numericos.fillna(0, inplace=True)
    numericos = numericos.round(decimales)
    return numericos

def normalizar_escalar(numericos,decimales):
    """
    Normaliza los datos numéricos al rango [0, 1] y los escala a enteros.

    Parámetros
    ----------
    numericos : pandas.DataFrame o pandas.Series
        Conjunto de columnas numéricas a procesar.
    decimales : int
        Número de decimales que define la escala (por ejemplo, 2 -> multiplica por 100).

    Retorna
    -------
    pandas.DataFrame o pandas.Series
        Datos normalizados y escalados como enteros.
    """
    numericos = normalizar(numericos,decimales)
    numericos = escalar(numericos,decimales)
    return numericos.astype(int)

def escalar(numericos,decimales):
    """
    Escala los valores numéricos sin normalización.

    Parámetros
    ----------
    numericos : pandas.DataFrame o pandas.Series
        Conjunto de columnas numéricas.
    decimales : int
        Factor de escala (por ejemplo, 2 -> multiplica por 100).

    Retorna
    -------
    pandas.DataFrame o pandas.Series
        Datos escalados a enteros.
    """
    if not isinstance(numericos, pd.DataFrame):
        raise TypeError("El parámetro 'numericos' debe ser un DataFrame.")
    numericos = numericos*(10**decimales)
    return numericos.astype(int)

def binario_a_gray(binario: str) -> str:
    gray = binario[0]
    for i in range(1, len(binario)):
        gray_bit = str(int(binario[i-1]) ^ int(binario[i]))
        gray += gray_bit
    return gray

def binarizar(datos, tipo="ambos"):
    """
    Convierte los valores numéricos en sus representaciones binarias y Gray,
    aplicando la codificación por rasgo (columna).

    Parámetros
    ----------
    datos : pandas.DataFrame
        Datos numéricos a binarizar.
    tipo : str, opcional
        'binario', 'gray' o 'ambos' (por defecto 'ambos').

    Retorna
    -------
    dict
        Diccionario con las claves:
        - 'binario': DataFrame de bits binarios.
        - 'gray': DataFrame de bits Gray (si aplica).
        - 'bits_por_columna': Series con la cantidad de bits usada por rasgo.
    """
    
    if not isinstance(datos, pd.DataFrame):
        raise TypeError("El parámetro 'df' debe ser un DataFrame.")
    
    max_values = pd.Series({col: datos[col].max() for col in datos})
    bits_por_columna = (max_values+1).apply(lambda x: max(1, (x-1).bit_length()))
    resultado = {"bits_por_columna": bits_por_columna}

    datos_int = datos.astype(int)
    binarios = {
        col: datos_int[col].apply(lambda x: format(x, f'0{bits_por_columna[col]}b'))
        for col in datos.columns
    }
    if tipo in ("binario", "ambos"):
        matriz_binaria = np.array(
            [''.join(fila) for fila in zip(*binarios.values())]
        )

        # Expandir a columnas binarias separadas
        df_binario = pd.DataFrame(
            [list(fila) for fila in matriz_binaria],
            columns=range(sum(bits_por_columna))
        ).astype(int)
        resultado['binario'] = df_binario

    if tipo in ("gray", "ambos"):
        datos_gray = {col: binarios[col].apply(binario_a_gray) for col in binarios}
        matriz_gray = np.array(
            [''.join(fila) for fila in zip(*datos_gray.values())]
        )

        # Expandir a columnas gray separadas
        df_gray  = pd.DataFrame(
            [list(fila) for fila in matriz_gray],
            columns=range(sum(bits_por_columna))
        ).astype(int)

        resultado['gray'] = df_gray


    return resultado

def to_one_hot(y, n_classes = None):
    y = np.asarray(y, dtype=int)
    if y.ndim == 2:
        return np.array(y)
    if n_classes is None:
        n_classes = int(y.max()) + 1
    oh = np.zeros((len(y), n_classes))
    for i, val in enumerate(y):
        oh[i, int(val)] = 1.0
    return oh

def describir_categoricos(data, categoricos):
    """
    Convierte columnas categóricas a números (Label Encoding) de forma independiente por columna.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene las columnas categóricas.
    categoricos : list
        Lista de nombres de columnas categóricas a codificar.

    Retorna
    -------
    pandas.DataFrame
        DataFrame con las columnas categóricas codificadas como enteros.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("El parámetro 'data' debe ser un DataFrame.")
    if not isinstance(categoricos, list):
        raise TypeError("El parámetro 'categoricos' debe ser una lista.")
    
    df = data.copy()
    for col in categoricos:
        if col not in df.columns:
            raise ValueError(f"La columna '{col}' no existe en el DataFrame.")
        valores_unicos = df[col].unique()
        # Crear diccionario de mapeo
        mapping = {valor: idx for idx, valor in enumerate(valores_unicos)}
        # Aplicar mapeo a la columna
        df[col] = df[col].map(mapping)
    return df

def imputar_por_clase(data, columnas, id_clase, tipo="numerico", modelo="media"):
    """
    Imputa valores faltantes ("?" o NaN) en las columnas especificadas por clase,
    usando media, mediana o moda según el tipo de columna y el modelo elegido.

    Parámetros
    ----------
    data : pandas.DataFrame
        Conjunto de datos.
    columnas : list
        Nombres o índices de columnas a imputar.
    idClase : str o int
        Nombre o índice de la columna que identifica la clase.
    tipo : str, default "numerico"
        Tipo de columna: "numerico" o "categorico".
    modelo : str, default "media"
        Método de imputación:
        - "media", "mediana", "moda" para numericos
        - "moda" categoricos

    Retorna
    -------
    pandas.DataFrame
        DataFrame con los valores imputados por clase.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("El parámetro 'data' debe ser un DataFrame.")
    if not isinstance(columnas, list):
        raise TypeError("El parámetro 'columnas' debe ser una lista.")
    if id_clase not in data.columns:
        raise ValueError(f"La columna de clase '{id_clase}' no existe en el DataFrame.")

    df = data.copy()
    
    for col in columnas:
        if col not in df.columns:
            raise ValueError(f"La columna '{col}' no existe en el DataFrame.")
        for clase in df[id_clase].unique():
            # Filtrar por clase
            subset = df[df[id_clase] == clase][col]

            match tipo:
                case "numerico":
                    subset = pd.to_numeric(subset.replace("?", np.nan), errors='coerce')
                    match modelo:
                        case  "media":
                            valor = subset.mean()
                        case  "mediana":
                            valor = subset.median()
                        case  "moda":
                            subset_validos = subset.dropna()
                            if len(subset_validos) == 0:
                                continue
                            valor = subset_validos.mode()[0]
                        case _:
                            raise ValueError(f'Modelo "{modelo}" no válido para numérico')
                case "categorico":
                    subset_validos = subset[subset != "?"]
                    if len(subset_validos) == 0:
                        continue
                    valor = subset_validos.mode()[0]  # Moda
                case _:
                    raise ValueError(f'Tipo "{tipo}" no reconocido, debe ser "numerico" o "categorico"')

            # Reemplazar los valores faltantes
            df.loc[(df[id_clase] == clase) & ((df[col] == "?") | (df[col].isna())), col] = valor

        # Asegurar tipo numérico si corresponde
        if tipo == "numerico":
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


# =============================================================
# --------------- CARGAR Y GUARDAR ARCHIVOS -------------------
# =============================================================

def cargar_archivo(path,datosNumericos,datosCategoricos):
    """
    Carga un archivo CSV, convierte las clases a valores numéricos e imputa valores faltantes.

    Parámetros
    ----------
    path : str
        Ruta al archivo CSV.
    datosNumericos : list
        Índices o nombres de columnas numéricas.
    datosCategoricos : list
        Índices o nombres de columnas categóricas.

    Retorna
    -------
    tuple (pandas.DataFrame, numpy.ndarray)
        - DataFrame procesado (valores numéricos y categóricos imputados).
        - Lista de clases originales.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    data = pd.read_csv(path, header=None)
    idClase = data.shape[1]-1
    clases = data[idClase].unique()
    for j in range(len(clases)):   
        data.loc[data[idClase] == clases[j], idClase] = j
    data = imputar_por_clase(data, datosNumericos, idClase)
    data = imputar_por_clase(data, datosCategoricos, idClase,tipo="categorico")
    data = describir_categoricos(data, datosCategoricos)
    return data, clases

def guardar_prediciones(path,archivo,datos,tipo):
    """
    Guarda predicciones en un archivo CSV dentro de una estructura de carpetas organizada.

    Parámetros
    ----------
    path : str
        Nombre de la ruta para la carpeta (se usará como subcarpeta).
    archivo : str
        Nombre del archivo CSV a guardar.
    datos : list
        Lista de valores o filas a escribir en el archivo.
    tipo : str
        Modo de apertura del archivo ('w' para escribir, 'a' para agregar).
    """ 

    os.makedirs(path, exist_ok=True)
    os.makedirs(path+"/predicciones", exist_ok=True)
    with open(path+"/predicciones/"+archivo, tipo, newline ='') as csvfile:
        wr = csv.writer(csvfile, delimiter=',')
        wr.writerow(datos)
    csvfile.close()

def guardar_objetos(path,archivo,datos):
    """
    Guarda un objeto Python (modelo, resultados, etc.) en formato pickle.

    Parámetros
    ----------
    path : list
        Lista jerárquica de carpetas (por ejemplo ['modelo', 'version1']).
    archivo : str
        Nombre del archivo .pkl.
    datos : object
        Objeto a guardar.
    """

    alg = ''
    for j in path:
        alg = alg+'/'+j
        os.makedirs(alg, exist_ok=True)
    with open(alg+archivo, "wb") as f:
        pickle.dump(datos, f)

def cargar_objetos(path,archivo):
    """
    Carga un objeto previamente guardado con pickle.

    Parámetros
    ----------
    path : str
        Nombre de la ruta
    archivo : str
        Nombre del archivo a cargar.

    Retorna
    -------
    object
        Objeto cargado desde el archivo pickle.
    """
    if not os.path.exists(path+"/"+archivo):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    with open(path+"/"+archivo, "rb") as f:
        obj = pickle.load(f)
    return obj