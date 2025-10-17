import pandas as pd
import csv,os
import pickle
import math

# =============================================================
# --------------- PREPROCESAMIENTO DE DATOS -------------------
# =============================================================


def normalizar(numericos,decimales):
    numericos = (numericos - numericos.min()) / ( numericos.max() - numericos.min())
    numericos.fillna(0, inplace=True)
    numericos = numericos.round(decimales)
    return numericos

def normalizarEscalar(numericos,decimales):
    numericos = (numericos - numericos.min()) / ( numericos.max() - numericos.min())
    numericos.fillna(0, inplace=True)
    numericos = numericos.round(decimales)
    numericos = numericos*(10**decimales)
    return numericos.astype(int)

def escalar(numericos,decimales):
    numericos = numericos*(10**decimales)
    return numericos.astype(int)

def binarizar(datos):
    numMax = datos.max()
    columnas = list()
    for columna in datos:
        col = 1 if math.ceil(math.log(numMax[columna]+1,2)) == 0 else math.ceil(math.log(numMax[columna]+1,2))
        columnas.append(col)
    dfBinario = pd.DataFrame(columns=range(sum(columnas)))
    dfBinarioGray = pd.DataFrame(columns=range(sum(columnas)))
    for i in range(datos.shape[0]):
        cont = 0
        datosBinario = list()
        datosBinarioGray = list()
        for columna in datos:
            binario = str(format(datos.loc[i,columna],"b")).zfill(columnas[cont])
            datosBinario += list(binario)
            binGray = binarioGray(binario)
            datosBinarioGray += list(binGray)
            cont += 1
        dfBinario.loc[i,:] = datosBinario
        dfBinarioGray.loc[i,:] = datosBinarioGray
    return dfBinario,dfBinarioGray

def describirCategoricos(data, categoricos):
    df = data.copy()
    elementos = []
    for i in categoricos:
        elementos = np.concatenate((elementos, df[i].unique()), axis=None)
    elementos = np.unique(elementos)
    for i in categoricos:
        for j in range(len(elementos)):           
            df.loc[df[i] == elementos[j],i] = j
    return df

def imputarNumericosPorClase(data, numericos, idClase):
    df = data.copy()
    for col in numericos:
        for clase in df[idClase].unique():
            # Filtrar por clase
            subset = df[df[idClase] == clase][col]
            # Convertir a numérico (en caso de que haya strings como "?")
            subset = pd.to_numeric(subset, errors='coerce')
            # Calcular la media sin NaN
            media = subset.mean()
            # Sustituir los "?" por la media de la clase
            df.loc[(df[idClase] == clase) & (df[col] == "?"), col] = media
            # También sustituir NaN si se generaron con errors='coerce'
            df.loc[(df[idClase] == clase) & (df[col].isna()), col] = media
        # Asegurar que la columna queda numérica
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def imputarCategoricosPorClase(data, categoricos, idClase):
    df = data.copy()
    for col in categoricos:
        for clase in df[idClase].unique():
            subset = df[df[idClase] == clase][col]
            # Filtrar los valores válidos (sin "?")
            valores_validos = subset[subset != "?"]
            if len(valores_validos) > 0:
                moda = valores_validos.mode()[0]
                # Sustituir "?" por la moda de esa clase
                df.loc[(df[idClase] == clase) & (df[col] == "?"), col] = moda
    return df


# =============================================================
# --------------- CARGAR Y GUARDAR ARCHIVOS -------------------
# =============================================================

def cargarArchivo(archivo,datosNumericos,datosCategoricos):
    data = pd.read_csv(archivo, header=None)
    idClase = data.shape[1]-1
    clases = data[idClase].unique()
    for j in range(len(clases)):   
        data.loc[data[idClase] == clases[j], idClase] = j
    data = imputarNumericosPorClase(data, datosNumericos, idClase)
    data = imputarCategoricosPorClase(data, datosCategoricos, idClase)
    data = describirCategoricos(data, datosCategoricos)
    return data, clases

def guardarPrediciones(algoritmo,archivo,datos,tipo):    
    os.makedirs('resultados', exist_ok=True)
    os.makedirs('resultados/'+algoritmo, exist_ok=True)
    os.makedirs('resultados/'+algoritmo+"/predicciones", exist_ok=True)
    with open('resultados/'+algoritmo+"/"+archivo, tipo, newline ='') as csvfile:
        wr = csv.writer(csvfile, delimiter=',')
        wr.writerow(datos)
    csvfile.close()

def guardarObjetos(algoritmo,archivo,datos):
    os.makedirs('resultados', exist_ok=True)
    alg = ''
    for j in algoritmo:
        alg = alg+'/'+j
        os.makedirs('resultados'+alg, exist_ok=True)
    with open('resultados'+alg+"/"+archivo, "wb") as f:
        pickle.dump(datos, f)

def cargarObjetos(algoritmo,archivo):
    with open('resultados/'+algoritmo+"/"+archivo, "rb") as f:
        obj = pickle.load(f)
    return obj