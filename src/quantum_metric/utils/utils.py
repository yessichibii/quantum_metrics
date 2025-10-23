# =============================================================
# ------------------------ UTILIDADES --------------------------
# =============================================================

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