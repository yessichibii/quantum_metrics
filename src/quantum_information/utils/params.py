from pennylane import numpy as pnp
from typing import Optional, Union
import warnings

def init_params(n: int, init: Optional[Union[pnp.ndarray, list]] = None, rng: float = 0.5) -> pnp.ndarray:
    """
    Inicializa parámetros para circuitos cuánticos.
    
    Parámetros
    ----------
    n : int
        Número de parámetros a inicializar.
    init : array-like, opcional
        Pesos específicos para el circuito cuántico. Si es None o vacío, se inicializan aleatoriamente.
    rng : float, default=0.5
        Rango para inicialización aleatoria (-rng a +rng).
        
    Retorna
    -------
    anp.ndarray
        Array de parámetros inicializados.
        
    Raises
    ------
    ValueError
        Si los pesos no tienen la longitud correcta.
    """
    if init is None or (isinstance(init, (list, pnp.ndarray)) and len(init) == 0):
        return pnp.array(pnp.random.uniform(-rng, rng, n), requires_grad=True)
    
    # Convertir a array de numpy/autograd
    init_array = pnp.array(init)
    
    # Validar longitud
    if len(init_array) != n:
        raise ValueError(
            f"Los pesos deben tener longitud {n}, "
            f"pero se proporcionaron {len(init_array)} pesos"
        )
    
    # Validar que sean números finitos
    if not pnp.all(pnp.isfinite(init_array)):
        raise ValueError("Los pesos deben ser números finitos")
    
    if not pnp.all((init_array >= -1) & (init_array <= 1)):
        raise ValueError("Los pesos deben estar en el rango [-1, 1]")

    return pnp.array(init_array, requires_grad=True)

def cross_entropy(labels, predictions):
    epsilon = 1e-15
    predictions = pnp.clip(predictions, epsilon, 1 - epsilon)
    loss = -pnp.sum(labels * pnp.log(predictions)) / len(labels)
    return loss