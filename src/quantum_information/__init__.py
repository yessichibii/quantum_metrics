"""
Paquete principal de la librería quantum_information.

Expone las API reorganizadas en submódulos temáticos: encoding, circuits,
metrics, preprocessing y utils.
"""

from . import circuits, encoding, metrics, preprocessing, utils

__all__ = [
    "circuits",
    "encoding",
    "metrics",
    "preprocessing",
    "utils",
]

