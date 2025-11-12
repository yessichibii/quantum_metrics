"""
Pipelines y utilidades de preprocesamiento clásico previos a la codificación cuántica.
"""

from .data import (
    binarizar,
    cargar_archivo,
    cargar_objetos,
    describir_categoricos,
    escalar,
    guardar_objetos,
    guardar_prediciones,
    imputar_por_clase,
    normalizar,
    normalizar_escalar,
    to_one_hot,
)

__all__ = [
    "binarizar",
    "cargar_archivo",
    "cargar_objetos",
    "describir_categoricos",
    "escalar",
    "guardar_objetos",
    "guardar_prediciones",
    "imputar_por_clase",
    "normalizar",
    "normalizar_escalar",
    "to_one_hot",
]

