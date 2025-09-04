import numpy as np
import pytest
from quantum_metrics import trace_distance, fidelity

def test_trace_distance_identical_states():
    rho = np.array([[1, 0], [0, 0]], dtype=complex)  # |0>
    sigma = np.array([[1, 0], [0, 0]], dtype=complex)  # igual
    assert trace_distance(rho, sigma) == pytest.approx(0.0)

def test_trace_distance_orthogonal_states():
    rho = np.array([[1, 0], [0, 0]], dtype=complex)  # |0>
    sigma = np.array([[0, 0], [0, 1]], dtype=complex)  # |1>
    # Distancia máxima entre estados ortogonales = 1
    assert trace_distance(rho, sigma) == pytest.approx(1.0)

def test_fidelity_identical_states():
    rho = np.array([[1, 0], [0, 0]], dtype=complex)
    sigma = np.array([[1, 0], [0, 0]], dtype=complex)
    assert fidelity(rho, sigma) == pytest.approx(1.0)

def test_fidelity_orthogonal_states():
    rho = np.array([[1, 0], [0, 0]], dtype=complex)  # |0>
    sigma = np.array([[0, 0], [0, 1]], dtype=complex)  # |1>
    # Fidelidad mínima entre estados ortogonales = 0
    assert fidelity(rho, sigma) == pytest.approx(0.0)
