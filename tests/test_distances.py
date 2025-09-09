import numpy as np
import pytest
from qdistancia import trace_distance, fidelity, h_amplitude_distance

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

def test_h_amplitude_distance_output_type():
    # Dataset simple: 2 muestras, 1 característica
    dataset = np.array([[0.0], [1.0]])
    test = [0.5]

    state = h_amplitude_distance(dataset, test)

    # El resultado debe ser un array de numpy de tipo complejo
    assert isinstance(state, np.ndarray)
    assert np.iscomplexobj(state)

def test_h_amplitude_distance_normalization():
    dataset = np.array([[0.0], [1.0]])
    test = [0.5]

    state = h_amplitude_distance(dataset, test)

    # El estado cuántico debe estar normalizado: ||state||^2 = 1
    norm = np.sum(np.abs(state) ** 2)
    assert norm == pytest.approx(1.0)

def test_h_amplitude_distance_consistency():
    dataset = np.array([[0.0], [1.0]])
    test1 = [0.0]
    test2 = [1.0]

    state1 = h_amplitude_distance(dataset, test1)
    state2 = h_amplitude_distance(dataset, test2)

    # Los estados deberían ser distintos para test diferentes
    assert not np.allclose(state1, state2)