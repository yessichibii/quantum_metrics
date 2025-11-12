import numpy as np
import pytest

from quantum_information.metrics.distances import state_mba_distance


def test_state_mba_distance_type():
    # Dataset simple: 2 muestras, 1 característica
    dataset = np.array([[0.0], [1.0]])
    test = [0.5]

    state = state_mba_distance(dataset, test, result = "state")

    # El resultado debe ser un array de numpy de tipo complejo
    assert isinstance(state, np.ndarray)
    assert np.iscomplexobj(state)

def test_state_mba_distance_normalization():
    dataset = np.array([[0.0], [1.0]])
    test = [0.5]

    state = state_mba_distance(dataset, test, result = "state")

    # El estado cuántico debe estar normalizado: ||state||^2 = 1
    norm = np.sum(np.abs(state) ** 2)
    assert norm == pytest.approx(1.0)

def test_state_mba_distance_distance_consistency():
    dataset = np.array([[0.0], [1.0]])
    test1 = [0.0]
    test2 = [1.0]

    state1 = state_mba_distance(dataset, test1, result = "state")
    state2 = state_mba_distance(dataset, test2, result = "state")

    # Los estados deberían ser distintos para test diferentes
    assert not np.allclose(state1, state2)