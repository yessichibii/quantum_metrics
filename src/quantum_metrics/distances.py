import numpy as np

def trace_distance(rho, sigma):
    """Calcula la distancia traza entre dos matrices densidad."""
    diff = rho - sigma
    eigvals = np.linalg.eigvals(diff @ diff.conj().T)
    return 0.5 * np.sum(np.sqrt(np.abs(eigvals)))

def fidelity(rho, sigma):
    """Calcula la fidelidad cuántica entre dos matrices densidad."""
    sqrt_rho = scipy.linalg.sqrtm(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho
    return np.real(np.trace(scipy.linalg.sqrtm(inner)) ** 2)



def penny_amplitude_distance():


def qiskit_amplitude_distance():
"""
Dados dos 
"""
    
