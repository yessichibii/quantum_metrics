import numpy as np
from quantum_metrics import trace_distance, fidelity

# Definir dos matrices densidad (estados cuánticos)
rho = np.array([[1, 0], [0, 0]], dtype=complex)  # |0⟩
sigma = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)  # |+⟩

print("Matriz rho:")
print(rho)
print("Matriz sigma:")
print(sigma)

# Calcular métricas
print("\n--- Métricas Cuánticas ---")
print(f"Distancia traza: {trace_distance(rho, sigma):.4f}")
print(f"Fidelidad: {fidelity(rho, sigma):.4f}")
