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


# Probamos

# Ejemplo: dataset con 4 patrones de 2 características cada uno
dataset = np.array([
    [1, 0],
    [0, 1],
    [1, 1],
    [0.5, 0.5]
], requires_grad=False)
test = [0.5,0.5]


state = penny_amplitude_distance(dataset,test)
print("Estado QRAM:\n", state)
