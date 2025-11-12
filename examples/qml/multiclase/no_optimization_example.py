"""
Ejemplo específico del uso del clasificador multiclase sin optimización de pesos.

Este ejemplo demuestra cómo usar el clasificador cuántico multiclase sin optimización
de pesos, útil para evaluar el rendimiento base del circuito cuántico.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from quantum_metric.models import QMLMultiClase


def main():
    """Función principal que demuestra el uso sin optimización."""
    
    print("🎯 Ejemplo de QMLMultiClase Sin Optimización de Pesos")
    print("=" * 70)
    
    # Generar datos sintéticos
    print("\n📊 Generando dataset sintético multiclase...")
    X, y = make_classification(
        n_samples=120,
        n_features=3,
        n_informative=2,
        n_redundant=1,
        n_classes=4,  # 4 clases
        n_clusters_per_class=1,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Comparar diferentes configuraciones
    configurations = {
        "Con optimización (por defecto)": {
            "use_weights": True,
            "epochs": 20,
            "lr": 0.1
        },
        "Sin optimización": {
            "use_weights": False,
            "epochs": 1,  # No importa cuando no hay optimización
            "lr": 0.1     # No se usa cuando no hay optimización
        },
        "Sin optimización + pesos específicos": {
            "use_weights": False,
            "epochs": 1,
            "lr": 0.1,
            "weights": np.array([0.1, -0.2, 0.15])
        }
    }
    
    results = {}
    
    for config_name, config in configurations.items():
        print(f"\n🔮 Configuración: {config_name}")
        
        model = QMLMultiClase(
            codigo="gray",
            noise=0.01,
            random_state=42,
            verbose=True,
            **config
        )
        
        print(f"Entrenando con configuración: {config}")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[config_name] = {
            'accuracy': accuracy,
            'loss': model.best_loss_,
            'config': config
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Loss: {model.best_loss_:.6f}")
    
    # Comparar resultados
    print("\n📊 Comparación de Resultados:")
    for config_name, result in results.items():
        print(f"{config_name}: {result['accuracy']:.4f} (Loss: {result['loss']:.6f})")
    
    # Análisis de diferencias
    opt_result = results["Con optimización (por defecto)"]
    no_opt_result = results["Sin optimización"]
    no_opt_specific_result = results["Sin optimización + pesos específicos"]
    
    print(f"Diferencia (optimizado vs sin optimizar): {abs(opt_result['accuracy'] - no_opt_result['accuracy']):.4f}")
    print(f"Diferencia (optimizado vs sin optimizar + pesos): {abs(opt_result['accuracy'] - no_opt_specific_result['accuracy']):.4f}")
    print(f"Diferencia (sin optimizar vs sin optimizar + pesos): {abs(no_opt_result['accuracy'] - no_opt_specific_result['accuracy']):.4f}")
    
    # Demostrar casos de uso prácticos
    print("\n🎯 Casos de Uso Prácticos:")
    print("\n1️⃣ Evaluación de Circuito Base:")
    print("   - Útil para evaluar el rendimiento del circuito cuántico sin entrenamiento")
    print("   - Permite comparar diferentes arquitecturas de circuito")
    print("   - Útil para análisis de capacidad expresiva del circuito")
    
    model_base = QMLMultiClase(
        codigo="gray",
        use_weights=False,
        verbose=False
    )
    model_base.fit(X, y)
    print(f"   Resultado: Accuracy base = {accuracy_score(y, model_base.predict(X)):.4f}")
    
    print("\n2️⃣ Benchmark de Inicialización:")
    print("   - Comparar diferentes estrategias de inicialización")
    print("   - Evaluar el impacto de los pesos iniciales")
    
    strategies = {
        "Aleatorio": None,  # Usar inicialización por defecto
        "Aleatorio pequeño": np.random.random(3) * 0.1,
        "Valores específicos": np.array([0.1, -0.2, 0.15])
    }
    
    for strategy_name, init_params in strategies.items():
        model = QMLMultiClase(
            codigo="gray",
            use_weights=False,
            weights=init_params,
            verbose=False
        )
        model.fit(X, y)
        acc = accuracy_score(y, model.predict(X))
        print(f"   {strategy_name}: {acc:.4f}")
    
    print("\n3️⃣ Análisis de Ruido:")
    print("   - Comparar rendimiento con y sin ruido")
    print("   - Análisis de robustez del circuito")
    
    noise_levels = [0.0, 0.01, 0.05, 0.1]
    for noise in noise_levels:
        model = QMLMultiClase(
            codigo="gray",
            use_weights=False,
            noise=noise,
            verbose=False
        )
        model.fit(X, y)
        acc = accuracy_score(y, model.predict(X))
        print(f"   Ruido {noise}: {acc:.4f}")
    
    print("\n4️⃣ Comparación de Codificaciones:")
    print("   - Comparar rendimiento base de cada codificación")
    print("   - Análisis de capacidad expresiva por codificación")
    
    codigos = ["gray", "binario", "diag"]
    for codigo in codigos:
        model = QMLMultiClase(
            codigo=codigo,
            use_weights=False,
            verbose=False
        )
        model.fit(X, y)
        acc = accuracy_score(y, model.predict(X))
        print(f"   Codificación {codigo}: {acc:.4f}")

    # Análisis de rendimiento y tiempo
    print("\n⏱️  Análisis de Rendimiento:")
    print("   - Comparar tiempo de entrenamiento con y sin optimización")
    print("   - Evaluar eficiencia computacional")
    
    import time
    
    # Medir tiempo de entrenamiento
    configs = {
        "Con optimización (20 epochs)": {"use_weights": True, "epochs": 20},
        "Con optimización (5 epochs)": {"use_weights": True, "epochs": 5},
        "Sin optimización": {"use_weights": False, "epochs": 1}
    }
    
    for config_name, config in configs.items():
        print(f"\n🕐 {config_name}:")
        
        start_time = time.time()
        model = QMLMultiClase(
            codigo="gray",
            verbose=False,
            **config
        )
        model.fit(X_train, y_train)
        end_time = time.time()
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"   Tiempo: {end_time - start_time:.2f} segundos")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Loss: {model.best_loss_:.6f}")
    
    print("\n✅ Ejemplo completado exitosamente!")
    print("\nCaracterísticas demostradas:")
    print("- Clasificador sin optimización de pesos")
    print("- Comparación de rendimiento con y sin optimización")
    print("- Casos de uso prácticos")
    print("- Análisis de rendimiento y tiempo")
    print("- Validación de diferentes configuraciones")


if __name__ == "__main__":
    main()
