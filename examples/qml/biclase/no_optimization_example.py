"""
Ejemplo específico del uso del clasificador sin optimización de pesos.

Este ejemplo demuestra cómo usar el clasificador cuántico sin optimización
de pesos, útil para evaluar el rendimiento base del circuito cuántico.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from quantum_metric.models import QMLBiClase


def demonstrate_no_optimization():
    """Demuestra el uso del clasificador sin optimización."""
    
    print("🎯 Ejemplo de Clasificador Sin Optimización")
    print("=" * 60)
    
    # Generar datos sintéticos
    print("\n📊 Generando dataset sintético...")
    X, y = make_classification(
        n_samples=100,
        n_features=3,
        n_redundant=0,
        n_informative=3,
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
        print(f"Parámetros: {config}")
        
        model = QMLBiClase(
            codigo="gray",
            noise=0.01,
            random_state=42,
            verbose=False,
            **config
        )
        
        # Entrenar modelo
        model.fit(X_train, y_train)
        
        # Hacer predicciones
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        
        results[config_name] = {
            'model': model,
            'accuracy': accuracy,
            'loss': model.best_loss_,
            'history_length': len(model.history_),
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Loss: {model.best_loss_:.6f}")
        print(f"History length: {len(model.history_)}")
    
    # Comparación detallada
    print("\n📊 Comparación Detallada:")
    print("-" * 80)
    print(f"{'Configuración':<40} {'Accuracy':<10} {'Loss':<12} {'History':<8}")
    print("-" * 80)
    
    for config_name, result in results.items():
        print(f"{config_name:<40} {result['accuracy']:<10.4f} {result['loss']:<12.6f} {result['history_length']:<8}")
    
    # Análisis de diferencias
    print("\n🔍 Análisis de Diferencias:")
    opt_result = results["Con optimización (por defecto)"]
    no_opt_result = results["Sin optimización"]
    no_opt_specific_result = results["Sin optimización + pesos específicos"]
    
    print(f"Diferencia (optimizado vs sin optimizar): {abs(opt_result['accuracy'] - no_opt_result['accuracy']):.4f}")
    print(f"Diferencia (optimizado vs sin optimizar específico): {abs(opt_result['accuracy'] - no_opt_specific_result['accuracy']):.4f}")
    print(f"Diferencia (sin optimizar vs sin optimizar específico): {abs(no_opt_result['accuracy'] - no_opt_specific_result['accuracy']):.4f}")
    
    # Reporte de clasificación para el modelo sin optimización
    print("\n📈 Reporte de Clasificación - Sin Optimización:")
    print(classification_report(y_test, no_opt_result['predictions']))
    
    # Matriz de confusión
    print("\n🎯 Matriz de Confusión - Sin Optimización:")
    cm = confusion_matrix(y_test, no_opt_result['predictions'])
    print(cm)
    
    # Análisis de probabilidades
    print("\n🎲 Análisis de Probabilidades - Sin Optimización:")
    probas = no_opt_result['probabilities']
    print(f"Probabilidades promedio clase 0: {np.mean(probas[:, 0]):.4f}")
    print(f"Probabilidades promedio clase 1: {np.mean(probas[:, 1]):.4f}")
    print(f"Desviación estándar probabilidades: {np.std(probas):.4f}")
    
    return results


def demonstrate_use_cases():
    """Demuestra casos de uso para el clasificador sin optimización."""
    
    print("\n🎯 Casos de Uso del Clasificador Sin Optimización")
    print("=" * 60)
    
    # Datos de prueba
    X = np.random.random((50, 2))
    y = np.random.randint(0, 2, 50)
    
    print("\n1️⃣ Evaluación de Circuito Base:")
    print("   - Útil para evaluar el rendimiento del circuito cuántico sin entrenamiento")
    print("   - Permite comparar diferentes arquitecturas de circuito")
    print("   - Útil para análisis de capacidad expresiva del circuito")
    
    model_base = QMLBiClase(
        codigo="gray",
        use_weights=False,
        verbose=False
    )
    model_base.fit(X, y)
    print(f"   Resultado: Accuracy base = {accuracy_score(y, model_base.predict(X)):.4f}")
    
    print("\n2️⃣ Benchmark de Inicialización:")
    print("   - Comparar diferentes estrategias de inicialización")
    print("   - Evaluar el impacto de parámetros iniciales específicos")
    print("   - Análisis de sensibilidad a la inicialización")
    
    strategies = {
        "Ceros": np.zeros(2),
        "Unos": np.ones(2),
        "Aleatorio pequeño": np.random.random(2) * 0.1,
        "Valores específicos": np.array([0.1, -0.2])
    }
    
    for strategy_name, init_params in strategies.items():
        model = QMLBiClase(
            codigo="gray",
            use_weights=False,
            weights=init_params,
            verbose=False
        )
        model.fit(X, y)
        acc = accuracy_score(y, model.predict(X))
        print(f"   {strategy_name}: {acc:.4f}")
    
    print("\n3️⃣ Análisis de Ruido:")
    print("   - Evaluar el impacto del ruido en el circuito base")
    print("   - Comparar rendimiento con y sin ruido")
    print("   - Análisis de robustez del circuito")
    
    noise_levels = [0.0, 0.01, 0.05, 0.1]
    for noise in noise_levels:
        model = QMLBiClase(
            codigo="gray",
            use_weights=False,
            noise=noise,
            verbose=False
        )
        model.fit(X, y)
        acc = accuracy_score(y, model.predict(X))
        print(f"   Ruido {noise}: {acc:.4f}")
    
    print("\n4️⃣ Comparación de Codificaciones:")
    print("   - Evaluar diferentes métodos de codificación")
    print("   - Comparar rendimiento base de cada codificación")
    print("   - Análisis de capacidad expresiva por codificación")
    
    codigos = ["gray", "binario", "diag"]
    for codigo in codigos:
        model = QMLBiClase(
            codigo=codigo,
            use_weights=False,
            verbose=False
        )
        model.fit(X, y)
        acc = accuracy_score(y, model.predict(X))
        print(f"   Codificación {codigo}: {acc:.4f}")


def demonstrate_performance_comparison():
    """Demuestra comparación de rendimiento entre diferentes configuraciones."""
    
    print("\n⚡ Comparación de Rendimiento")
    print("=" * 50)
    
    # Datos más grandes para mejor evaluación
    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_redundant=0,
        n_informative=4,
        n_clusters_per_class=1,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
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
        model = QMLBiClase(
            codigo="gray",
            random_state=42,
            verbose=False,
            **config
        )
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        start_time = time.time()
        predictions = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"   Tiempo de entrenamiento: {training_time:.4f}s")
        print(f"   Tiempo de predicción: {prediction_time:.4f}s")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Loss: {model.best_loss_:.6f}")


if __name__ == "__main__":
    results = demonstrate_no_optimization()
    demonstrate_use_cases()
    demonstrate_performance_comparison()
    
    print("\n✅ Ejemplo completado exitosamente!")
    print("\nCaracterísticas demostradas:")
    print("- Clasificador sin optimización")
    print("- Comparación con optimización")
    print("- Casos de uso prácticos")
    print("- Análisis de rendimiento")
    print("- Evaluación de diferentes configuraciones")
