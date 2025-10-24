"""
Ejemplo específico del uso de pesos personalizados en QMLMultiClase.

Este ejemplo demuestra cómo usar pesos específicos
en lugar de inicialización aleatoria en los modelos cuánticos multiclase.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from quantum_metric.models import QMLMultiClase


def demonstrate_custom_weights():
    """Demuestra el uso de pesos personalizados."""
    
    print("🎯 Ejemplo de Pesos Personalizados - QMLMultiClase")
    print("=" * 60)
    
    # Generar datos sintéticos
    print("\n📊 Generando dataset sintético...")
    X, y = make_classification(
        n_samples=120,
        n_features=3,
        n_informative=2,
        n_redundant=1,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Classes: {np.unique(y)}")
    
    # Modelo con inicialización aleatoria (por defecto)
    print("\n🎲 Modelo con inicialización aleatoria (por defecto)...")
    model_random = QMLMultiClase(
        codigo="gray",
        epochs=15,
        lr=0.1,
        random_state=42,
        verbose=False
    )
    
    model_random.fit(X_train, y_train)
    y_pred_random = model_random.predict(X_test)
    accuracy_random = accuracy_score(y_test, y_pred_random)
    
    print(f"Accuracy (random init): {accuracy_random:.4f}")
    print(f"Loss: {model_random.best_loss_:.6f}")
    
    # Diferentes estrategias de inicialización
    print("\n🎯 Comparando diferentes estrategias de inicialización...")
    strategies = {
        "Todos ceros": np.zeros(3),
        "Valores pequeños": np.array([0.01, -0.01, 0.02]),
        "Aleatorio pequeño": np.random.random(3) * 0.1,
        "Valores específicos": np.array([0.1, -0.2, 0.15])
    }
    
    results = {}
    
    for strategy_name, weights in strategies.items():
        print(f"\n  📈 Estrategia: {strategy_name}")
        print(f"  Pesos: {weights}")
        
        model_custom = QMLMultiClase(
            codigo="gray",
            epochs=15,
            lr=0.1,
            random_state=42,
            verbose=False,
            weights=weights
        )
        
        model_custom.fit(X_train, y_train)
        y_pred_custom = model_custom.predict(X_test)
        accuracy_custom = accuracy_score(y_test, y_pred_custom)
        
        results[strategy_name] = {
            'accuracy': accuracy_custom,
            'loss': model_custom.best_loss_,
            'weights': weights
        }
        
        print(f"  Accuracy: {accuracy_custom:.4f}")
        print(f"  Pérdida final: {model_custom.best_loss_:.6f}")
    
    # Comparación de resultados
    print("\n📊 Comparación de Resultados:")
    print(f"Random init: {accuracy_random:.4f}")
    for strategy_name, result in results.items():
        print(f"{strategy_name}: {result['accuracy']:.4f}")
    
    # Encontrar la mejor estrategia
    best_strategy = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 Mejor estrategia: {best_strategy[0]} (Accuracy: {best_strategy[1]['accuracy']:.4f})")
    
    # Ejemplo de validación de parámetros incorrectos
    print("\n⚠️  Ejemplo de validación de parámetros incorrectos...")
    
    try:
        # Pesos con longitud incorrecta
        wrong_weights = np.array([0.1, 0.2])  # Solo 2 pesos cuando necesitamos 3
        model_wrong = QMLMultiClase(
            codigo="gray",
            epochs=5,
            weights=wrong_weights
        )
        model_wrong.fit(X_train, y_train)
    except ValueError as e:
        print(f"Error esperado: {e}")
    
    try:
        # Pesos con valores no finitos
        invalid_weights = np.array([0.1, np.inf, 0.3])
        model_invalid = QMLMultiClase(
            codigo="gray",
            epochs=5,
            weights=invalid_weights
        )
        model_invalid.fit(X_train, y_train)
    except ValueError as e:
        print(f"Error esperado: {e}")
    
    print("\n✅ Ejemplo completado exitosamente!")
    print("\nCaracterísticas demostradas:")
    print("- Uso de pesos personalizados")
    print("- Comparación de diferentes estrategias de inicialización")
    print("- Validación automática de pesos")
    print("- Manejo de errores con mensajes informativos")


def demonstrate_weight_validation():
    """Demuestra la validación de pesos."""
    
    print("\n🔍 Demostración de Validación de Pesos")
    print("=" * 50)
    
    # Datos de prueba
    X = np.random.random((20, 2))
    y = np.random.randint(0, 3, 20)
    
    # Casos de prueba para validación
    test_cases = [
        {
            "name": "Pesos válidos",
            "weights": np.array([0.1, -0.2]),
            "should_fail": False
        },
        {
            "name": "Lista válida",
            "weights": [0.1, -0.2],
            "should_fail": False
        },
        {
            "name": "Longitud incorrecta",
            "weights": np.array([0.1]),
            "should_fail": True
        },
        {
            "name": "Valores no finitos",
            "weights": np.array([0.1, np.nan]),
            "should_fail": True
        },
        {
            "name": "Valores infinitos",
            "weights": np.array([0.1, np.inf]),
            "should_fail": True
        },
        {
            "name": "Tipo incorrecto",
            "weights": "invalid",
            "should_fail": True
        }
    ]
    
    for case in test_cases:
        print(f"\n🧪 Caso: {case['name']}")
        print(f"Pesos: {case['weights']}")
        
        try:
            model = QMLMultiClase(
                codigo="gray",
                epochs=2,
                weights=case['weights']
            )
            model.fit(X, y)
            
            if case['should_fail']:
                print("❌ ERROR: Debería haber fallado pero no falló")
            else:
                print("✅ Éxito: Pesos válidos aceptados")
                
        except Exception as e:
            if case['should_fail']:
                print(f"✅ Éxito: Error capturado correctamente - {e}")
            else:
                print(f"❌ ERROR: No debería haber fallado - {e}")


if __name__ == "__main__":
    demonstrate_custom_weights()
    demonstrate_weight_validation()
