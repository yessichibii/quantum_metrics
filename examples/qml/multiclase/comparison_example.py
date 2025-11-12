"""
Ejemplo comparativo entre QMLBiClase y QMLMultiClase.

Este ejemplo demuestra las diferencias entre el clasificador binario y multiclase,
mostrando las diferencias en la arquitectura del circuito cuántico.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from quantum_metric.models import QMLBiClase, QMLMultiClase


def main():
    """Función principal que demuestra las diferencias entre los modelos."""
    
    print("🎯 Comparación entre QMLBiClase y QMLMultiClase")
    print("=" * 60)
    
    # Generar datos para clasificación binaria
    print("\n📊 Generando datos para clasificación binaria...")
    X_binary, y_binary = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X_binary, y_binary, test_size=0.3, random_state=42, stratify=y_binary
    )
    
    # Generar datos para clasificación multiclase
    print("\n📊 Generando datos para clasificación multiclase...")
    X_multi, y_multi = make_classification(
        n_samples=120,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42
    )
    
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_multi, y_multi, test_size=0.3, random_state=42, stratify=y_multi
    )
    
    print(f"Binary dataset: {X_binary.shape}, Classes: {np.unique(y_binary)}")
    print(f"Multi-class dataset: {X_multi.shape}, Classes: {np.unique(y_multi)}")
    
    # Entrenar modelo binario
    print("\n🔵 Entrenando modelo QMLBiClase...")
    model_binary = QMLBiClase(
        codigo="gray",
        epochs=15,
        lr=0.1,
        random_state=42,
        verbose=True
    )
    
    model_binary.fit(X_train_bin, y_train_bin)
    y_pred_bin = model_binary.predict(X_test_bin)
    accuracy_bin = accuracy_score(y_test_bin, y_pred_bin)
    
    print(f"Binary model accuracy: {accuracy_bin:.4f}")
    print(f"Binary model loss: {model_binary.best_loss_:.6f}")
    
    # Entrenar modelo multiclase
    print("\n🟡 Entrenando modelo QMLMultiClase...")
    model_multiclass = QMLMultiClase(
        codigo="gray",
        epochs=15,
        lr=0.1,
        random_state=42,
        verbose=True
    )
    
    model_multiclass.fit(X_train_multi, y_train_multi)
    y_pred_multi = model_multiclass.predict(X_test_multi)
    accuracy_multi = accuracy_score(y_test_multi, y_pred_multi)
    
    print(f"Multi-class model accuracy: {accuracy_multi:.4f}")
    print(f"Multi-class model loss: {model_multiclass.best_loss_:.6f}")
    
    # Comparar arquitecturas de circuito
    print("\n🏗️  Comparación de Arquitecturas de Circuito:")
    print("\nQMLBiClase:")
    print("  - Circuito: qubits_qram + qubits_dato + 2 qubits")
    print("  - Incluye CNOT después del bucle de rotación")
    print("  - Optimizado para clasificación binaria")
    print("  - Salida: 2 clases")
    
    print("\nQMLMultiClase:")
    print("  - Circuito: qubits_qram + qubits_dato + qubits_label + 1 qubit")
    print("  - NO incluye CNOT después del bucle de rotación")
    print("  - Optimizado para clasificación multiclase")
    print("  - Salida: múltiples clases")
    
    # Demostrar diferencias en parámetros de circuito
    print("\n🔧 Parámetros de Circuito:")
    
    # Para modelo binario
    m_bin, qubits_dato_bin = X_train_bin.shape
    qubits_qram_bin = int(np.ceil(np.log2(m_bin)))
    n_totales_bin = qubits_qram_bin + qubits_dato_bin + 2
    
    print(f"\nQMLBiClase:")
    print(f"  - Muestras de entrenamiento: {m_bin}")
    print(f"  - Qubits de datos: {qubits_dato_bin}")
    print(f"  - Qubits QRAM: {qubits_qram_bin}")
    print(f"  - Total de qubits: {n_totales_bin}")
    
    # Para modelo multiclase
    m_multi, qubits_dato_multi = X_train_multi.shape
    qubits_qram_multi = int(np.ceil(np.log2(m_multi)))
    qubits_label_multi = int(np.ceil(np.log2(len(np.unique(y_multi)))))
    n_totales_multi = qubits_qram_multi + qubits_dato_multi + qubits_label_multi + 1
    
    print(f"\nQMLMultiClase:")
    print(f"  - Muestras de entrenamiento: {m_multi}")
    print(f"  - Qubits de datos: {qubits_dato_multi}")
    print(f"  - Qubits QRAM: {qubits_qram_multi}")
    print(f"  - Qubits de etiquetas: {qubits_label_multi}")
    print(f"  - Total de qubits: {n_totales_multi}")
    
    # Comparar rendimiento con diferentes números de clases
    print("\n📊 Análisis de Rendimiento por Número de Clases:")
    
    for n_classes in [2, 3, 4]:
        print(f"\n🔢 {n_classes} clases:")
        
        # Generar datos
        X_temp, y_temp = make_classification(
            n_samples=80,
            n_features=3,
            n_informative=2,
            n_redundant=1,
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=42
        )
        
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
            X_temp, y_temp, test_size=0.3, random_state=42, stratify=y_temp
        )
        
        # Entrenar modelo multiclase
        model_temp = QMLMultiClase(
            codigo="gray",
            epochs=10,
            lr=0.1,
            random_state=42,
            verbose=False
        )
        
        model_temp.fit(X_train_temp, y_train_temp)
        y_pred_temp = model_temp.predict(X_test_temp)
        accuracy_temp = accuracy_score(y_test_temp, y_pred_temp)
        
        print(f"  Accuracy: {accuracy_temp:.4f}")
        print(f"  Loss: {model_temp.best_loss_:.6f}")
        
        # Calcular parámetros de circuito
        m_temp, qubits_dato_temp = X_train_temp.shape
        qubits_qram_temp = int(np.ceil(np.log2(m_temp)))
        qubits_label_temp = int(np.ceil(np.log2(n_classes)))
        n_totales_temp = qubits_qram_temp + qubits_dato_temp + qubits_label_temp + 1
        
        print(f"  Total de qubits: {n_totales_temp}")
    
    n_classes = 5
    print(f"\n🔢 {n_classes} clases:")
        
        # Generar datos
    X_temp, y_temp = make_classification(
        n_samples=80,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=42
    )
        
    X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
        X_temp, y_temp, test_size=0.3, random_state=42, stratify=y_temp
    )
        
        # Entrenar modelo multiclase
    model_temp = QMLMultiClase(
        codigo="gray",
        epochs=10,
        lr=0.1,
        random_state=42,
        verbose=False
    )
        
    model_temp.fit(X_train_temp, y_train_temp)
    y_pred_temp = model_temp.predict(X_test_temp)
    accuracy_temp = accuracy_score(y_test_temp, y_pred_temp)
    
    print(f"  Accuracy: {accuracy_temp:.4f}")
    print(f"  Loss: {model_temp.best_loss_:.6f}")
        
    # Calcular parámetros de circuito
    m_temp, qubits_dato_temp = X_train_temp.shape
    qubits_qram_temp = int(np.ceil(np.log2(m_temp)))
    qubits_label_temp = int(np.ceil(np.log2(n_classes)))
    n_totales_temp = qubits_qram_temp + qubits_dato_temp + qubits_label_temp + 1
        
    print(f"  Total de qubits: {n_totales_temp}")

    # Demostrar diferencias en predicciones de probabilidad
    print("\n🎲 Comparación de Predicciones de Probabilidad:")
    
    print("\nQMLBiClase (primera muestra):")
    proba_bin = model_binary.predict_proba(X_test_bin[:1])
    print(f"  Probabilidades: {proba_bin[0]}")
    print(f"  Suma: {np.sum(proba_bin[0]):.6f}")
    
    print("\nQMLMultiClase (primera muestra):")
    proba_multi = model_multiclass.predict_proba(X_test_multi[:1])
    print(f"  Probabilidades: {proba_multi[0]}")
    print(f"  Suma: {np.sum(proba_multi[0]):.6f}")
    
    # Resumen de diferencias clave
    print("\n📋 Resumen de Diferencias Clave:")
    print("\n1. Arquitectura del Circuito:")
    print("   - QMLBiClase: Incluye CNOT después del bucle de rotación")
    print("   - QMLMultiClase: NO incluye CNOT después del bucle de rotación")
    
    print("\n2. Número de Qubits:")
    print("   - QMLBiClase: qubits_qram + qubits_dato + 2")
    print("   - QMLMultiClase: qubits_qram + qubits_dato + qubits_label + 1")
    
    print("\n3. Aplicaciones:")
    print("   - QMLBiClase: Clasificación binaria (2 clases)")
    print("   - QMLMultiClase: Clasificación multiclase (múltiples clases)")
    
    print("\n4. Complejidad:")
    print("   - QMLBiClase: Más simple, optimizado para 2 clases")
    print("   - QMLMultiClase: Más plantable, maneja múltiples clases")
    
    print("\n✅ Comparación completada exitosamente!")


if __name__ == "__main__":
    main()
