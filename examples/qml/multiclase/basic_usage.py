"""
Ejemplo básico de uso del modelo QMLMultiClase.

Este ejemplo demuestra cómo usar el clasificador cuántico multiclase
para tareas de clasificación con múltiples clases.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from quantum_metric.models import QMLMultiClase


def main():
    """Función principal que demuestra el uso del modelo QMLMultiClase."""
    
    print("🎯 Ejemplo de QMLMultiClase - Clasificador Cuántico Multiclase")
    print("=" * 70)
    
    # Generar datos sintéticos para clasificación multiclase
    print("\n📊 Generando dataset sintético multiclase...")
    X, y = make_classification(
        n_samples=150,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=3,  # 3 clases
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Classes: {np.unique(y)}")
    
    # Crear y entrenar el modelo básico
    print("\n🚀 Creando modelo QMLMultiClase...")
    model = QMLMultiClase(
        codigo="gray",           # Método de codificación
        noise=0.01,             # Nivel de ruido
        epochs=20,              # Número de épocas
        lr=0.1,                 # Tasa de aprendizaje
        shots=1024,             # Número de mediciones cuánticas
        random_state=42,        # Para reproducibilidad
        verbose=True            # Mostrar progreso del entrenamiento
    )
    
    # Demonstrate custom weights
    print("\n🎯 Demonstrating custom weights...")
    initial_weights = np.random.random(X_train.shape[1]) * 0.1  # Small random weights
    print(f"Custom weights: {initial_weights}")
    
    model_with_weights = QMLMultiClase(
        codigo="gray",
        noise=0.01,
        epochs=20,
        lr=0.1,
        shots=1024,
        random_state=42,
        verbose=True,
        weights=initial_weights  # Custom weights
    )
    
    print(f"Model parameters: {model.get_params()}")
    
    # Train the model
    print("\n🏋️ Training the quantum model...")
    model.fit(X_train, y_train)
    
    print(f"Training completed!")
    print(f"Best loss: {model.best_loss_:.6f}")
    print(f"Training history length: {len(model.history_)}")
    
    # Train model with custom weights
    print("\n🏋️ Training model with custom weights...")
    model_with_weights.fit(X_train, y_train)
    
    print(f"Custom weights training completed!")
    print(f"Best loss (custom weights): {model_with_weights.best_loss_:.6f}")
    print(f"Training history length (custom weights): {len(model_with_weights.history_)}")
    
    # Demonstrate classifier without weight optimization
    print("\n🎯 Demonstrating classifier without weight optimization...")
    model_no_weights = QMLMultiClase(
        codigo="gray",
        noise=0.01,
        epochs=20,
        lr=0.1,
        shots=1024,
        random_state=42,
        verbose=True,
        use_weights=False  # No weight optimization
    )
    
    print("\n🏋️ Running classifier without weight optimization...")
    model_no_weights.fit(X_train, y_train)
    
    print(f"No weight optimization completed!")
    print(f"Best loss (no weights): {model_no_weights.best_loss_:.6f}")
    print(f"Training history length (no weights): {len(model_no_weights.history_)}")
    
    # Make predictions
    print("\n🔮 Making predictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Make predictions with custom weights model
    print("\n🔮 Making predictions with custom weights model...")
    y_pred_weights = model_with_weights.predict(X_test)
    y_proba_weights = model_with_weights.predict_proba(X_test)
    
    # Make predictions with no weight optimization model
    print("\n🔮 Making predictions with no weight optimization model...")
    y_pred_no_weights = model_no_weights.predict(X_test)
    y_proba_no_weights = model_no_weights.predict_proba(X_test)
    
    # Calculate accuracies
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_weights = accuracy_score(y_test, y_pred_weights)
    accuracy_no_weights = accuracy_score(y_test, y_pred_no_weights)
    
    print(f"Test accuracy (random weights): {accuracy:.4f}")
    print(f"Test accuracy (custom weights): {accuracy_weights:.4f}")
    print(f"Test accuracy (no weights): {accuracy_no_weights:.4f}")
    
    # Compare results
    print(f"\n📊 Model Comparison:")
    print(f"Random weights accuracy: {accuracy:.4f}")
    print(f"Custom weights accuracy: {accuracy_weights:.4f}")
    print(f"No weight optimization accuracy: {accuracy_no_weights:.4f}")
    print(f"Difference (random vs custom): {abs(accuracy - accuracy_weights):.4f}")
    print(f"Difference (random vs no weights): {abs(accuracy - accuracy_no_weights):.4f}")
    
    # Print detailed classification report
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print confusion matrix
    print("\n🔢 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Print probability predictions for first few samples
    print("\n🎲 Probability Predictions (first 5 samples):")
    for i in range(min(5, len(y_proba))):
        print(f"Sample {i}: True class={y_test[i]}, Predicted class={y_pred[i]}")
        print(f"  Probabilities: {y_proba[i]}")
    
    # Demonstrate cross-validation
    print("\n🔄 Demonstrating cross-validation...")
    model_cv = QMLMultiClase(
        codigo="gray",
        noise=0.01,
        epochs=15,
        lr=0.1,
        shots=1024,
        random_state=42,
        verbose=False,
        k_folds=3  # 3-fold cross-validation
    )
    
    model_cv.fit_validation(X_train, y_train)
    y_pred_cv = model_cv.predict(X_test)
    accuracy_cv = accuracy_score(y_test, y_pred_cv)
    print(f"Cross-validated test accuracy: {accuracy_cv:.4f}")
    
    print("\n✅ Example completed successfully!")
    print("\nKey features demonstrated:")
    print("- scikit-learn compatible interface")
    print("- Robust parameter validation")
    print("- Cross-validation support")
    print("- Probability predictions")
    print("- Comprehensive error handling")
    print("- Custom weights support")
    print("- Model comparison capabilities")
    print("- Classifier without weight optimization")


if __name__ == "__main__":
    main()
