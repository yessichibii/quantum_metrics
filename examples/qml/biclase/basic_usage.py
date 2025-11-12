"""
Example usage of the Quantum Machine Learning Library.

This example demonstrates how to use the QMLBiClase classifier
following scikit-learn conventions.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from quantum_metric.models import QMLBiClase


def main():
    """Main example function."""
    
    print("🚀 Quantum Machine Learning Library Example")
    print("=" * 50)
    
    # Generate synthetic binary classification data
    print("\n📊 Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_redundant=0,
        n_informative=4,
        n_clusters_per_class=1,
        random_state=42
    )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {np.unique(y)}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Initialize quantum classifier
    print("\n🔮 Initializing Quantum Binary Classifier...")
    model = QMLBiClase(
        codigo="gray",           # Data encoding method
        noise=0.01,              # Small noise level
        epochs=20,                # Number of training epochs
        lr=0.1,                  # Learning rate
        shots=1024,              # Number of quantum measurements
        random_state=42,         # For reproducibility
        verbose=True             # Show training progress
    )
    
    # Demonstrate custom weights
    print("\n🎯 Demonstrating custom weights...")
    initial_weights = np.random.random(X_train.shape[1]) * 0.1  # Small random weights
    print(f"Custom weights: {initial_weights}")
    
    model_with_weights = QMLBiClase(
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
    model_no_weights = QMLBiClase(
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
    
    # Show probability predictions for first few samples
    print("\n🎯 Probability predictions (first 5 samples):")
    for i in range(min(5, len(X_test))):
        print(f"Sample {i}: True={y_test[i]}, Pred={y_pred[i]}, Probs={y_proba[i]}")
    
    # Demonstrate scikit-learn compatibility
    print("\n🔧 Scikit-learn compatibility test...")
    
    # Test score method
    score = model.score(X_test, y_test)
    print(f"Model score: {score:.4f}")
    
    # Test parameter setting
    model.set_params(epochs=10, lr=0.05)
    print(f"Updated parameters: {model.get_params()}")
    
    # Test with validation
    print("\n🔄 Cross-validation example...")
    model_cv = QMLBiClase(
        codigo="gray",
        epochs=15,
        k_folds=3,
        random_state=42,
        verbose=False
    )
    
    model_cv.fit_validation(X_train, y_train)
    
    # Make predictions using cross-validated model
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
