"""
Simple example usage of the Quantum Machine Learning Library.

This example demonstrates how to use the QMLBiClase classifier
with a simple dataset and no noise to avoid compatibility issues.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from quantum_metric.models import QMLBiClase


def main():
    """Main example function."""
    
    print("🚀 Quantum Machine Learning Library - Simple Example")
    print("=" * 60)
    
    # Generate synthetic binary classification data
    print("\n📊 Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=50,  # Smaller dataset for faster execution
        n_features=2,  # Fewer features for simpler quantum circuit
        n_redundant=0,
        n_informative=2,
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
    
    # Initialize quantum classifier (no noise for compatibility)
    print("\n🔮 Initializing Quantum Binary Classifier...")
    model = QMLBiClase(
        codigo="gray",           # Data encoding method
        noise=0.0,               # No noise for compatibility
        epochs=10,               # Fewer epochs for faster execution
        lr=0.1,                  # Learning rate
        shots=1024,              # Number of quantum measurements
        random_state=42,         # For reproducibility
        verbose=True             # Show training progress
    )
    
    print(f"Model parameters: {model.get_params()}")
    
    # Train the model
    print("\n🏋️ Training the quantum model...")
    try:
        model.fit(X_train, y_train)
        print(f"Training completed!")
        print(f"Best loss: {model.best_loss_:.6f}")
        print(f"Training history length: {len(model.history_)}")
        
        # Make predictions
        print("\n🔮 Making predictions...")
        print(f"X_test: {X_test}")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Print detailed classification report
        print("\n📈 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Show probability predictions for first few samples
        print("\n🎯 Probability predictions (first 5 samples):")
        for i in range(min(5, len(X_test))):
            print(f"Sample {i}: True={y_test[i]}, Pred={y_pred[i]}, "
                  f"Probs={y_proba[i]}")
        
        # Demonstrate scikit-learn compatibility
        print("\n🔧 Scikit-learn compatibility test...")
        
        # Test score method
        score = model.score(X_test, y_test)
        print(f"Model score: {score:.4f}")
        
        print("\n✅ Example completed successfully!")
        print("\nKey features demonstrated:")
        print("- scikit-learn compatible interface")
        print("- Robust parameter validation")
        print("- Probability predictions")
        print("- Comprehensive error handling")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        print("This might be due to quantum circuit complexity or compatibility issues.")
        print("Try reducing the dataset size or circuit complexity.")


if __name__ == "__main__":
    main()
