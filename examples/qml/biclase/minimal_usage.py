"""
Minimal example usage of the Quantum Machine Learning Library.

This example uses a very small dataset to test basic functionality.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from quantum_metric.models import QMLBiClase


def main():
    """Main example function."""
    
    print("🚀 Quantum Machine Learning Library - Minimal Example")
    print("=" * 60)
    
    # Generate very small synthetic binary classification data
    print("\n📊 Generating minimal dataset...")
    X, y = make_classification(
        n_samples=8,   # Very small dataset
        n_features=2,  # Only 2 features
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
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Initialize quantum classifier
    print("\n🔮 Initializing Quantum Binary Classifier...")
    model = QMLBiClase(
        codigo="gray",           # Data encoding method
        noise=0.0,               # No noise for compatibility
        epochs=5,                # Very few epochs
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
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Show predictions
        print("\n🎯 Predictions:")
        for i in range(len(X_test)):
            print(f"Sample {i}: True={y_test[i]}, Pred={y_pred[i]}")
        
        print("\n✅ Minimal example completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
