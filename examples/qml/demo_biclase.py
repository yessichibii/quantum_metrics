from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
from quantum_metric.validation.cross_validation import cross_validate_biclase
from quantum_metric.utils.preprocesamiento import to_one_hot
from quantum_metric import QMLBiClase
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report


def main():
    print("🌸 Cargando dataset Iris...")

    iris = load_iris()
    X = iris.data
    y = iris.target

    # Estandarizamos los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convertimos etiquetas a enteros si es necesario
    y_encoded = LabelEncoder().fit_transform(y)

    # Tomamos solo dos clases (binaria) para este ejemplo
    mask = y_encoded < 2
    X_bin = X_scaled[mask]
    y_bin = y_encoded[mask]
    y_bin_oh = y_bin
    print(f"✅ Datos cargados: {X_bin.shape[0]} muestras, {X_bin.shape[1]} características")

    model = QMLBiClase(
        codigo="gray",
        noise=0.0,
        backend=None,
        noise_model=None,
        result="probs",
        shots=1024,
        lr=0.1,
        epochs=10,   # para demo, pocas epochs
        optimize=True,
        rng=0.5
    )

    # Usamos validación cruzada K-Fold
    print("\n🚀 Ejecutando validación cruzada cuántica...")
    result = cross_validate_biclase(
        model=model,
        data=X_bin,
        labels=y_bin_oh,
        k_folds=3
    )

    print("\n📊 Resultados:")
    print(f" - Parámetros óptimos: {result['best_params']}")
    print(f" - preds: {result['preds']}")
    print(f" - y_val: {result['y_val']}")

    preds = np.array(result['preds'], dtype=int)
    y_val = np.array(result['y_val'], dtype=int)

    matriz = confusion_matrix(y_val, preds)
    accuracy = accuracy_score(y_val, preds)
    precision = precision_score(y_val, preds)
    recall = recall_score(y_val, preds)
    f1 = f1_score(y_val, preds)

    print("Matriz de confusión:")
    print(matriz)
    print("\nExactitud (Accuracy):", accuracy)
    print("Precisión:", precision)
    print("Recall:", recall)
    print("F1:", f1)

    # Opcional: reporte completo
    print("\nReporte de clasificación:")
    print(classification_report(y_val, preds))



    model2 = QMLBiClase(
        codigo="gray",
        noise=0.0,
        backend=None,
        noise_model=None,
        result="probs",
        shots=1024,
        lr=0.1,
        epochs=10,   # para demo, pocas epochs
        optimize=True,
        rng=0.5,
        k_folds=3, test_size=0.2
    )
    model2.fit_validation(X_bin, y_bin_oh)
    preds2 = model2.predict()



    print("\n📊 Resultados:")
    print(f" - Parámetros óptimos: {model2.best_params_}")
    print(f" - preds: {preds2}")
    print(f" - y_val: {model2.val_labels_}")

    preds2a = np.array(preds2, dtype=int)
    y_val2 = np.array(model2.val_labels_, dtype=int)


    matriz = confusion_matrix(y_val2, preds2a)
    accuracy = accuracy_score(y_val2, preds2a)
    precision = precision_score(y_val2, preds2a)
    recall = recall_score(y_val2, preds2a)
    f1 = f1_score(y_val2, preds2a)

    print("Matriz de confusión:")
    print(matriz)
    print("\nExactitud (Accuracy):", accuracy)
    print("Precisión:", precision)
    print("Recall:", recall)
    print("F1:", f1)

    # Opcional: reporte completo
    print("\nReporte de clasificación:")
    print(classification_report(y_val2, preds2a))


    

    model3 = QMLBiClase(
        codigo="gray",
        noise=0.0,
        backend=None,
        noise_model=None,
        result="probs",
        shots=1024,
        lr=0.1,
        epochs=10,   # para demo, pocas epochs
        optimize=True,
        rng=0.5,
        k_folds=3, test_size=0.2
    )
    model3.fit_validation(X_bin, y_bin_oh)
    preds3 = model2.predict(method="mean")



    print("\n📊 Resultados:")
    print(f" - Parámetros óptimos: {model3.best_params_}")
    print(f" - preds: {preds3}")
    print(f" - y_val: {model3.val_labels_}")

    preds3a = np.array(preds3, dtype=int)
    y_val3 = np.array(model3.val_labels_, dtype=int)


    matriz = confusion_matrix(y_val3, preds3a)
    accuracy = accuracy_score(y_val3, preds3a)
    precision = precision_score(y_val3, preds3a)
    recall = recall_score(y_val3, preds3a)
    f1 = f1_score(y_val3, preds3a)

    print("Matriz de confusión:")
    print(matriz)
    print("\nExactitud (Accuracy):", accuracy)
    print("Precisión:", precision)
    print("Recall:", recall)
    print("F1:", f1)

    # Opcional: reporte completo
    print("\nReporte de clasificación:")
    print(classification_report(y_val3, preds3a))


if __name__ == "__main__":
    main()