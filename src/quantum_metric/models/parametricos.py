import pennylane as qml
from qiskit_aer.noise import NoiseModel
from pennylane import numpy as np
import autograd.numpy as anp
from typing import Optional, Union
from sklearn.model_selection import KFold, train_test_split
from ..quantum.distances import mba_distance, distance
from ..quantum.circuits import make_device, build_biclase_qnode
from ..utils.params import init_params, cross_entropy
import warnings


def biclase_param(train, test, labels=None, codigo="gray", noise = 0.0, backend = None, noise_model = None, result = "probs", shots = 1024, parametros_iniciales=None, optimize=False, epochs=50, lr=0.1, rng: float = 0.5,):

    m, qubits_dato = train.shape
    qubits_qram= int(np.ceil(np.log2(m)))
    n_totales = qubits_qram + qubits_dato
    n_totales += 2 
            
    dev = make_device(n_totales, backend=backend, shots=shots, noise_model=noise_model)

    circuit = build_biclase_qnode(train, test, n_totales, qubits_qram, qubits_dato, dev, codigo = codigo, noise=noise, result=result)

    params = init_params(qubits_dato, init=parametros_iniciales, rng=rng)
    
    if not optimize:
        preds = circuit(parametros_iniciales)
        return preds, None

    opt = qml.AdamOptimizer(stepsize=lr)

    def cost(p,labels):
        probs = circuit(p)
        return cross_entropy(labels, probs)
    
    loss_cost = []
    predictionsByEpoch = []
    for _ in range(epochs):
        params, cost_new = opt.step_and_cost(lambda v: cost(v,labels), params)
        loss_cost.append(cost_new)
        predictions = circuit(params)
        predictionsByEpoch.append(predictions)

    return predictionsByEpoch, params

def biclase_fit( data, labels, k_folds = 5, holdout_frac = 0.2, partitions = None, codigo = "gray", noise = 0.0, backend = None, noise_model = None, result = "probs", shots = 1024, parametros_iniciales = None, epochs = 50, lr = 0.1, rng = 0.5, optimize = True):
    """
    Ejecuta validación cruzada (K-Fold o LOOCV) usando el modelo cuántico definido en biclase().
    Devuelve el promedio de pérdida de validación y los mejores parámetros encontrados.
    """

    fold_losses = []
    best_params = None
    best_loss = float("inf")
    
    if partitions is not None:
        folds = partitions
        k_folds = len(folds)
    elif k_folds == 1:
        # Hold-out
        train_idx, val_idx = train_test_split(
            np.arange(len(data)), test_size=holdout_frac, shuffle=True, random_state=42
        )
        folds = [(train_idx, val_idx)]
    else:
        # K-Fold normal
        if k_folds >= len(data):
            warnings.warn(
                f"k_folds={k_folds} es mayor que el número de muestras={len(data)}. "
                "Se ajustará automáticamente a LOOCV.",
                UserWarning
            )
            k_folds = len(data)
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        folds = list(kf.split(data))
        
    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"\n🌀 Fold {fold + 1}/{k_folds}")

        # División de datos
        X_train, X_val = data[train_idx], data[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Entrenamiento en este fold
        predictions_by_epoch, params = biclase_param(
            train=X_train,
            test=X_val,
            labels=y_train,
            codigo=codigo,
            noise=noise,
            backend=backend,
            noise_model=noise_model,
            result=result,
            shots=shots,
            parametros_iniciales=parametros_iniciales,
            optimize=optimize,
            epochs=epochs,
            lr=lr,
            rng=rng
        )

        # Predicciones finales del último epoch
        preds = predictions_by_epoch[-1]

        # Evaluación en validación
        val_loss = cross_entropy(y_val, preds)
        fold_losses.append(val_loss)
        print(f"📉 Pérdida de validación Fold {fold + 1}: {val_loss:.6f}")

        # Guardar si mejora
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = params

    print("\n✅ Validación cruzada completada")
    print(f"Pérdida promedio: {np.mean(fold_losses):.6f}")
    print(f"Mejor pérdida: {best_loss:.6f}")

    return {
        "mean_loss": np.mean(fold_losses),
        "best_loss": best_loss,
        "best_params": best_params,
        "fold_losses": fold_losses
    }

def biclase_predit(X_train, Y_train, x_val, codigo = "gray", noise = 0.0, backend = None, noise_model = None,parametros_iniciales = None):
           
    predictions = []
    for _, X_val in enumerate(x_val):

        prediction, _ = biclase_param(
            train = X_train,
            test = X_val,
            labels = Y_train,
            codigo = codigo,
            noise = noise,
            backend = backend,
            noise_model = noise_model,
            parametros_iniciales = parametros_iniciales
        )

        predictions.append(prediction)

    return predictions


 