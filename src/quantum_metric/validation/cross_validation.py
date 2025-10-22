import numpy as np
import warnings
from sklearn.model_selection import KFold, train_test_split

def cross_validate_biclase(model, data, labels, k_folds=5, test_size=0.2, partitions=None):
    """
    Ejecuta validación cruzada, LOOCV o hold-out para modelos cuánticos.
    """

    X_train1, X_val, y_train1, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

    if partitions is not None:
        folds = partitions
    elif k_folds == 1:
        X_train, X_test, y_train, y_test = train_test_split(X_train1, y_train1, test_size=test_size, random_state=42)
        folds = [(X_train, y_train, X_test, y_test)]
    else:
        if k_folds > len(data):
            warnings.warn(f"k_folds={k_folds} > muestras={len(data)}; se ajusta a LOOCV.", UserWarning)
            k_folds = len(data)
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        folds = [(data[train], labels[train], data[test], labels[test]) for train, test in kf.split(data)]

    for i, (X_train, y_train, X_test, y_test) in enumerate(folds):
        print(f"\n🌀 Fold {i+1}/{len(folds)}")
        model.fit(X_train, y_train, X_test, y_test)

    preds = model.predict(X_val,X_train1,y_train1)

    print(f"\n✅ Validación completada")
    return {"preds": preds, "best_params": model.best_params_, "y_val": y_val}
