import os
import sys
import dill
from src.exception import custom_exception
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise custom_exception(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise custom_exception(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models: dict, hyperparams: dict):
    try:
        report = {}

        for name in models:
            model = models[name]
            param_grid = hyperparams.get(name, {})

            # Skip if no hyperparameters provided
            if not param_grid:
                raise ValueError(f"No hyperparameters provided for model: {name}")

            gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[name] = {
                "r2_train": train_model_score,
                "r2_test": test_model_score,
                "best_params": gs.best_params_
            }

        return report

    except Exception as e:
        raise custom_exception(e, sys)
