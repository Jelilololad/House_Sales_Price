import json
from pathlib import Path
import pickle
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, RobustScaler

# Import both main_wrangler and data_wrangler
from src.wrangler import data_wrangler, main_wrangler

optuna.logging.set_verbosity(optuna.logging.WARNING)


def test_train(dataset):
    X = dataset.drop(columns=["SalePrice"])
    y_all = dataset["SalePrice"]

    # Log-transform target if skewed
    if y_all.skew() > 1:
        y = np.log1p(y_all)
    else:
        y = y_all

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    return X_train, X_test, y_train, y_test


def save_model(model, metrics: dict):
    model_dir = Path.cwd() / "artifacts"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.pkl"
    metrics_path = model_dir / "metrics.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nModel saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    return model_path, metrics_path


def build_pipeline(params: dict, categorical_cols: list[str], numerical_cols: list[str]):
    # 1. Custom wrangler step using FunctionTransformer
    wrangler_step = FunctionTransformer(data_wrangler)

    # 2. Preprocessor step
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), numerical_cols),
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ),
                categorical_cols,
            ),
        ]
    )

    # 3. Complete pipeline
    return Pipeline(
        [
            ("wrangler", wrangler_step),
            ("preprocessor", preprocessor),
            ("model", LGBMRegressor(**params, random_state=42, verbosity=-1)),
        ]
    )


def objective(trial, X_train, y_train, cat_cols, num_cols):
    params = {
        "objective": "regression",
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 12, 63),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "subsample": trial.suggest_float("subsample", 0.5, 0.95),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.85),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipeline = build_pipeline(params, cat_cols, num_cols)
        pipeline.fit(X_tr, y_tr)

        preds_log = pipeline.predict(X_val)
        fold_rmse = root_mean_squared_error(y_val, preds_log)
        cv_scores.append(fold_rmse)

    mean_log_rmse = float(np.mean(cv_scores))

    print(
        f"Trial {trial.number:03d} | "
        f"5-Fold Mean RMSLE: {mean_log_rmse:.5f}"
    )

    return mean_log_rmse


def main_model():
    dataset = main_wrangler()

    X_train, X_test, y_train, y_test = test_train(dataset)

    # Run data_wrangler once on a dummy copy to dynamically infer column types
    # AFTER string-to-numeric mappings occur inside data_wrangler
    temp_wrangled = data_wrangler(X_train)
    cat_cols = temp_wrangled.select_dtypes(include=["object", "category"]).columns.to_list()
    num_cols = temp_wrangled.select_dtypes(include=["number"]).columns.to_list()

    print("--- Starting 5-Fold Cross-Validated Optuna Tuning ---")

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, cat_cols, num_cols),
        n_trials=100,
    )

    print("\n" + "=" * 50)
    print(f"Best 5-Fold Mean Log RMSE (RMSLE): {study.best_value:.5f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("=" * 50)

    best_params = study.best_params.copy()
    best_params["objective"] = "regression"
    best_params["subsample_freq"] = 1

    best_pipeline = build_pipeline(best_params, cat_cols, num_cols)
    best_pipeline.fit(X_train, y_train)

    y_pred_log = best_pipeline.predict(X_test)
    final_log_rmse = root_mean_squared_error(y_test, y_pred_log)
    final_dollar_rmse = root_mean_squared_error(
        np.expm1(y_test), np.expm1(y_pred_log)
    )

    print(f"\nHeld-out Test Log RMSE (RMSLE): {final_log_rmse:.5f}")
    print(f"Held-out Test Dollar RMSE: ${final_dollar_rmse:,.2f}")

    metrics = {
        "cv_best_log_rmse": float(study.best_value),
        "test_log_rmse": float(final_log_rmse),
        "test_dollar_rmse": float(final_dollar_rmse),
        "hyperparameters": best_params,
    }

    save_model(best_pipeline, metrics)

    return best_pipeline


if __name__ == "__main__":
    main_model()