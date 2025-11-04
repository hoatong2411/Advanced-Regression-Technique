# ames_housing/tuning.py
import optuna
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from .modeling import get_base_models, build_feature_selector
from .preprocessing import get_optimal_preprocessor
from .transformation import wrapper_model_with_target_transform
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import (
    Ridge, Lasso, HuberRegressor, QuantileRegressor, RANSACRegressor, ElasticNet
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

RANDOM_STATE = 42

def tune_model(model_name, X, y, group_name, config, n_trials=50):
    def objective(trial):
        # Tạo model từ trial (giữ logic create_model_from_trial)
        if model_name == "Ridge":
            model = Ridge(alpha=trial.suggest_float("alpha", 1e-3, 1000, log=True), random_state=RANDOM_STATE)
        elif model_name == "Lasso":
            model = Lasso(alpha=trial.suggest_float("alpha", 1e-4, 10, log=True), max_iter=10000, random_state=RANDOM_STATE)
        elif model_name == "ElasticNet":
            model = ElasticNet(
                alpha=trial.suggest_float("alpha", 1e-4, 10, log=True),
                l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
                max_iter=10000, random_state=RANDOM_STATE
            )
        elif model_name == "Huber":
            model = HuberRegressor(
                epsilon=trial.suggest_float("epsilon", 1.1, 2.0),
                alpha=trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
            )
        elif model_name == "Quantile":
            model = QuantileRegressor(
                alpha=trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                quantile=trial.suggest_float("quantile", 0.1, 0.9)
            )
        elif model_name == "RANSAC":
            est_type = trial.suggest_categorical("estimator", ["Ridge", "Lasso"])
            if est_type == "Ridge":
                est = Ridge(alpha=trial.suggest_float("alpha_ridge", 1e-3, 1000, log=True))
            else:
                est = Lasso(alpha=trial.suggest_float("alpha_lasso", 1e-4, 10, log=True))
            model = RANSACRegressor(
                estimator=est,
                min_samples=trial.suggest_float("min_samples", 0.3, 0.9),
                max_trials=trial.suggest_int("max_trials", 50, 500),
                random_state=RANDOM_STATE
            )
        elif model_name == "GB":
            model = GradientBoostingRegressor(
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_int("max_depth", 2, 8),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                random_state=RANDOM_STATE
            )
        elif model_name == "RF":
            model = RandomForestRegressor(
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_int("max_depth", 3, 15),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        elif model_name == "ADB":
            model = AdaBoostRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 500),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 1.0),
                random_state=RANDOM_STATE
            )
        elif model_name == "XGB":
            model = XGBRegressor(
                n_estimators=trial.suggest_int("n_estimators", 200, 1000),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist"
            )
        elif model_name == "LGBM":
            model = LGBMRegressor(
                n_estimators=trial.suggest_int("n_estimators", 200, 1000),
                num_leaves=trial.suggest_int("num_leaves", 20, 100),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1
            )
        elif model_name == "CatBoost":
            model = CatBoostRegressor(
                depth=trial.suggest_int("depth", 4, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                iterations=trial.suggest_int("iterations", 300, 1000),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 10),
                random_state=RANDOM_STATE,
                verbose=False
            )
        else:
            raise ValueError(f"Model {model_name} chưa được hỗ trợ trong tuning.")

        # Lấy preprocessor từ get_optimal_preprocessor (thay thế xây thủ công)
        preprocessor, tgt_tf, (selector_name, selector) = get_optimal_preprocessor(
            X_train=X, y_train=y, group=group_name, config=config
        )

        # Áp target transform
        model_t = wrapper_model_with_target_transform(model, method=tgt_tf)

        # Xây pipeline
        pipe_steps = [("preprocessor", preprocessor)]
        if selector_name != "passthrough":
            pipe_steps.append(("selector", selector))
        pipe_steps.append(("model", model_t))
        pipeline = Pipeline(pipe_steps)

        # CV với cross_val_score (tối ưu hơn loop thủ công)
        scores = cross_val_score(
            pipeline, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        return -scores.mean()
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study.best_value

def tune_all_models(X, y, model_groups, config, n_trials=50):
    tuning_results = []
    for group_name, models_in_group in model_groups.items():
        print(f"\n{'='*70}")
        print(f" TUNING GROUP: {group_name}")
        print(f"{'='*70}")

        for model_name in models_in_group:
            print(f"\n→ Tuning {model_name}...")

            best_params, best_rmse = tune_model(
                model_name, X, y, group_name, config, n_trials
            )
            tuning_results.append({
                "Group": group_name,
                "Model": model_name,
                "Best_RMSE": best_rmse,
                "Best_Params": best_params
            })
    return pd.DataFrame(tuning_results)