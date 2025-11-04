import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler,
    OneHotEncoder, FunctionTransformer, PolynomialFeatures
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, HuberRegressor, QuantileRegressor, RANSACRegressor, ElasticNet,
    LassoCV, RidgeCV
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, AdaBoostRegressor
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression
import optuna
from .preprocessing import get_optimal_preprocessor, build_feature_selector
from .transformation import (
    apply_target_transformation, inverse_target_transformation, SafeYeoJohnson, safe_clip, BoxCoxTransformer, wrapper_model_with_target_transform, iqr_cap, winsorize_normalize
)
from .utils import get_param, save_ensemble_complete, load_config

RANDOM_STATE = 42


def get_base_models():
    return {
        'Linear': LinearRegression(),
        'Ridge': Ridge(random_state=RANDOM_STATE),
        'Lasso': Lasso(random_state=RANDOM_STATE),
        'Huber': HuberRegressor(),
        'Quantile': QuantileRegressor(),
        'RANSAC' : RANSACRegressor(estimator=Ridge(), random_state=RANDOM_STATE, min_samples=0.5),
        'ElasticNet' : ElasticNet(),
        'XGB': XGBRegressor(random_state=RANDOM_STATE),
        'RF': RandomForestRegressor(random_state=RANDOM_STATE),
        'ADB': AdaBoostRegressor(random_state=RANDOM_STATE),
        'GB': GradientBoostingRegressor(random_state=RANDOM_STATE),
        'LGBM': LGBMRegressor(random_state=RANDOM_STATE, verbosity=-1),
        'CatBoost': CatBoostRegressor(loss_function='RMSE', verbose=False, random_state=RANDOM_STATE)
    }

# ======= FEATURE SELECTION (EDA) =======

def hybrid_feature_selection(X, y, top_k=30, methods=None):
    """
    Kết hợp 5 phương pháp FS → chọn features xuất hiện trong ≥3/5
    """
    if methods is None:
        methods = ['lgbm', 'xgb', 'rf', 'lasso', 'univariate']

    # === Preprocessing ===
    num_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", MinMaxScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("target", TargetEncoder(smoothing=10))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ], sparse_threshold=0)

    X_proc = preprocessor.fit_transform(X, y)
    # Lấy tên cột sau khi xử lý
    feature_names = (
        num_cols +
        [f"{col}_target_encoded" for col in cat_cols]
    )

    n_features = X_proc.shape[1]
    feature_count = {name: 0 for name in feature_names}
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Lấy top k features từ importance
    def get_top_k(imp, k):
        idx = np.argsort(imp)[-k:]
        return [feature_names[i] for i in idx]
    
    # === 1. Tree-based: LGBM, XGB, RF ===
    tree_models = {
        'lgbm': LGBMRegressor(random_state=RANDOM_STATE, verbose=-1),
        'xgb': XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
        'rf': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    }

    importance = {name: np.zeros(n_features) for name in tree_models}

    for train_idx, _ in kf.split(X_proc):
        X_train, y_train = X_proc[train_idx], y.iloc[train_idx]
        for name, model in tree_models.items():
            model.fit(X_train, y_train)
            importance[name] += model.feature_importances_
    for name in importance:
        importance[name] /= 5

    # === 2. LassoCV ===
    if 'lasso' in methods:
        lasso = LassoCV(cv=5, random_state=RANDOM_STATE, n_jobs=-1, max_iter=1000)
        lasso.fit(X_proc, y)
        top_lasso = [n for i, n in enumerate(feature_names) if abs(lasso.coef_[i]) > 1e-5]
    else:
        top_lasso = []

    # === 3. Univariate ===
    if 'univariate' in methods:
        k = min(top_k, n_features)
        selector = SelectKBest(f_regression, k=k).fit(X_proc, y)
        top_uni = [feature_names[i] for i in selector.get_support(indices=True)]
    else:
        top_uni = []

    # === Gộp kết quả ===
    top_lists = [
        get_top_k(importance['lgbm'], top_k) if 'lgbm' in methods else [],
        get_top_k(importance['xgb'], top_k) if 'xgb' in methods else [],
        get_top_k(importance['rf'], top_k) if 'rf' in methods else [],
        top_lasso,
        top_uni
    ]

    for feat_list in top_lists:
        for f in feat_list:
            if f in feature_count:
                feature_count[f] += 1

    # === Lọc ≥3 lần ===
    threshold = 3
    stable_features = [f for f, c in feature_count.items() if c >= threshold]
    stable_features = sorted(stable_features)

    print(f"\nStable features (≥{threshold}/{len(methods)} methods): {len(stable_features)}")
    print(f"Top 10: {stable_features[:10]}")

    return stable_features, feature_count


# ======= BASELINE EVALUTAION (PREPROCESSING) ==========
def baseline_evaluate_dataset(df, dataset_name, base_models, n_splits=5):
    print(f"\n===============================")
    print(f"📊 Đang đánh giá dataset: {dataset_name}")
    print(f"===============================")

    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"].values

    num_cols = [col for col in X.columns if X[col].dtype in ["float64", "int64"]]
    cat_cols = [col for col in X.columns if X[col].dtype not in ["float64", "int64"]]

    # Bộ tiền xử lý
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])

    cat_pipeline = Pipeline([
        ('encoder', TargetEncoder()),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    # Thiết lập KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    results = []

    # Vòng lặp qua từng model
    for name, model in base_models.items():
        print(f"\n🔹 Mô hình: {name}")
        fold_rmse, fold_r2 = [], []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)

            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)

            fold_rmse.append(rmse)
            fold_r2.append(r2)

            print(f"   Fold {fold} → RMSE={rmse:.2f}, R²={r2:.4f}")

        results.append({
            'Dataset': dataset_name,
            'Model': name,
            'Mean_RMSE': np.mean(fold_rmse),
            'Std_RMSE': np.std(fold_rmse),
            'Mean_R2': np.mean(fold_r2),
            'Std_R2': np.std(fold_r2)
        })

    return pd.DataFrame(results)


def build_pipeline_for_model(model_name, base_model, group_name, config, X_train, y_train):
    """
    Xây dựng full pipeline cho một mô hình cụ thể theo group và config.
    """
    preprocessor, tgt_tf, (selector_name, selector) = get_optimal_preprocessor(
        X_train=X_train, y_train=y_train, group=group_name, config=config
    )

    # Xây pipeline
    steps = [("preprocessor", preprocessor)]
    if selector_name != "passthrough" and selector is not None:
        steps.append(("selector", selector))
    steps.append(("model", base_model))

    return Pipeline(steps), tgt_tf

# ========== FUNCTION FOR OPTIMIZING PREPROCESSING ==============

# HÀM CHO PHÉP THỬ NGHIỆM CÁC PREPROCESSOR KHÁC NHAU
def evaluate_dataset(
    df,
    dataset_name,
    base_models,           
    model_groups,          
    group_name=None,       
    config = None,
    n_splits=5,
    imputer_type="simple",
    scaler_type="minmax",
    encoder_type="onehot",
    outlier_method=None,
    input_transform="org",
    target_transform="org",
    multicol_method=None,
    RANDOM_STATE=RANDOM_STATE,
    knn_neighbors=5,
    target_encoder_smoothing=10
):
    if config is None:
        config = load_config()
    
    print(f"\n{'='*60}")
    print(f"GROUP: {group_name or 'All'} | DATASET: {dataset_name}")
    print(f"{'='*60}")

    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"].values
    num_cols = [c for c in X.columns if X[c].dtype in ["float64", "int64"]]
    cat_cols = [c for c in X.columns if X[c].dtype not in ["float64", "int64"]]

    results = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    # Lấy danh sách model cần chạy (theo group nếu có, nếu không thì tất cả)
    if group_name:
        model_names = model_groups.get(group_name, [])
    else:
        model_names = [name for group in model_groups.values() for name in group]

    # Loại bỏ trùng lặp và giữ thứ tự
    model_names = list(dict.fromkeys(model_names))

    for name in model_names:
        if name not in base_models:
            print(f"   [SKIP] Model {name} not in base_models")
            continue

        model = base_models[name]
        print(f"\nModel: {name}")

        imp_type = get_param(config, group_name or "All", "imputer_type", imputer_type)
        scl_type = get_param(config, group_name or "All", "scaler_type", scaler_type)
        enc_type = get_param(config, group_name or "All", "encoder_type", encoder_type)
        outlier = get_param(config, group_name or "All", "outlier_method", outlier_method)
        in_tf = get_param(config, group_name or "All", "input_transform", input_transform)
        tgt_tf = get_param(config, group_name or "All", "target_transform", target_transform)
        multicol = get_param(config, group_name or "All", "multicollinearity", multicol_method)
        poly_enabled = get_param(config, group_name, "poly_features", False)

        if knn_neighbors:
          knn_k = knn_neighbors
        else:
          knn_k = get_param(config, group_name, "knn_neighbors", 5)
        # === IMPUTER ===
        if imp_type == "knn":
            imputer = KNNImputer(n_neighbors=knn_k)
        elif imp_type == "iterative":
            imputer = IterativeImputer(random_state=RANDOM_STATE)
        elif imp_type == "simple":
            imputer = SimpleImputer(strategy="mean")
        else:  # simple_median
            imputer = SimpleImputer(strategy="median")

        # === SCALER ===
        if scl_type == "standard":
            scaler = StandardScaler()
        elif scl_type == "robust":
            scaler = RobustScaler()
        else:
            scaler = MinMaxScaler()

        if target_encoder_smoothing:
          tg_smoothing = target_encoder_smoothing
        else:
          tg_smoothing = get_param(config, group_name, "target_encoder_smoothing", 10)
        # === ENCODER ===
        if enc_type == "target":
            encoder = TargetEncoder(cols=cat_cols, smoothing=tg_smoothing)
        else:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        # === Numeric Pipeline ===
        num_steps = [("imputer", imputer)]

        if outlier == "winsor":
            num_steps.append(("winsor", FunctionTransformer(winsorize_normalize, validate=False)))
        elif outlier == "iqr":
            num_steps.append(("iqr", FunctionTransformer(iqr_cap, validate=False)))

        if in_tf == "log":
            num_steps.append(("log", FunctionTransformer(np.log1p, validate=False)))
        elif in_tf == "sqrt":
            num_steps.append(("sqrt", FunctionTransformer(np.sqrt, validate=False)))
        elif in_tf == "yeo-johnson":
            num_steps.append(("yeo", SafeYeoJohnson()))
        elif in_tf == "box-cox":
            num_steps.append(("box-cox", BoxCoxTransformer()))

        num_steps.append(("clip", FunctionTransformer(safe_clip, validate=False)))
        num_steps.append(("scaler", scaler))
        if poly_enabled:
            from sklearn.preprocessing import PolynomialFeatures
            num_steps.insert(-2, ("poly", PolynomialFeatures(degree=2, include_bias=False)))
        num_pipeline = Pipeline(num_steps)

        # === Categorical Pipeline ===
        cat_pipeline = Pipeline([
            ("encoder", encoder),
            ("scaler", StandardScaler())
        ])

        preprocessor = ColumnTransformer([
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ])

        # === Cross-validation ===
        fold_rmse, fold_r2, fold_time = [], [], []
        model_t = wrapper_model_with_target_transform(model, method=tgt_tf)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Fit preprocessor
            X_train_pre = preprocessor.fit_transform(X_train, y_train)
            X_val_pre = preprocessor.transform(X_val)

            # Xử lý multicollinearity
            selector_name, selector = build_feature_selector(
                method=multicol,
                X_train=X_train_pre,
                y_train=y_train
            )

            pipe_steps = [("preprocessor", preprocessor)]
            if selector_name != "passthrough":
                pipe_steps.append(("feature_selection", selector))
            pipe_steps.append(("model", model_t))
            pipe = Pipeline(pipe_steps)

            start = time.time()
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            elapsed = time.time() - start

            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)

            fold_rmse.append(rmse)
            fold_r2.append(r2)
            fold_time.append(elapsed)

            print(f"   Fold {fold} → RMSE={rmse:.2f}, R²={r2:.4f}, Time={elapsed:.2f}s")

        # Xác định nhóm model
        model_group = "All"
        if group_name:
            model_group = group_name
        else:
            for g, models in model_groups.items():
                if name in models:
                    model_group = g
                    break

        results.append({
            "Group": model_group,
            "Dataset": dataset_name,
            "Model": name,
            "Imputer": imp_type,
            "Scaler": scl_type,
            "Encoder": enc_type,
            "Outlier": outlier,
            "Input_Transform": in_tf,
            "Target_Transform": tgt_tf,
            "Multicollinearity": multicol,
            "Mean_RMSE": np.mean(fold_rmse),
            "Std_RMSE": np.std(fold_rmse),
            "Mean_R2": np.mean(fold_r2),
            "Std_R2": np.std(fold_r2),
            "Mean_Time": np.mean(fold_time)
        })

    print(f"\nHOÀN TẤT {group_name or 'All'}")
    return pd.DataFrame(results)

# HÀM TÍNH ĐIỂM CỦA GIẢI PHÁP SO VỚI BASELINE
def compare_experiment_results(base_df, test_dfs, test_names,
                               group_col="Group", weights=(0.5, 0.2, 0.3)):
    """
    So sánh hiệu năng giữa baseline và các bộ test theo nhóm model.
    """
    if isinstance(test_dfs, pd.DataFrame):
        test_dfs = [test_dfs]
    if isinstance(test_names, str):
        test_names = [test_names]

    w_rmse, w_std, w_r2 = weights
    summary_rows = []

    print("\n===============================")
    print("BẮT ĐẦU SO SÁNH HIỆU NĂNG")
    print("===============================")

    groups = sorted(base_df[group_col].unique())

    for group in groups:
        print(f"\nNHÓM: {group}")
        base_group = base_df[base_df[group_col] == group]
        group_scores = []

        for name, test_df in zip(test_names, test_dfs):
            test_group = test_df[test_df[group_col] == group]
            if test_group.empty:
                continue

            common_models = sorted(set(base_group["Model"]) & set(test_group["Model"]))
            if not common_models:
                continue

            model_scores = []
            for model in common_models:
                b = base_group[base_group["Model"] == model].iloc[0]
                t = test_group[test_group["Model"] == model].iloc[0]

                improve_rmse = 1 - (t["Mean_RMSE"] / b["Mean_RMSE"])
                improve_std = 1 - (t["Std_RMSE"] / b["Std_RMSE"]) if b["Std_RMSE"] > 0 else 0
                improve_r2 = (t["Mean_R2"] / b["Mean_R2"]) - 1 if b["Mean_R2"] > 0 else 0

                score = w_rmse * improve_rmse + w_std * improve_std + w_r2 * improve_r2
                model_scores.append({"Model": model, "Score": score, "Improve_RMSE": improve_rmse})

            if model_scores:
                df_scores = pd.DataFrame(model_scores)
                mean_score = df_scores["Score"].mean()
                mean_improve = df_scores["Improve_RMSE"].mean()
                group_scores.append({
                    "Experiment": name,
                    "Group": group,
                    "Mean_Score": mean_score,
                    "Mean_Improve_RMSE": mean_improve,
                    "Num_Models": len(common_models)
                })
                print(f"  {name}: Score={mean_score:.4f}, RMSE improve={mean_improve:.1%} ({len(common_models)} models)")

        if group_scores:
            group_df = pd.DataFrame(group_scores)
            best_in_group = group_df.sort_values("Mean_Score", ascending=False).iloc[0]
            summary_rows.append(best_in_group)
            print(f"  BEST: {best_in_group['Experiment']} (Score: {best_in_group['Mean_Score']:.4f})")

    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()
    if not summary_df.empty:
        print("\nTỔNG HỢP:")
        print(summary_df[["Group", "Experiment", "Mean_Score", "Mean_Improve_RMSE"]].to_string(index=False))

    return summary_df

# ====== Ensemble ============

def get_three_model_sets(oof_results, cv_scores):
    """
    Trả về 3 bộ model để ensemble:
    - all_models: Tất cả
    - top4_global: Top 4 theo Mean RMSE toàn cục
    - top1_per_group: Top 1 mỗi group
    """
    # 1. Tạo DataFrame tổng hợp CV scores
    records = []
    for group in cv_scores:
        for model, score in cv_scores[group].items():
            records.append({"Group": group, "Model": model, "Mean_RMSE": score})
    final_results = pd.DataFrame(records).sort_values("Mean_RMSE").reset_index(drop=True)

    # 2. all_models
    all_models = [(g, m) for g in oof_results for m in oof_results[g]]
    print(f"   → All models: {len(all_models)} models")

    # 3. top4_global (lọc hợp lệ)
    top4_raw = final_results.nsmallest(4, "Mean_RMSE")[["Group", "Model"]].values.tolist()
    top4_global = [(g, m) for g, m in top4_raw if g in oof_results and m in oof_results[g]]
    print(f"   → Top 4 Global: {top4_global}")

    # 4. top1_per_group
    top1_per_group = []
    for group in cv_scores:
        if not cv_scores[group]:
            continue
        best_model = min(cv_scores[group], key=cv_scores[group].get)
        top1_per_group.append((group, best_model))
    print(f"   → Top 1 per Group: {top1_per_group} ({len(top1_per_group)} groups)")

    return {
        "all_models": all_models,
        "top4_global": top4_global,
        "top1_per_group": top1_per_group,
        "final_results_df": final_results
    }

def create_results_df_from_cv(cv_scores):
    rows = []
    for group, models in cv_scores.items():
        for model, rmse in models.items():
            rows.append({"Group": group, "Model": model, "Mean_RMSE": rmse, "Std_RMSE": 0, "Mean_R2": 0})
    return pd.DataFrame(rows)

def run_per_group_with_oof(df, base_models, model_groups, config, n_splits=5, RANDOM_STATE=RANDOM_STATE):
    X = df.drop(columns=["SalePrice"]).reset_index(drop=True)
    y = df["SalePrice"].values
    n_samples = len(X)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    oof_results = {}
    cv_scores = {}
    trained_pipelines = {}
    target_methods = {}

    for group in model_groups.keys():
        print(f"\n→ Running group: {group}")
        model_names = model_groups[group]
        oof_results[group] = {}
        cv_scores[group] = {}
        trained_pipelines[group] = {}
        target_methods[group] = {}

        tgt_tf = get_param(config, group, "target_transform", "org")

        for name in model_names:
            if name not in base_models:
                continue
            model = base_models[name]

            oof_pred_full = np.zeros(n_samples)
            fold_rmse = []
            fold_pipelines = []

            print(f"   Training {name} (target: {tgt_tf})...")

            for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train_raw, y_val_raw = y[train_idx], y[val_idx]

                # BIẾN ĐỔI TARGET
                y_train = apply_target_transformation(y_train_raw, tgt_tf)

                pipe, _ = build_pipeline_for_model(
                    model_name=name, base_model=model, group_name=group,
                    config=config, X_train=X_train, y_train=y_train
                )

                pipe.fit(X_train, y_train)
                y_pred_t = pipe.predict(X_val)
                # NGƯỢC LẠI TARGET
                y_pred = inverse_target_transformation(y_pred_t, tgt_tf)

                oof_pred_full[val_idx] = y_pred
                rmse = np.sqrt(mean_squared_error(y_val_raw, y_pred))
                fold_rmse.append(rmse)
                fold_pipelines.append(pipe)

                print(f"      Fold {fold} → RMSE={rmse:.2f}")

            oof_results[group][name] = oof_pred_full
            cv_scores[group][name] = np.mean(fold_rmse)
            trained_pipelines[group][name] = fold_pipelines
            target_methods[group][name] = tgt_tf

            print(f"   {name}: Mean RMSE = {cv_scores[group][name]:.2f}")

    return oof_results, cv_scores, trained_pipelines, target_methods

# ====== Weighted Averaging ========

# HÀM WARMUP TUNING WEIGHTED AVERAGING VỚI OPTUNA
def tune_weighted_averaging_with_optuna(oof_results, y_true, n_trials=100, timeout=300):
    """
    Tìm weight toàn cục (warm-up)
    """
    print(f"\nWARMUP TUNING WEIGHTED AVERAGING VỚI OPTUNA (n_trials={n_trials}, timeout={timeout}s)...")

    # Lấy tất cả dự đoán OOF từ các model
    model_names = []
    all_preds = []
    for group in oof_results:
        for name in oof_results[group]:
            model_names.append(f"{group}_{name}")
            all_preds.append(oof_results[group][name])
    X = np.column_stack(all_preds)  # shape: (n_samples, n_models)

    def objective(trial):
        # Gợi ý trọng số trong [0, 1], tổng = 1
        weights = []
        for i in range(len(model_names)):
            w = trial.suggest_float(f"w_{i}", 0.0, 1.0)
            weights.append(w)
        weights = np.array(weights)
        if weights.sum() == 0:
            weights += 1e-6
        weights = weights / weights.sum()

        # Dự đoán ensemble
        pred = X @ weights
        rmse = np.sqrt(mean_squared_error(y_true, pred))
        return rmse

    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    optuna.logging.disable_default_handler()
    # Tạo study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    # Lấy kết quả tốt nhất
    best_weights = study.best_params
    norm_weights = np.array([best_weights[f"w_{i}"] for i in range(len(model_names))])
    norm_weights = norm_weights / norm_weights.sum()

    final_pred = X @ norm_weights
    final_rmse = np.sqrt(mean_squared_error(y_true, final_pred))

    print(f"\nBEST RMSE: {final_rmse:.2f}")
    print("BEST WEIGHTS:")
    for name, w in zip(model_names, norm_weights):
        print(f"   {name}: {w:.4f}")

    return {
        "Optuna_Weighted": final_pred
    }, dict(zip(model_names, norm_weights.round(6))), study

# 2. HILL CLIMBING using optimal weight from OPTUNA WARM-START
def hill_climbing(oof_results, y_true, optuna_weights, model_set, step_size=0.001, max_steps=200, min_improve=0.001):
    """
    Tìm weight tối ưu cục bộ dựa trên warm-up weight 
    """
    # Lấy predictions đúng thứ tự theo model_set
    all_preds = []
    model_keys = []
    for group, model in model_set:
        key = f"{group}_{model}"
        all_preds.append(oof_results[group][model])
        model_keys.append(key)
    X = np.column_stack(all_preds)

    # Lấy weights theo đúng thứ tự
    weights = np.array([optuna_weights[key] for key in model_keys])
    best_pred = X @ weights
    best_rmse = np.sqrt(mean_squared_error(y_true, best_pred))
    current_weights = weights.copy()

    improved = False
    for step in range(max_steps):
        made_better = False
        for i in range(len(weights)):
            for delta in [-step_size, step_size]:
                new_weights = current_weights.copy()
                new_weights[i] += delta
                if new_weights[i] < 0:
                    continue
                new_weights = new_weights / new_weights.sum()
                pred = X @ new_weights
                rmse = np.sqrt(mean_squared_error(y_true, pred))
                if rmse < best_rmse - min_improve:
                    best_rmse = rmse
                    current_weights = new_weights
                    best_pred = pred
                    made_better = True
                    improved = True
        if not made_better:
            break

    if not improved:
        print("   Hill Climbing: Không cải thiện → giữ Optuna weights")
        final_weights = optuna_weights
    else:
        final_weights = dict(zip(model_keys, current_weights.round(6)))

    return {"HillClimbing_OptunaWarm": best_pred}, final_weights


# 3. TUNING HILL CLIMBING
def tune_hill_climbing_with_optuna(oof_results, y_true, optuna_weights, n_trials=50):
    print("\nTUNING HILL CLIMBING VỚI OPTUNA...")

    def objective(trial):
        step_size = trial.suggest_float("step_size", 1e-5, 0.01, log=True)
        min_improve = trial.suggest_float("min_improve", 1e-5, 0.1, log=True)

        pred, _ = hill_climbing(
            oof_results=oof_results,
            y_true=y_true,
            optuna_weights=optuna_weights,
            step_size=step_size,
            max_steps=200,
            min_improve=min_improve
        )
        return np.sqrt(mean_squared_error(y_true, pred["HillClimbing_OptunaWarm"]))

    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    optuna.logging.disable_default_handler()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_rmse = study.best_value

    # Chạy lại với best params
    final_pred_dict, final_weights = hill_climbing(
        oof_results=oof_results,
        y_true=y_true,
        optuna_weights=optuna_weights,
        step_size=best_params["step_size"],
        max_steps=200,
        min_improve=best_params["min_improve"]
    )

    final_pred = final_pred_dict["HillClimbing_OptunaWarm"]
    final_rmse = np.sqrt(mean_squared_error(y_true, final_pred))

    print(f"   BEST HILL RMSE: {final_rmse:.5f} (vs Optuna: {np.sqrt(mean_squared_error(y_true, X @ np.array(list(optuna_weights.values())))):.5f})")

    return final_pred_dict, {**best_params, "final_weights": final_weights}


# 4. HÀM FINAL CHẠY WEIGHTED AVERAGING CHO 1 MODEL SET
def run_ensemble_for_set(oof_results, y_true, model_set, set_name, n_trials_optuna=300, n_trials_hill=50):
    print(f"\n{'-'*50}")
    print(f"ENSEMBLE: {set_name.upper()} ({len(model_set)} models)")
    print(f"{'-'*50}")

    # Lọc OOF preds theo set
    set_preds = []
    set_names = []
    for g, m in model_set:
        set_preds.append(oof_results[g][m])
        set_names.append(f"{g}_{m}")
    X_set = np.column_stack(set_preds)

    # Optuna
    def objective(trial):
        weights = [trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(len(set_names))]
        weights = np.array(weights)
        if weights.sum() == 0: weights += 1e-6
        weights /= weights.sum()
        pred = X_set @ weights
        return np.sqrt(mean_squared_error(y_true, pred))

    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    optuna.logging.disable_default_handler()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials_optuna, timeout=600)
    best_w = np.array([study.best_params[f"w_{i}"] for i in range(len(set_names))])
    best_w = best_w / best_w.sum()
    opt_pred = X_set @ best_w
    opt_rmse = np.sqrt(mean_squared_error(y_true, opt_pred))

    opt_weights = dict(zip(set_names, best_w.round(6)))

    # Hill Climbing
    hill_pred_dict, hill_weights = hill_climbing(
        oof_results=oof_results,
        y_true=y_true,
        optuna_weights=opt_weights,
        model_set=model_set,
        step_size=0.001,
        max_steps=100,
        min_improve=0.001
    )
    hill_pred = hill_pred_dict["HillClimbing_OptunaWarm"]
    hill_rmse = np.sqrt(mean_squared_error(y_true, hill_pred))

    if hill_rmse < opt_rmse:
        print(f"   Hill Climbing: CẢI THIỆN → {opt_rmse - hill_rmse:+.5f}")
        final_pred = hill_pred
        final_rmse = hill_rmse
        final_weights = hill_weights
    else:
        print(f"   Hill Climbing: KHÔNG CẢI THIỆN → giữ Optuna")
        final_pred = opt_pred
        final_rmse = opt_rmse
        final_weights = opt_weights

    return {
        "opt_pred": opt_pred,
        "opt_weights": opt_weights,
        "opt_rmse": opt_rmse,
        "hill_pred": hill_pred,
        "hill_weights": hill_weights,
        "hill_rmse": hill_rmse,
        "study": study,
        "model_names": set_names
    }

# ==============================
# FULL PIPELINE
# ==============================
def run_optuna_weighted_ensemble(df, dataset_name, base_models, model_groups, config,
                                 n_trials_optuna=500, n_trials_hill=50):
    # 1. OOF
    oof_results, cv_scores, trained_pipelines, _ = run_per_group_with_oof(
        df=df, base_models=base_models, model_groups=model_groups, config=config, n_splits=5
    )
    y_true = df["SalePrice"].values

    # 2. 3 model sets
    model_sets = get_three_model_sets(oof_results, cv_scores)
    sets_info = [
        (model_sets["all_models"], "all_models"),
        (model_sets["top4_global"], "top4_global"),
        (model_sets["top1_per_group"], "top1_per_group")
    ]

    # 3. Ensemble
    ensemble_results = {}
    weight_figs = {}
    idx = 1
    for model_set, set_name in sets_info:
        if not model_set:
            continue
        result = run_ensemble_for_set(
            oof_results=oof_results, y_true=y_true, model_set=model_set, set_name=set_name,
            n_trials_optuna=n_trials_optuna, n_trials_hill=n_trials_hill
        )
        ensemble_results[set_name] = result

        # VẼ + LƯU TRỌNG SỐ
        fig = plt.figure(figsize=(10, max(5, len(result["opt_weights"])*0.35)))
        names = list(result["opt_weights"].keys())
        weights = list(result["opt_weights"].values())
        plt.barh(names, weights, color='lightcoral' if 'top' in set_name else 'skyblue', edgecolor='black')
        plt.title(f"{set_name.replace('_', ' ').title()} - Optuna Weights")
        plt.xlabel("Weight")
        for i, (n, w) in enumerate(zip(names, weights)):
            plt.text(w + 0.001, i, f"{w:.4f}", va='center', fontsize=8)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        weight_figs[set_name] = (fig, f"{idx:02d}_weights_{set_name}.png")
        idx += 1

    # 4. So sánh
    comparison = []
    for name, res in ensemble_results.items():
        comparison.append({
            "Set": name, "N_Models": len(res["model_names"]),
            "Optuna_RMSE": res["opt_rmse"], "Hill_RMSE": res["hill_rmse"],
            "Improvement": res["opt_rmse"] - res["hill_rmse"]
        })
    comparison_df = pd.DataFrame(comparison).round(5)

    best_set = comparison_df.loc[comparison_df["Hill_RMSE"].idxmin(), "Set"]

    # === RMSE BAR + HEATMAP ===
    baseline_df = create_results_df_from_cv(cv_scores)
    best_single = baseline_df.nsmallest(1, "Mean_RMSE").iloc[0]

    # RMSE Bar
    fig_rmse = plt.figure(figsize=(10, 6))
    sets = ["Best Single"] + list(ensemble_results.keys())
    rmses = [best_single["Mean_RMSE"]] + [res["hill_rmse"] for res in ensemble_results.values()]
    colors = ["#95a5a6"] + ["#3498db", "#e74c3c", "#2ecc71"]
    bars = plt.bar(sets, rmses, color=colors, edgecolor="black", alpha=0.8)
    plt.ylabel("RMSE"); plt.title("RMSE Comparison: Best Single vs Ensemble Sets")
    plt.grid(axis="y", alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 100, f"{h:,.0f}", ha="center", fontsize=10)
    plt.tight_layout()

    # Heatmap
    imp_matrix = []
    for set_name, res in ensemble_results.items():
        improve = (best_single["Mean_RMSE"] - res["hill_rmse"]) / best_single["Mean_RMSE"] * 100
        imp_matrix.append([set_name.replace("_", " ").title(), improve])
    imp_df = pd.DataFrame(imp_matrix, columns=["Set", "Improvement_%"]).set_index("Set")
    fig_heatmap = plt.figure(figsize=(8, 4))
    sns.heatmap(imp_df.T, annot=True, fmt=".2f", cmap="RdYlGn", center=0, cbar_kws={'label': 'Improvement (%)'})
    plt.title("RMSE Improvement vs Best Single Model")
    plt.tight_layout()

    # 5. Lưu
    full_results = {
        "oof_results": oof_results, "cv_scores": cv_scores, "trained_pipelines": trained_pipelines,
        "model_sets": model_sets, "ensemble_results": ensemble_results,
        "comparison_df": comparison_df, "best_set": best_set,
        "figs": {"weight_figs": weight_figs, "rmse": fig_rmse, "heatmap": fig_heatmap}
    }
    save_folder = save_ensemble_complete(full_results, y_true, dataset_name, method="weighted")
    full_results["save_folder"] = save_folder

    return full_results, save_folder

# ====== STACKING ===============
def run_stacking_for_set(oof_results, y_true, model_set, set_name, n_trials=50):
    print(f"\n{'-'*60}")
    print(f"STACKING: {set_name.upper()} ({len(model_set)} base models)")
    print(f"{'-'*60}")

    # Lọc hợp lệ
    valid = [(g, m) for g, m in model_set if g in oof_results and m in oof_results[g]]
    if len(valid) < 2:
        print("   Không đủ base models để stacking!")
        return None

    X_stack = np.column_stack([oof_results[g][m] for g, m in valid])
    print(f"   Sử dụng {len(valid)} base models: {[f'{g}_{m}' for g, m in valid]}")

    meta_candidates = {
        "Ridge": lambda: Ridge(alpha=1.0, random_state=42),
        "Lasso": lambda: Lasso(alpha=0.1, random_state=42),
        "ElasticNet": lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    }

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = {}

    for name, meta_fn in meta_candidates.items():
        meta = meta_fn()
        rmses = []
        for train_idx, val_idx in kf.split(X_stack):
            meta.fit(X_stack[train_idx], y_true[train_idx])
            pred = meta.predict(X_stack[val_idx])
            rmses.append(np.sqrt(mean_squared_error(y_true[val_idx], pred)))
        cv_scores[name] = np.mean(rmses)
        print(f"   {name:10}: CV RMSE = {cv_scores[name]:.5f}")

    best_meta_name = min(cv_scores, key=cv_scores.get)
    print(f"\nBEST META LEARNER: {best_meta_name} (CV RMSE: {cv_scores[best_meta_name]:.5f})")

    # === TUNING BEST META ===
    def objective(trial):
        if best_meta_name == "Ridge":
            model = Ridge(alpha=trial.suggest_float("alpha", 0.01, 100, log=True), random_state=42)
        elif best_meta_name == "Lasso":
            model = Lasso(alpha=trial.suggest_float("alpha", 0.001, 10, log=True), random_state=42)
        elif best_meta_name == "ElasticNet":
            model = ElasticNet(
                alpha=trial.suggest_float("alpha", 0.001, 10, log=True),
                l1_ratio=trial.suggest_float("l1_ratio", 0.1, 0.9),
                random_state=42
            )


        rmses = []
        for train_idx, val_idx in kf.split(X_stack):
            model.fit(X_stack[train_idx], y_true[train_idx])
            pred = model.predict(X_stack[val_idx])
            rmses.append(np.sqrt(mean_squared_error(y_true[val_idx], pred)))
        return np.mean(rmses)

    # Tắt log Optuna
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Lấy best model
    best_params = study.best_params
    best_cv_rmse = study.best_value
    best_model = None
    if best_meta_name == "Ridge":
        best_model = Ridge(alpha=best_params["alpha"], random_state=42)
    elif best_meta_name == "Lasso":
        best_model = Lasso(alpha=best_params["alpha"], random_state=42)
    elif best_meta_name == "ElasticNet":
        best_model = ElasticNet(alpha=best_params["alpha"], l1_ratio=best_params["l1_ratio"], random_state=42)
    elif best_meta_name == "RF":
        best_model = RandomForestRegressor(**{k: v for k, v in best_params.items() if k != "alpha"}, random_state=42, n_jobs=-1)
    elif best_meta_name == "GB":
        best_model = GradientBoostingRegressor(**{k: v for k, v in best_params.items() if k != "alpha"}, random_state=42)
    else:
        best_model = XGBRegressor(**{k: v for k, v in best_params.items() if k != "alpha"}, random_state=42, n_jobs=-1, tree_method='hist')

    best_model.fit(X_stack, y_true)
    final_pred = best_model.predict(X_stack)
    final_rmse = np.sqrt(mean_squared_error(y_true, final_pred))

    print(f"   FINAL STACKING RMSE: {final_rmse:.5f} (vs CV: {best_cv_rmse:.5f})")

    return {
        "pred": final_pred,
        "rmse": final_rmse,
        "cv_rmse": best_cv_rmse,
        "meta_name": best_meta_name,
        "meta_params": best_params,
        "meta_model": best_model,
        "base_models": [f"{g}_{m}" for g, m in valid]
    }


def run_stacking_ensemble(df, dataset_name, base_models, model_groups, config, n_trials=50):
    print(f"\n{'='*80}")
    print(f"STACKING - {dataset_name.upper()}")
    print(f"{'='*80}")

    oof_results, cv_scores, _, _ = run_per_group_with_oof(
        df=df, base_models=base_models, model_groups=model_groups, config=config, n_splits=5
    )
    y_true = df["SalePrice"].values

    model_sets = get_three_model_sets(oof_results, cv_scores)
    sets_info = [
        (model_sets["all_models"], "all_models"),
        (model_sets["top4_global"], "top4_global"),
        (model_sets["top1_per_group"], "top1_per_group")
    ]

    stacking_results = {}
    figs_meta = {}
    idx = 1
    for model_set, set_name in sets_info:
        if len(model_set) < 2:
            continue
        res = run_stacking_for_set(oof_results=oof_results, y_true=y_true, model_set=model_set, set_name=set_name, n_trials=n_trials)
        if res:
            stacking_results[set_name] = res

            # VẼ BIỂU ĐỒ ĐÓNG GÓP CỦA BASE MODELS
            fig = plt.figure(figsize=(10, 5))
            base_names = res["base_models"]
            X_stack = np.column_stack([oof_results[g][m] for g, m in model_set if f"{g}_{m}" in base_names])
            corrs = [np.corrcoef(res["pred"], X_stack[:, i])[0,1] for i in range(X_stack.shape[1])]
            plt.barh(base_names, corrs, color='purple', alpha=0.7)
            plt.title(f"{set_name.upper()} - Correlation of Base Models with Stacking Pred")
            plt.xlabel("Correlation")
            for i, c in enumerate(corrs):
                plt.text(c + 0.001, i, f"{c:.3f}", va='center')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            fname = f"{idx:02d}_correlation_{set_name}.png"
            figs_meta[set_name] = (fig, fname)
            idx += 1

    full_results = {
        "oof_results": oof_results, "cv_scores": cv_scores,
        "model_sets": model_sets, "stacking_results": stacking_results,
        "figs": {"meta_corr": figs_meta}
    }
    save_folder = save_ensemble_complete(full_results, y_true, dataset_name, method="stacking")

    return full_results, save_folder
