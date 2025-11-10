import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler,
    OneHotEncoder, FunctionTransformer, PolynomialFeatures
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from .transformation import (
    winsorize_normalize, iqr_cap, SafeYeoJohnson,
    BoxCoxTransformer, safe_clip
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42

def calculate_vif(df, features=None):
    if features is None:
        features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        # Loại bỏ target nếu có
        if "SalePrice" in features:
            features.remove("SalePrice")
    
    X = df[features].copy()
    X = X.fillna(X.mean())
    vif = pd.DataFrame()
    vif["Feature"] = features
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif.sort_values("VIF", ascending=False)

def calculate_mi(df, target="SalePrice", features=None):
    if features is None:
        features = df.select_dtypes(include=["int64", "float64"]).columns.drop(target, errors="ignore")
    
    X = df[features].copy()
    X = X.fillna(X.mean())
    y = df[target]
    
    # Scale để MI ổn định hơn
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    mi_scores = mutual_info_regression(X_scaled, y, random_state=42)
    mi = pd.DataFrame({"Feature": features, "MI_Score": mi_scores})
    return mi.sort_values("MI_Score", ascending=False)

def handle_missing_and_encode(df):
    """Ames-specific: ordinal + fill + one-hot"""
    df = df.copy()
    
    # Fill missing
    fill_dict = {
        "BsmtQual": "NoBasement", "BsmtCond": "NoBasement", "BsmtExposure": "NoBasement",
        "BsmtFinType1": "NoBasement", "BsmtFinType2": "NoBasement"
    }
    df = df.fillna(value=fill_dict)

    # Ordinal mapping
    qual_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, np.nan: 0, "NoBasement": 0}
    exposure_map = {"Gd": 3, "Av": 2, "Mn": 1, "No": 0, "NoBasement": 0}
    
    ordinal_cols = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure",
                    "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond"]
    for col in ordinal_cols:
        if col in df.columns:
            df[col] = df[col].map(qual_map if col != "BsmtExposure" else exposure_map)
    
    # One-hot
    one_hot_cols = ["BsmtFinType1", "BsmtFinType2"]
    if all(col in df.columns for col in one_hot_cols):
        df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)
    
    return df

def build_feature_selector(method, X_train, y_train):
    """
    Xây dựng bước chọn lọc đặc trưng dựa trên phương pháp.
    - method: str hoặc dict {model: method}
    - X_train: dữ liệu đã qua preprocessor (numpy array)
    - y_train: target
    """
    if method is None or method == "none":
        return ("passthrough", None)
    if X_train is None or y_train is None:
        return ("passthrough", None)

    if method == "selectkbest":
        k = min(100, X_train.shape[1] // 2)  # chọn tối đa 100 hoặc nửa số đặc trưng
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X_train, y_train)
        return ("selectkbest", selector)

    elif method == "lassocv":
        # Dùng LassoCV để chọn đặc trưng (coef != 0)
        lasso = LassoCV(cv=3, random_state=42, max_iter=10000)
        lasso.fit(X_train, y_train)
        selected = lasso.coef_ != 0
        if not np.any(selected):
            selected = np.arange(X_train.shape[1])  # fallback
        return ("lassocv", FunctionTransformer(lambda X: X[:, selected], validate=False))

    elif method == "pca":
        n_components = min(50, X_train.shape[1] // 3)
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X_train)
        return ("pca", pca)

    else:
        return ("passthrough", None)

def get_optimal_preprocessor(
    X_train, 
    y_train=None, 
    group="default", 
    config=None
):
    """
    Tạo preprocessor tối ưu theo config, trả về:
    - preprocessor: ColumnTransformer
    - target_transform: str (tên method để dùng với TransformedTargetRegressor)
    - selector: (name, selector) hoặc ("passthrough", None)
    """
    if config is None:
        raise ValueError("config is required")

    # === LẤY COLUMNS ===
    num_cols = X_train.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=["float64", "int64"]).columns.tolist()

    # === LẤY CONFIG CHO GROUP ===
    imp_type = config["imputer_type"].get(group, "simple")
    scl_type = config["scaler_type"].get(group, "minmax")
    enc_type = config["encoder_type"].get(group, "onehot")
    outlier = config["outlier_method"].get(group, None)
    in_tf = config["input_transform"].get(group, "org")
    tgt_tf = config["target_transform"].get(group, "org")
    multicol = config["multicollinearity"].get(group, None)
    poly = config["poly_features"].get(group, False)
    knn_k = config["knn_neighbors"].get(group, 5)
    tg_smoothing = config["target_encoder_smoothing"].get(group, 10)

    # === IMPUTER ===
    imputer = {
        "knn": KNNImputer(n_neighbors=knn_k),
        "iterative": IterativeImputer(random_state=RANDOM_STATE),
        "simple_median": SimpleImputer(strategy="median"),
        "simple": SimpleImputer(strategy="mean")
    }.get(imp_type, SimpleImputer(strategy="mean"))

    # === SCALER ===
    scaler = {
        "standard": StandardScaler(),
        "robust": RobustScaler(),
        "minmax": MinMaxScaler()
    }[scl_type]

    # === ENCODER ===
    if enc_type == "target":
        if y_train is None:
            raise ValueError("y_train required for TargetEncoder")
        encoder = TargetEncoder(cols=cat_cols, smoothing=tg_smoothing)
        encoder.fit(X_train[cat_cols], y_train)
    else:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # === NUMERIC PIPELINE ===
    num_steps = [("imputer", imputer)]

    # Outlier handling
    if outlier == "winsor":
        num_steps.append(("winsor", FunctionTransformer(winsorize_normalize, validate=False)))
    elif outlier == "iqr":
        num_steps.append(("iqr", FunctionTransformer(iqr_cap, validate=False)))

    # Input transformation
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

    # Polynomial features (trước selector)
    if poly:
        num_steps.insert(-2, ("poly", PolynomialFeatures(degree=2, include_bias=False)))

    num_pipeline = Pipeline(num_steps)

    # === CATEGORICAL PIPELINE ===
    # Chỉ scale nếu dùng TargetEncoder
    cat_pipeline = Pipeline([
        ("encoder", encoder),
        ("scaler", StandardScaler()) if enc_type == "target" else ("passthrough", None)
    ])

    # === PREPROCESSOR ===
    transformers = [("num", num_pipeline, num_cols)]
    if cat_cols:
        transformers.append(("cat", cat_pipeline, cat_cols))

    preprocessor = ColumnTransformer(transformers)

    # === FEATURE SELECTION ===
    selector_name, selector = build_feature_selector(method=multicol, X_train=None, y_train=None)
    if selector_name == "passthrough":
        selector = None

    return preprocessor, tgt_tf, (selector_name, selector)
