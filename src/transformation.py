import numpy as np
from sklearn.preprocessing import PowerTransformer, FunctionTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from scipy import stats
from scipy.stats.mstats import winsorize
import warnings

def safe_clip(X):
    return np.clip(np.nan_to_num(X, nan=0.0), -1e6, 1e6)

def winsorize_normalize(X, limits=(0.01, 0.01)):
    X = np.array(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    for i in range(X.shape[1]):
        X[:, i] = winsorize(X[:, i], limits=limits)
    return X

def iqr_cap(X):
    """
    Cắt ngoại lai theo quy tắc IQR:
      - Giới hạn dưới = Q1 - 1.5*IQR
      - Giới hạn trên = Q3 + 1.5*IQR
    """
    X = np.array(X, dtype=float)
    for i in range(X.shape[1]):
        q1, q3 = np.percentile(X[:, i], [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        X[:, i] = np.clip(X[:, i], lower, upper)
    return X


# SafeYeoJohnson: xử lý cột gây lỗi
class SafeYeoJohnson(BaseEstimator, TransformerMixin):
    def __init__(self, clip=1e6):
        self.clip = clip
        self.pt_ = PowerTransformer(method='yeo-johnson', standardize=False)
        self.constant_cols_ = []
        self.fitted_ = False

    def fit(self, X, y=None):
        X = np.nan_to_num(X, nan=0.0, posinf=0, neginf=0)
        X = np.clip(X, -self.clip, self.clip)

        # Tìm cột constant
        var = np.var(X, axis=0)
        self.constant_cols_ = np.where(np.allclose(var, 0))[0]

        if len(self.constant_cols_) == X.shape[1]:
            self.fitted_ = False
            return self

        X_fit = np.delete(X, self.constant_cols_, axis=1)
        try:
            self.pt_.fit(X_fit)
            self.fitted_ = True
        except Exception as e:
            warnings.warn(f"Fit failed: {e}")
            self.fitted_ = False
        return self

    def transform(self, X):
        X = np.nan_to_num(X, nan=0.0, posinf=0, neginf=0)
        X = np.clip(X, -self.clip, self.clip)

        if not self.fitted_ or X.shape[1] == 0:
            return X

        X_t = X.copy()
        X_non_const = np.delete(X_t, self.constant_cols_, axis=1)
        X_trans = self.pt_.transform(X_non_const)

        # Gộp lại
        result = np.zeros_like(X)
        result[:, self.constant_cols_] = X[:, self.constant_cols_]
        non_const_idx = [i for i in range(X.shape[1]) if i not in self.constant_cols_]
        for i, col_idx in enumerate(non_const_idx):
            result[:, col_idx] = X_trans[:, i]
        return result
    
class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lambda_ = None

    def fit(self, y, X=None):
        y = check_array(y, ensure_2d=False)
        y_pos = y + 1 if np.any(y <= 0) else y
        _, self.lambda_ = stats.boxcox(y_pos)
        return self

    def transform(self, y):
        y = check_array(y, ensure_2d=False)
        y_pos = y + 1 if np.any(y <= 0) else y
        return stats.boxcox(y_pos, lmbda=self.lambda_)

    def inverse_transform(self, y):
        y = check_array(y, ensure_2d=False)
        if self.lambda_ == 0:
            return np.exp(y) - 1
        else:
            return np.exp(np.log(self.lambda_ * y + 1) / self.lambda_) - 1
    

# Feature selection
def create_FS_data(df, stable_features, target_col="SalePrice"):
    corrected_features = []
    for feat in stable_features:
        if "_target_encoded" in feat:
            original_name = feat.replace("_target_encoded", "")
            corrected_features.append(original_name)
        else:
            corrected_features.append(feat)

    corrected_features = list(set(corrected_features)) # Loại bỏ trùng lặp
    fs_data = df[corrected_features + [target_col]].copy()
    return fs_data

# Target transforms
def apply_target_transformation(y, method):
    if method == "log":
        return np.log1p(y)
    elif method == "sqrt":
        return np.sqrt(y)
    elif method == "org":
        return y.copy()
    else:
        return y.copy()
    
def inverse_target_transformation(y_pred, method):
    if method == "log":
        return np.expm1(y_pred)
    elif method == "sqrt":
        return np.square(y_pred)
    elif method == "org":
        return y_pred.copy()
    else:
        return y_pred.copy()

def wrapper_model_with_target_transform(model, method):
    if method == "log":
        transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1)
    elif method == "sqrt":
        transformer = FunctionTransformer(np.sqrt, inverse_func=np.square)
    elif method == "yeo-johnson":
        transformer = SafeYeoJohnson()
    elif method == "box-cox":
        transformer = BoxCoxTransformer()
    else:
        return model  # org: no transform
    return TransformedTargetRegressor(regressor=model, transformer=transformer)