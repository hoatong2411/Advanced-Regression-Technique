import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os

def plot_target_distribution(df, target_col="SalePrice"):
    plt.figure(figsize=(8, 5))
    sns.histplot(df[target_col], kde=True, color="skyblue")
    plt.axvline(df[target_col].mean(), color="red",  linewidth=2, linestyle="--", label="Mean")
    plt.title(f"Distribution of {target_col}")
    plt.legend()
    plt.show()

def plot_correlation_heatmap(df, target_col="SalePrice", top_n=15):
    numeric_data = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_data.corr()[target_col].abs().sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[corr.index].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Top {top_n} Correlations with {target_col}")
    plt.tight_layout()
    plt.show()

def plot_missing_data(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    plt.figure(figsize=(10, 8))
    missing.plot.barh(color="skyblue", edgecolor="black")
    plt.title("Missing Data by Feature", fontsize=14)
    plt.xlabel("Number of Missing Values")
    plt.ylabel("Feature Name")
    plt.grid(axis="x")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("missing_data.png")
    plt.show()

def plot_baseline_results(baseline_results):
    """ 
    Visualization: So sánh RMSE giữa các mô hình
    """
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=baseline_results,
        x="Model",
        y="Mean_RMSE",
        order=baseline_results.sort_values("Mean_RMSE")["Model"]
    )

    plt.title("Baseline RMSE giữa các mô hình", fontsize=14, weight='bold')
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Mean RMSE", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # 💬 Thêm nhãn giá trị trên đầu mỗi cột
    for p in ax.patches:
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2,  
            height + 0.002,                 
            f"{height:.0f}",                
            ha="center", va="bottom", fontsize=10, color="black", weight="bold"
        )

    plt.tight_layout()
    plt.show()


def visualize_all_models_contribution(weighted_res, stacking_res, dataset_name, save_dir="results/all_models_viz"):
    """
    Visualize đóng góp của từng model trong:
      - Weighted Ensemble: Trọng số (weights)
      - Stacking: Correlation với dự đoán cuối
    Chỉ dùng cho set 'all_models'
    """
    os.makedirs(save_dir, exist_ok=True)
    set_name = "all_models"

    # === 1. LẤY DỮ LIỆU ===
    w_res = weighted_res["ensemble_results"].get(set_name)
    s_res = stacking_res["stacking_results"].get(set_name)
    oof_results = weighted_res["oof_results"]  # hoặc stacking_res["oof_results"]

    if not w_res or not s_res:
        print(f"   Không có kết quả cho {set_name}")
        return

    # Lấy danh sách model
    model_keys = [f"{g}_{m}" for g, m in weighted_res["model_sets"][set_name]]
    X_stack = np.column_stack([oof_results[g][m] for g, m in weighted_res["model_sets"][set_name]])

    # === 2. VẼ WEIGHTED: TRỌNG SỐ ===
    plt.figure(figsize=(10, 8))
    weights = [w_res["hill_weights"].get(k, 0) for k in model_keys]
    colors = ['#3498db' if w > 0 else '#e74c3c' for w in weights]

    bars = plt.barh(model_keys, weights, color=colors, edgecolor='black')
    plt.xlabel("Weight", fontsize=12)
    plt.title(f"{dataset_name.upper()} - Weighted Ensemble Weights (All Models)\nRMSE: {w_res['hill_rmse']:,.0f}", fontsize=14, pad=20)

    # Ghi số
    for i, (bar, w) in enumerate(zip(bars, weights)):
        plt.text(w + (0.001 if w >= 0 else -0.001), bar.get_y() + bar.get_height()/2,
                 f"{w:.4f}", va='center', ha='left' if w >= 0 else 'right', fontsize=9, fontweight='bold')

    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    weight_path = os.path.join(save_dir, f"01_weighted_weights_{dataset_name}.png")
    plt.savefig(weight_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"   Đã lưu: {weight_path}")

    # === 3. VẼ STACKING: CORRELATION ===
    plt.figure(figsize=(10, 8))
    stacking_pred = s_res["pred"]
    corrs = [pearsonr(X_stack[:, i], stacking_pred)[0] for i in range(X_stack.shape[1])]
    colors = ['#9b59b6' if c > 0.5 else '#95a5a6' for c in corrs]

    bars = plt.barh(model_keys, corrs, color=colors, edgecolor='black')
    plt.xlabel("Correlation with Stacking Prediction", fontsize=12)
    plt.title(f"{dataset_name.upper()} - Stacking Base Model Contribution (All Models)\nRMSE: {s_res['rmse']:,.0f}", fontsize=14, pad=20)

    # Ghi số
    for i, (bar, c) in enumerate(zip(bars, corrs)):
        plt.text(c + 0.001, bar.get_y() + bar.get_height()/2,
                 f"{c:.3f}", va='center', ha='left', fontsize=9, fontweight='bold')

    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    corr_path = os.path.join(save_dir, f"02_stacking_correlation_{dataset_name}.png")
    plt.savefig(corr_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"   Đã lưu: {corr_path}")

    # === 4. BẢNG TỔNG HỢP ===
    summary = pd.DataFrame({
        "Model": model_keys,
        "Weighted_Weight": weights,
        "Stacking_Correlation": corrs
    }).round(4)
    print("\nTỔNG HỢP ĐÓNG GÓP (ALL MODELS)")
    print(summary.to_string(index=False))

    csv_path = os.path.join(save_dir, f"contribution_summary_{dataset_name}.csv")
    summary.to_csv(csv_path, index=False)
    print(f"   Đã lưu bảng: {csv_path}")
