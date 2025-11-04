import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import json
from datetime import datetime
import joblib

# CẤU HÌNH ĐƯỜNG DẪN
ROOT_DIR = Path.cwd().parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# LOAD & SAVE
def load_df(file_name, folder="data"):
    path = ROOT_DIR / folder / file_name
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported format: {ext}")
    print(f"Loaded: {path}")
    return df

def save_df(df, file_name, folder="results"):
    path = ROOT_DIR / folder / file_name
    ext = path.suffix.lower()
    if ext == ".csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
    elif ext in [".xlsx", ".xls"]:
        df.to_excel(path, index=False)
    else:
        raise ValueError("Use .csv or .xlsx")
    print(f"Saved: {path}")
    return path

def load_json(file_name, folder="data"):
    path = ROOT_DIR / folder / file_name
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
     
    ext = path.suffix.lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    print(f"Loaded: {path}")
    return data

def save_json(data, file_name, folder="data"):
    path = ROOT_DIR / folder / file_name
    ext = path.suffix.lower()

    if ext != ".json":
        raise ValueError("Use .json.")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Saved: {path}")
    return path

# GET CONFIG
def get_param(config_dict, group_name, attribute : str, default=None):
    if not isinstance(config_dict, dict):
        return default
    param_dict = config_dict.get(attribute, {})
    return param_dict.get(group_name, default)

def load_config(file_name="optimal_config.json", folder="data"):
    return load_json(file_name, folder)

# Compare ensembles
def compare_ensembles(weighted_res, stacking_res, dataset_name):
    print(f"\n{'='*80}")
    print(f"SO SÁNH: BEST SINGLE vs WEIGHTED vs STACKING - {dataset_name.upper()}")
    print(f"{'='*80}")

    cv_scores = weighted_res["cv_scores"]
    rows = []
    for group, models in cv_scores.items():
        for model, rmse in models.items():
            rows.append({"Group": group, "Model": model, "Mean_RMSE": rmse})
    baseline_df = pd.DataFrame(rows)
    best_single_rmse = baseline_df.nsmallest(1, "Mean_RMSE").iloc[0]["Mean_RMSE"]
    best_single_name = baseline_df.nsmallest(1, "Mean_RMSE").iloc[0][["Group", "Model"]].tolist()
    best_single_label = f"{best_single_name[0]}_{best_single_name[1]}"

    sets = ["all_models", "top4_global", "top1_per_group"]
    data = []
    for s in sets:
        w = weighted_res["ensemble_results"].get(s)
        st = stacking_res["stacking_results"].get(s)
        data.append({
            "Set": s,
            "Best_Single_RMSE": best_single_rmse,
            "Weighted_RMSE": w["hill_rmse"] if w else np.nan,
            "Stacking_RMSE": st["rmse"] if st else np.nan
        })
    df = pd.DataFrame(data).round(2)
    print(df.to_string(index=False))

    # VẼ 3 CỘT: Best Single | Weighted | Stacking
    plt.figure(figsize=(12, 7))
    x = np.arange(len(sets))
    width = 0.25

    # Cột 1: Best Single (xanh lá)
    plt.bar(x - width, df["Best_Single_RMSE"], width, label="Best Single", color="#27ae60", edgecolor="black")

    # Cột 2: Weighted (xanh dương)
    plt.bar(x, df["Weighted_RMSE"], width, label="Weighted", color="#3498db", edgecolor="black")

    # Cột 3: Stacking (đỏ)
    plt.bar(x + width, df["Stacking_RMSE"], width, label="Stacking", color="#e74c3c", edgecolor="black")

    plt.xticks(x, [s.replace("_", " ").title() for s in sets], rotation=15)
    plt.ylabel("RMSE")
    plt.title(f"{dataset_name.upper()} - Best Single vs Weighted vs Stacking")
    plt.legend(fontsize=10)
    plt.grid(axis="y", alpha=0.3)

    # Ghi số lên đầu cột
    for i in range(len(sets)):
        bs = df.iloc[i]["Best_Single_RMSE"]
        w = df.iloc[i]["Weighted_RMSE"]
        s = df.iloc[i]["Stacking_RMSE"]

        plt.text(i - width, bs + 80, f"{bs:,.0f}", ha="center", fontsize=9, fontweight="bold")
        if not np.isnan(w):
            plt.text(i, w + 80, f"{w:,.0f}", ha="center", fontsize=9, fontweight="bold")
        if not np.isnan(s):
            plt.text(i + width, s + 80, f"{s:,.0f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()

    # Lưu ảnh
    save_dir = os.path.join(os.path.dirname(weighted_res.get("save_folder", "")), "final_comparison")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"final_comparison_3way_{dataset_name}.png"),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    print(f"Đã lưu biểu đồ so sánh 3 cột tại:\n   {save_dir}")

# ========= ENSEMBLE ===========
def save_ensemble_complete(full_results, y_true, dataset_name, method="weighted", base_folder="results"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(base_folder, f"{dataset_name}_{method}_{timestamp}")
    vis_dir = os.path.join(folder, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # === LƯU DỮ LIỆU ===
    joblib.dump(full_results["oof_results"], os.path.join(folder, "oof_results.pkl"))
    joblib.dump(full_results["cv_scores"], os.path.join(folder, "cv_scores.pkl"))

    # === LƯU ẢNH CHO BÁO CÁO ===
    if method == "weighted":
        # 1. Trọng số
        for set_name, (fig, fname) in full_results["figs"]["weight_figs"].items():
            plt.figure(fig.number)
            plt.savefig(os.path.join(vis_dir, fname))
            plt.close()
        # 2. RMSE + Heatmap
        plt.figure(full_results["figs"]["rmse"].number)
        plt.savefig(os.path.join(vis_dir, "04_rmse_comparison.png"))
        plt.close()
        plt.figure(full_results["figs"]["heatmap"].number)
        plt.savefig(os.path.join(vis_dir, "05_improvement_heatmap.png"))
        plt.close()

    elif method == "stacking":
        if "meta_corr" in full_results["figs"] and full_results["figs"]["meta_corr"]:
            for set_name, (fig, fname) in full_results["figs"]["meta_corr"].items():
                plt.figure(fig.number)
                plt.savefig(os.path.join(vis_dir, fname), dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
            print(f"   Đã lưu {len(full_results['figs']['meta_corr'])} ảnh correlation")
        else:
            print("   Không có set nào đủ điều kiện → không lưu ảnh correlation")

    print(f"ĐÃ LƯU {method.upper()} + ẢNH BÁO CÁO TẠI:\n   {vis_dir}")
    return folder