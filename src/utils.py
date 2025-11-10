import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import json
from datetime import datetime
import joblib
from typing import List, Tuple

# CẤU HÌNH ĐƯỜNG DẪN
ROOT_DIR = Path(__file__).parent.parent
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

def load_all_comparison_images():
    """
    Load tất cả ảnh trong results/final_comparison/
    """
    folder = RESULTS_DIR / "final_comparison"
    if not folder.exists():
        return []
    
    images = []
    for img_path in folder.glob("final_comparison_3way_*.png"):
        # Lấy dataset_name từ tên file
        dataset_key = img_path.stem.replace("final_comparison_3way_", "")
        caption = f"So sánh Ensemble – {dataset_key.replace('_', ' + ').title()}"
        images.append((caption, str(img_path), dataset_key))
    
    # Sắp xếp theo thời gian tạo
    images.sort(key=lambda x: os.path.getctime(x[1]), reverse=True)
    return images

def load_ensemble_visualizations(dataset_name: str, method: str) -> List[Tuple[str, str]]:
    """
    Tìm thư mục mới nhất theo dataset + method
    → Load tất cả ảnh trong visualizations/
    → Trả về list: [(caption, image_path), ...]
    """
    folder_pattern = RESULTS_DIR / f"{dataset_name}_{method}_*"
    import glob
    folders = glob.glob(str(folder_pattern))
    
    if not folders:
        return []
    
    # Lấy thư mục mới nhất
    latest_folder = Path(max(folders, key=os.path.getctime))
    vis_dir = latest_folder / "visualizations"
    
    if not vis_dir.exists():
        return []
    
    # Danh sách file ảnh (png, jpg, jpeg)
    image_extensions = ("*.png", "*.jpg", "*.jpeg")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(vis_dir.glob(ext))
    
    # Sắp xếp theo tên file (theo thứ tự lưu)
    image_paths = sorted(image_paths, key=lambda x: x.name)
    
    # Tạo caption từ tên file
    captions = {
        "04_rmse_comparison.png": "So sánh RMSE các mô hình",
        "05_improvement_heatmap.png": "Heatmap cải thiện RMSE",
    }
    
    results = []
    for img_path in image_paths:
        default_caption = img_path.stem.replace("_", " ").title()
        caption = captions.get(img_path.name, default_caption)
        results.append((caption, str(img_path)))
    
    return results

def analyze_oof_results(oof_result_path: str or Path, y_true: pd.Series = None):
    """
    Phân tích file oof_results.pkl → tính RMSE, Std, R² cho từng model
    Args:
        oof_result_path: đường dẫn đến file .pkl
        y_true: pd.Series chứa SalePrice thật (nếu không có → bỏ qua R²)
    Returns:
        pd.DataFrame: bảng kết quả
    """
    path = Path(oof_result_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {path}")

    data = joblib.load(path)
    results = []

    # === 1. Lấy y_true từ data (nếu có) ===
    if y_true is None and 'y_true' in data:
        y_true = data['y_true']
    elif y_true is None:
        st.warning("Không có y_true → bỏ qua R²")
        y_true = None

    # === 2. Duyệt từng nhóm model ===
    for group_name, models in data.items():
        if not isinstance(models, dict):
            continue  # bỏ qua key như best_weights, stacking_pred

        for model_name, oof_pred in models.items():
            if not isinstance(oof_pred, np.ndarray):
                continue

            # Chuyển về numpy array
            pred = np.array(oof_pred).flatten()
            if y_true is not None and len(pred) != len(y_true):
                continue

            # Tính metric
            rmse = np.sqrt(mean_squared_error(y_true, pred)) if y_true is not None else np.nan
            std = pred.std() if len(pred) > 1 else 0
            r2 = r2_score(y_true, pred) if y_true is not None else np.nan

            results.append({
                "Group": group_name.replace("_Models", ""),
                "Model": model_name,
                "RMSE": rmse,
                "Std": std,
                "R²": r2,
                "N": len(pred)
            })

    df = pd.DataFrame(results)

    # === 3. Tính Weighted Ensemble (nếu có) ===
    if 'best_weights' in data and y_true is not None:
        weights = data['best_weights']
        pred_sum = np.zeros(len(y_true))
        total_weight = 0
        for model_name, w in weights.items():
            full_name = None
            for group in data:
                if isinstance(data[group], dict) and model_name in data[group]:
                    full_name = data[group][model_name]
                    break
            if full_name is not None:
                pred_sum += w * np.array(full_name).flatten()
                total_weight += w
        if total_weight > 0:
            ensemble_pred = pred_sum / total_weight
            ensemble_rmse = np.sqrt(mean_squared_error(y_true, ensemble_pred))
            df.loc[len(df)] = {
                "Group": "Ensemble",
                "Model": "Weighted",
                "RMSE": ensemble_rmse,
                "Std": ensemble_pred.std(),
                "R²": r2_score(y_true, ensemble_pred),
                "N": len(ensemble_pred)
            }

    # === 4. Tính Stacking (nếu có) ===
    if 'stacking_pred' in data and y_true is not None:
        pred = np.array(data['stacking_pred']).flatten()
        stacking_rmse = np.sqrt(mean_squared_error(y_true, pred))
        df.loc[len(df)] = {
            "Group": "Ensemble",
            "Model": "Stacking",
            "RMSE": stacking_rmse,
            "Std": pred.std(),
            "R²": r2_score(y_true, pred),
            "N": len(pred)
        }

    # Sắp xếp
    df = df.round({"RMSE": 0, "Std": 0, "R²": 4})
    return df.sort_values("RMSE").reset_index(drop=True)
