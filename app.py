# app.py
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
from pathlib import Path
import markdown

# === Cấu hình trang ===
st.set_page_config(
    page_title="Ames Housing Price Prediction",
    page_icon="house",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Đường dẫn ===
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
SRC_DIR = ROOT_DIR / "src"

import sys
sys.path.append(str(SRC_DIR))

# === Import từ src ===
from src.utils import load_df, save_df, load_json, load_ensemble_visualizations, load_all_comparison_images
from src.preprocessing import get_optimal_preprocessor, calculate_vif, calculate_mi, handle_missing_and_encode
from src.modeling import (
    get_base_models, run_optuna_weighted_ensemble, run_stacking_ensemble, 
    baseline_evaluate_dataset,
    
)
from src.transformation import wrapper_model_with_target_transform
from src.visualization import plot_target_distribution, plot_correlation_heatmap

# === Load config & data ===
@st.cache_data
def load_config():
    return load_json("optimal_config.json", folder="data")

@st.cache_data
def load_data():
    df = load_df("train-house-prices-advanced-regression-techniques.csv", folder="data")
    return df

# === Load trained ensemble (nếu có) ===
@st.cache_resource
def load_ensemble(dataset_name="FE_FS_data", method="weighted"):
    folder = RESULTS_DIR / f"{dataset_name}_{method}_*"
    import glob
    folders = glob.glob(str(folder))
    if not folders:
        return None
    latest_folder = max(folders, key=Path)
    oof_path = Path(latest_folder) / "oof_results.pkl"
    if oof_path.exists():
        return joblib.load(oof_path)
    return None

def render_markdown_to_html(md_path: str):
    if not Path(md_path).exists():
        st.error(f"Không tìm thấy file: `{md_path}`")
        return

    md_text = Path(md_path).read_text(encoding="utf-8")
    st.markdown(md_text, unsafe_allow_html=True)

# === Main App ===
def main():
    st.title("Ames Housing Price Prediction")
    st.markdown("**Advanced Regression Technique**")

    config = load_config()
    house_df = load_data()

    # Sidebar
    with st.sidebar:
        st.header("Cấu hình")
        task = st.selectbox("Chọn tác vụ", [
            "Tổng quan",
            "Xem dữ liệu & EDA",
            "Huấn luyện mô hình (Baseline)",
            "Huấn luyện Ensemble (Weighted)",
            "Huấn luyện Ensemble (Stacking)",
            "So sánh Ensemble",
            "Dự đoán giá nhà"
        ])

        st.markdown("---")
        st.caption(f"Data: {len(house_df):,} mẫu | Features: {house_df.shape[1]-1}")

    # === 0. Tổng quan ===
    if task == "Tổng quan":
        st.title("Tổng quan Dự án")
        render_markdown_to_html("PROJECT_SUMMARY.md")

    # === 1. EDA ===
    elif task == "Xem dữ liệu & EDA":
        st.subheader("Khám phá dữ liệu")

        # --- Dòng 1: Thông tin tổng quan ---
        with st.container():
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric("Số mẫu", f"{len(house_df):,}")
            with col2:
                st.metric("Số đặc trưng", f"{house_df.shape[1]-1}")
        
        st.markdown("---")
        
        # --- Dòng 2: Bảng dữ liệu ---
        with st.container():
            st.subheader("**5 dòng đầu tiên**")
            st.dataframe(house_df.head(), use_container_width=True)

        st.markdown("---")

        # --- Dòng 3: Thông tin Các cột missing ---
        with st.container():
            st.subheader("**Các cột thiếu dữ liệu**")
            missing = house_df.isnull().sum().sort_values(ascending=False)
            missing = missing[missing > 0]
            missing_perc = (missing / len(house_df)) * 100
            missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_perc})
            st.dataframe(missing_df, use_container_width=True)

        st.markdown("---")

        # --- Dòng 4: Phân phối giá nhà ---
        with st.container():
            st.subheader("**Phân phối giá nhà (SalePrice)**")
            col_empty, col_chart, col_empty = st.columns([1, 8, 1])
    
            with col_chart:
                plot_target_distribution(house_df, "SalePrice")
                fig = plt.gcf()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        st.markdown("---")

        # --- Dòng 5: Tương quan với giá nhà ---
        with st.container():
            st.subheader("**Top 10 đặc trưng tương quan mạnh nhất với SalePrice**")

            col_empty, col_chart, col_empty = st.columns([1, 8, 1])
    
            with col_chart:
                plot_correlation_heatmap(house_df, "SalePrice", top_n=10)
                fig = plt.gcf()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        raw_data = load_df("raw_data.csv", folder="data")

        # --- Dòng 6: Kiểm tra Multicollinearity của 
        with st.container():
            st.subheader("**Kiểm tra đa cộng tuyến của bộ dữ liệu gốc**")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("**VIF - Data gốc (Top 15)**")
                vif_original = calculate_vif(raw_data)
                st.dataframe(vif_original.head(15), use_container_width=True)
            with col2:
                st.subheader("MI Score với Target - Data gốc (Top 15)")
                mi_original = calculate_mi(raw_data)
                st.dataframe(mi_original.head(15), use_container_width=True)   
        
        st.markdown("---")
        
        FE_data = load_df("FE_data.csv", folder="data")
        with st.container():
            st.subheader("**Kiểm tra đa cộng tuyến của bộ dữ liệu đã Feature Engineering**")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("**VIF - FE Data (Top 15)**")
                vif_FE = calculate_vif(FE_data)
                st.dataframe(vif_FE.head(15), use_container_width=True)
            with col2:
                st.subheader("MI Score với Target - FE Data (Top 15)")
                mi_FE = calculate_mi(FE_data)
                st.dataframe(mi_FE.head(15), use_container_width=True)   



    # === 2. Baseline ===
    data = load_df("raw_FS_data.csv", folder="data")
    base_models = get_base_models()
    model_groups = {
            'Linear_Models': ['Ridge', 'Lasso', 'ElasticNet', 'Huber', 'Quantile'],
            'Tree_Models': ['RF', 'XGB', 'GB', 'LGBM', 'ADB'],
            'Special_Models': ['CatBoost', 'RANSAC']
        }

    if task == "Huấn luyện mô hình (Baseline)":
        st.subheader("Huấn luyện mô hình cơ bản")
        col1, col2 = st.columns([1, 1])
        with col1:
            model_name = st.selectbox("Chọn mô hình", list(base_models.keys()))
        with col2:
            n_splits = st.selectbox("Số fold (KFold)", np.arange(3, 11), index=1)

        if st.button("Chạy huấn luyện"):
            with st.spinner(f"Đang huấn luyện {model_name} với {n_splits}-fold CV..."):
                selected_model = {model_name : base_models[model_name]}
                baseline_results = baseline_evaluate_dataset(data, "data", selected_model, n_splits=n_splits).sort_values(by=["Mean_RMSE"])
                st.dataframe(baseline_results, use_container_width=True)

    # === 3. Weighted Ensemble ===
    elif task == "Huấn luyện Ensemble (Weighted)":
        st.subheader("Weighted Ensemble (Optuna)")
        dataset_options = {
            "Raw Data": "raw_data",
            "Raw + Feature Selection": "raw_FS_data",
            "Feature Engineering": "FE_data",
            "FE + Feature Selection": "FE_FS_data"
        }
        selected_name = st.selectbox("Chọn bộ dữ liệu", list(dataset_options.keys()))
        dataset_key = dataset_options[selected_name]

        # === HIỂN THỊ Kết quả đã có ===
        if st.button("Chạy Ensemble"):
            images = load_ensemble_visualizations(dataset_key, "weighted")
            if images:
                st.markdown("---")
                st.subheader("Biểu đồ trực quan")
                for caption, img_path in images:
                    st.write(f"**{caption}**")
                    st.image(img_path, use_container_width=True)

        # === Chạy thực nghiệm ===
        # col1, col2 = st.columns([1, 1])
        # with col1:
        #     if st.button("Chạy Ensemble"):
        #         with st.spinner("Đang tối ưu trọng số..."):
        #             if result:
        #                 pass
        #                 # data_path = DATA_DIR / f"{dataset_key}.csv"
        #                 # if data_path.exists():
        #                 #     df_data = load_df(data_path.name, folder="data")

        #                 #     weighted_res, _ = run_optuna_weighted_ensemble(df_data, selected_name, base_models, model_groups, config)
        #                 #     res_df = pd.DataFrame(weighted_res)
        #                 #     st.dataframe(res_df, use_container_width=True)
        # with col2:
        #     st.caption(f"Dataset: `{dataset_key}` | Method: `weighted`")    

    # === 4. Stacking Ensemble ===
    elif task == "Huấn luyện Ensemble (Stacking)":
        st.subheader("Stacking Ensemble")
        dataset_options = {
            "Raw Data": "raw_data",
            "Raw + Feature Selection": "raw_FS_data",
            "Feature Engineering": "FE_data",
            "FE + Feature Selection": "FE_FS_data"
        }
        selected_name = st.selectbox("Chọn bộ dữ liệu", list(dataset_options.keys()))
        dataset_key = dataset_options[selected_name]      

        # === HIỂN THỊ Kết quả đã có ===
        if st.button("Chạy Stacking"):
            images = load_ensemble_visualizations(dataset_key, "stacking")
            if images:
                st.markdown("---")
                st.subheader("Biểu đồ trực quan")
                for caption, img_path in images:
                    st.write(f"**{caption}**")
                    st.image(img_path, use_container_width=True)

        # === Chạy thực nghiệm ===
        # col1, col2 = st.columns([1, 1])
        # with col1:
        #     if st.button("Chạy Stacking"):
        #         with st.spinner("Đang tối ưu trọng số..."):
        #             if result:
        #                 pass
        #                 # data_path = DATA_DIR / f"{dataset_key}.csv"
        #                 # if data_path.exists():
        #                 #     df_data = load_df(data_path.name, folder="data")

        #                 #     stacking_res, _ = run_stacking_ensemble(df_data, selected_name, base_models, model_groups, config)
        #                 #     res_df = pd.DataFrame(stacking_res)
        #                 #     st.dataframe(res_df, use_container_width=True)
        # with col2:
        #     st.caption(f"Dataset: `{dataset_key}` | Method: `stacking`")   
    
    elif task == "So sánh Ensemble":
        st.subheader("So sánh: Best Single vs Weighted vs Stacking")

        # Tự động load tất cả ảnh có sẵn
        comparison_images = load_all_comparison_images()

        # Hiển thị theo tab hoặc dropdown
        dataset_names = {cap: key for cap, _, key in comparison_images}
        selected_caption = st.selectbox("Chọn bộ dữ liệu", list(dataset_names.keys()))
        selected_key = dataset_names[selected_caption]
        selected_img = next(img for cap, img, key in comparison_images if key == selected_key)

        st.markdown("---")
        st.write(f"**{selected_caption}**")
        st.image(selected_img, use_container_width=True)

    # === 6. Dự đoán ===
    elif task == "Dự đoán giá nhà":
        st.subheader("Dự đoán giá nhà mới")
        st.info("Tính năng này cần file `test.csv` và mô hình đã lưu. Hiện tại đang phát triển...")

# === Run ===
if __name__ == "__main__":
    main()