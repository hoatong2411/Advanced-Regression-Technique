## 1. Tổng quan dự án – End-to-end ML Pipeline

Dự án tối ưu hóa mô hình dự đoán giá nhà (Ames Housing) qua pipeline toàn diện: EDA, tiền xử lý, feature engineering, feature selection, huấn luyện nhiều mô hình, tuning và ensemble. 

## 2. Mục tiêu

- **EDA:** khám phá để hiểu rõ các vấn đề của bộ dữ liệu.
- **Optimize Preprocessing:** tìm các preprocessor tối ưu để giải quyết các vấn đề trên.
- **Ensemble:** kết hợp nhiều mô hình (weighted + stacking) để giảm sai số.

## 3. Các giai đoạn đã hoàn thành

### 3.1. Khám phá dữ liệu

- Khám phá dữ liệu
- Loại bỏ các feature có nhiều missing value (>50%)
- Feature Engineering
- Feature Selection

### 3.2. Tiền xử lý

Các preprocessor được lựa chọn tối ưu theo thứ tự:

1. Xử lý ngoại lai (Outlier Handling)
2. Xử lý dữ liệu thiếu (Imputation)
3. Biến đổi dữ liệu (Data transformation)
4. Chuẩn hóa dữ liệu (Scaling)
5. Mã hóa dữ liệu (Encoding)
6. Polynomial feature
7. Xử lý đa cộng tuyến (Multicollinearity Handling)

### 3.3. Mô hình hóa

Thử nghiệm ensemble model với 2 kỹ thuật Voting và Stacking:

- **Weighted averaging**: kết hợp Optuna + hill-climbing để tối ưu trọng số đóng góp của từng mô hình
- **Stacking**: thử nghiệm với nhiều meta-learners và fine-tuning learner tốt nhất

## 4.Điểm nổi bật kỹ thuật
- **Tiền xử lý tối ưu:** outlier handling, imputation, scaling, encoding, polynomial interactions.
- **Feature engineering:** tạo các biến area, age, quality score và flags (IsNew, HasGarage…).
- **Hybrid feature selection:** kết hợp các kỹ thuật LGBM, XGB, RF, Lasso, f_regression (giữ feature xuất hiện ≥3/5).
-  **Nhóm mô hình:** Linear, Tree-based, Special (CatBoost, RANSAC).
-  **Hyperparameter tuning:** Optuna + CV (KFold 5).
-  **Ensemble:**
    - Weighted averaging (Optuna + hill-climbing)
    - Stacking (nhiều meta-learners + fine-tuning learner tốt nhất)


## 5. Cấu trúc dự án
```
project
├── data/         # Raw & processed datasets
├── notebook/            # Jupyter notebooks
│   ├── Baseline.ipynb
│   ├── EDA.ipynb
│   ├── Preprocessing.ipynb
│   ├── Modeling.ipynb
│   └── Tuning.ipynb
│
├── results/      # Outputs &  visualizations
│   ├── experiments_results/
│   ├── {dataset}_weighted_{timestamp}/
│   └── {dataset}_stacking_{timestamp}/
│
└── src/          # Source code
│   ├── modeling.py
│   ├── preprocessing.py
│   ├── ...
│   
└── README.md
└── requirements.txt
```