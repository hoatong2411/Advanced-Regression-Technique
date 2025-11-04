# Ames Housing Price Prediction - Advanced Regression Technique

## Tổng quan
Dự án tối ưu hóa mô hình dự đoán giá nhà (Ames Housing) qua pipeline toàn diện: EDA, tiền xử lý, feature engineering, feature selection, huấn luyện nhiều mô hình, tuning và ensemble. Mục tiêu: cải thiện RMSE/R² trên tập kiểm thử để hỗ trợ quyết định bất động sản.

## Mục tiêu
- ***EDA:*** khám phá để hiểu rõ các vấn đề của bộ dữ liệu.
- ***Optimize Preprocessing:*** tìm các preprocessor tối ưu để giải quyết các vấn đề trên.
- ***Ensemble:*** kết hợp nhiều mô hình (weighted + stacking) để giảm sai số.

## Điểm nổi bật kỹ thuật
- ***Tiền xử lý tối ưu:*** outlier handling, imputation, scaling, encoding, polynomial interactions.
- ***Feature engineering:*** tạo các biến area, age, quality score và flags (IsNew, HasGarage…).
- ***Hybrid feature selection:*** kết hợp các kỹ thuật LGBM, XGB, RF, Lasso, f_regression (giữ feature xuất hiện ≥3/5).
-  ***Nhóm mô hình:*** Linear, Tree-based, Special (CatBoost, RANSAC).
-  ***Hyperparameter tuning:*** Optuna + CV (KFold 5).
-  ***Ensemble:***
    - Weighted averaging (Optuna + hill-climbing)
    - Stacking (nhiều meta-learners + fine-tuning learner tốt nhất)

## Cấu trúc dự án
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


## Cài đặt

Cài nhanh:
````bash
# Clone repository
git clone https://github.com/hoatong2411/Advanced-Regression-Technique/
cd Project

# Cài dependencies
pip install -r requirements.txt
>>>>>>> 589cf43 (First commit)
