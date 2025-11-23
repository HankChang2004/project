"""
Train SVM model to predict max hip-shoulder separation and time from max shoulder external rotation and time
訓練SVM模型，輸入是最大肩外旋和時間，輸出是最大髖肩分離和時間
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
import joblib
import json

# Load data
df = pd.read_csv('max_angles_extracted.csv')

# Define features (input) and targets (output)
# Input: max_shoulder_external_rotation, time_max_shoulder_external_rotation
# Output: max_hip_shoulder_separation, time_max_hip_shoulder_separation
X = df[['max_shoulder_external_rotation', 'time_max_shoulder_external_rotation']].values
y = df[['max_hip_shoulder_separation', 'time_max_hip_shoulder_separation']].values

print(f"數據集大小: {len(df)} 個樣本")
print(f"輸入特徵形狀: {X.shape}")
print(f"輸出目標形狀: {y.shape}")
print()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"訓練集大小: {len(X_train)}")
print(f"測試集大小: {len(X_test)}")
print()

# Standardize features
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)

# Train SVM with hyperparameter tuning
print("開始訓練SVM模型...")
print("正在進行超參數調優...")

# Define parameter grid for GridSearch
param_grid = {
    'estimator__C': [0.1, 1, 10, 100],
    'estimator__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'estimator__epsilon': [0.01, 0.1, 0.2]
}

# Create base SVR model
base_svr = SVR(kernel='rbf')

# Use MultiOutputRegressor for multiple outputs
multi_svr = MultiOutputRegressor(base_svr)

# Grid search with cross-validation
grid_search = GridSearchCV(
    multi_svr,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train_scaled)

print(f"\n最佳參數: {grid_search.best_params_}")
print()

# Use best model
best_model = grid_search.best_estimator_

# Make predictions
y_train_pred_scaled = best_model.predict(X_train_scaled)
y_test_pred_scaled = best_model.predict(X_test_scaled)

# Inverse transform predictions back to original scale
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

# Calculate metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Calculate per-output metrics
output_names = ['max_hip_shoulder_separation', 'time_max_hip_shoulder_separation']
print("=" * 60)
print("模型評估結果")
print("=" * 60)
print(f"\n整體指標:")
print(f"訓練集 MSE: {train_mse:.4f}")
print(f"測試集 MSE: {test_mse:.4f}")
print(f"訓練集 MAE: {train_mae:.4f}")
print(f"測試集 MAE: {test_mae:.4f}")
print(f"訓練集 R²: {train_r2:.4f}")
print(f"測試集 R²: {test_r2:.4f}")

print(f"\n各輸出的詳細指標:")
for i, name in enumerate(output_names):
    train_mse_i = mean_squared_error(y_train[:, i], y_train_pred[:, i])
    test_mse_i = mean_squared_error(y_test[:, i], y_test_pred[:, i])
    train_mae_i = mean_absolute_error(y_train[:, i], y_train_pred[:, i])
    test_mae_i = mean_absolute_error(y_test[:, i], y_test_pred[:, i])
    train_r2_i = r2_score(y_train[:, i], y_train_pred[:, i])
    test_r2_i = r2_score(y_test[:, i], y_test_pred[:, i])
    
    print(f"\n{name}:")
    print(f"  訓練集 - MSE: {train_mse_i:.4f}, MAE: {train_mae_i:.4f}, R²: {train_r2_i:.4f}")
    print(f"  測試集 - MSE: {test_mse_i:.4f}, MAE: {test_mae_i:.4f}, R²: {test_r2_i:.4f}")

# Save model and scalers
print("\n正在保存模型...")
joblib.dump(best_model, 'svm_hip_shoulder_separation_model.pkl')
joblib.dump(scaler_X, 'scaler_X_hip_shoulder_separation.pkl')
joblib.dump(scaler_y, 'scaler_y_hip_shoulder_separation.pkl')

# Save model summary
model_summary = {
    'model_type': 'SVM (Support Vector Regression)',
    'input_features': ['max_shoulder_external_rotation', 'time_max_shoulder_external_rotation'],
    'output_targets': output_names,
    'best_params': grid_search.best_params_,
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'metrics': {
        'train_mse': float(train_mse),
        'test_mse': float(test_mse),
        'train_mae': float(train_mae),
        'test_mae': float(test_mae),
        'train_r2': float(train_r2),
        'test_r2': float(test_r2)
    },
    'per_output_metrics': {
        output_names[i]: {
            'train_mse': float(mean_squared_error(y_train[:, i], y_train_pred[:, i])),
            'test_mse': float(mean_squared_error(y_test[:, i], y_test_pred[:, i])),
            'train_mae': float(mean_absolute_error(y_train[:, i], y_train_pred[:, i])),
            'test_mae': float(mean_absolute_error(y_test[:, i], y_test_pred[:, i])),
            'train_r2': float(r2_score(y_train[:, i], y_train_pred[:, i])),
            'test_r2': float(r2_score(y_test[:, i], y_test_pred[:, i]))
        }
        for i in range(len(output_names))
    }
}

with open('svm_hip_shoulder_separation_model_summary.json', 'w', encoding='utf-8') as f:
    json.dump(model_summary, f, indent=2, ensure_ascii=False)

print("模型已保存:")
print("  - svm_hip_shoulder_separation_model.pkl")
print("  - scaler_X_hip_shoulder_separation.pkl")
print("  - scaler_y_hip_shoulder_separation.pkl")
print("  - svm_hip_shoulder_separation_model_summary.json")

# Show some example predictions
print("\n" + "=" * 60)
print("示例預測 (前5個測試樣本)")
print("=" * 60)
print(f"{'輸入':<50} | {'實際輸出':<35} | {'預測輸出':<35}")
print(f"{'(肩外旋, 時間)':<50} | {'(髖肩分離, 時間)':<35} | {'(髖肩分離, 時間)':<35}")
print("-" * 120)
for i in range(min(5, len(X_test))):
    input_str = f"({X_test[i, 0]:.2f}°, {X_test[i, 1]:.4f}s)"
    actual_str = f"({y_test[i, 0]:.2f}°, {y_test[i, 1]:.4f}s)"
    pred_str = f"({y_test_pred[i, 0]:.2f}°, {y_test_pred[i, 1]:.4f}s)"
    print(f"{input_str:<50} | {actual_str:<35} | {pred_str:<35}")

print("\n訓練完成!")
