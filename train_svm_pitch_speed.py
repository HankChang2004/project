import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib

# 讀取數據
print("Loading data...")
angles_df = pd.read_csv(r'c:\git\openbiomechanics\baseball_pitching\max_angles_extracted.csv')
metadata_df = pd.read_csv(r'c:\git\openbiomechanics\baseball_pitching\data\metadata.csv')

print(f"Angles data shape: {angles_df.shape}")
print(f"Metadata shape: {metadata_df.shape}")

# 合併數據，使用 session_pitch 作為鍵
print("\nMerging data...")
merged_df = pd.merge(angles_df, metadata_df[['session_pitch', 'pitch_speed_mph']], 
                     on='session_pitch', how='inner')

print(f"Merged data shape: {merged_df.shape}")
print(f"\nFirst few rows:")
print(merged_df.head())

# 檢查是否有缺失值
print(f"\nMissing values:\n{merged_df.isnull().sum()}")

# 準備特徵和目標變量
feature_columns = [
    'max_shoulder_external_rotation',
    'time_max_shoulder_external_rotation',
    'max_elbow_extension',
    'time_max_elbow_extension',
    'max_hip_shoulder_separation',
    'time_max_hip_shoulder_separation'
]

X = merged_df[feature_columns].values
y = merged_df['pitch_speed_mph'].values

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"\nPitch speed statistics:")
print(f"  Mean: {y.mean():.2f} mph")
print(f"  Std: {y.std():.2f} mph")
print(f"  Min: {y.min():.2f} mph")
print(f"  Max: {y.max():.2f} mph")

# 分割訓練集和測試集
print("\nSplitting data into train and test sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# 標準化特徵
print("\nStandardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 訓練 SVM 回歸模型
print("\n" + "="*60)
print("Training SVM model...")
print("="*60)

# 使用網格搜索找到最佳超參數
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'epsilon': [0.01, 0.1, 0.5, 1]
}

print("\nPerforming grid search with cross-validation...")
svm_model = SVR(kernel='rbf')
grid_search = GridSearchCV(
    svm_model, 
    param_grid, 
    cv=5, 
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score (MSE): {-grid_search.best_score_:.4f}")

# 使用最佳模型
best_model = grid_search.best_estimator_

# 在訓練集上評估
print("\n" + "="*60)
print("Evaluating on training set...")
print("="*60)
y_train_pred = best_model.predict(X_train_scaled)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print(f"Training MSE: {train_mse:.4f}")
print(f"Training RMSE: {train_rmse:.4f} mph")
print(f"Training MAE: {train_mae:.4f} mph")
print(f"Training R²: {train_r2:.4f}")

# 在測試集上評估
print("\n" + "="*60)
print("Evaluating on test set...")
print("="*60)
y_test_pred = best_model.predict(X_test_scaled)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Test MSE: {test_mse:.4f}")
print(f"Test RMSE: {test_rmse:.4f} mph")
print(f"Test MAE: {test_mae:.4f} mph")
print(f"Test R²: {test_r2:.4f}")

# 交叉驗證
print("\n" + "="*60)
print("Performing 5-fold cross-validation...")
print("="*60)
cv_scores = cross_val_score(
    best_model, 
    X_train_scaled, 
    y_train, 
    cv=5, 
    scoring='neg_mean_squared_error'
)
cv_rmse = np.sqrt(-cv_scores)
print(f"CV RMSE scores: {cv_rmse}")
print(f"CV RMSE mean: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f} mph")

# 保存模型和標準化器
print("\n" + "="*60)
print("Saving model and scaler...")
print("="*60)
joblib.dump(best_model, r'c:\git\openbiomechanics\baseball_pitching\svm_pitch_speed_model.pkl')
joblib.dump(scaler, r'c:\git\openbiomechanics\baseball_pitching\scaler.pkl')
print("Model saved to: svm_pitch_speed_model.pkl")
print("Scaler saved to: scaler.pkl")

# 創建可視化
print("\nCreating visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 訓練集預測 vs 實際
axes[0, 0].scatter(y_train, y_train_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
                'r--', lw=2, label='Perfect prediction')
axes[0, 0].set_xlabel('Actual Pitch Speed (mph)', fontsize=12)
axes[0, 0].set_ylabel('Predicted Pitch Speed (mph)', fontsize=12)
axes[0, 0].set_title(f'Training Set\nR² = {train_r2:.4f}, RMSE = {train_rmse:.4f} mph', 
                     fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. 測試集預測 vs 實際
axes[0, 1].scatter(y_test, y_test_pred, alpha=0.5, edgecolors='k', linewidth=0.5, color='orange')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect prediction')
axes[0, 1].set_xlabel('Actual Pitch Speed (mph)', fontsize=12)
axes[0, 1].set_ylabel('Predicted Pitch Speed (mph)', fontsize=12)
axes[0, 1].set_title(f'Test Set\nR² = {test_r2:.4f}, RMSE = {test_rmse:.4f} mph', 
                     fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. 訓練集殘差圖
train_residuals = y_train - y_train_pred
axes[1, 0].scatter(y_train_pred, train_residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Predicted Pitch Speed (mph)', fontsize=12)
axes[1, 0].set_ylabel('Residuals (mph)', fontsize=12)
axes[1, 0].set_title('Training Set Residuals', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 4. 測試集殘差圖
test_residuals = y_test - y_test_pred
axes[1, 1].scatter(y_test_pred, test_residuals, alpha=0.5, edgecolors='k', 
                   linewidth=0.5, color='orange')
axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Predicted Pitch Speed (mph)', fontsize=12)
axes[1, 1].set_ylabel('Residuals (mph)', fontsize=12)
axes[1, 1].set_title('Test Set Residuals', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'c:\git\openbiomechanics\baseball_pitching\svm_results.png', dpi=300, bbox_inches='tight')
print("Visualization saved to: svm_results.png")

# 特徵重要性分析（使用排列重要性）
print("\n" + "="*60)
print("Feature importance analysis...")
print("="*60)
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(
    best_model, X_test_scaled, y_test, 
    n_repeats=10, random_state=42, n_jobs=-1
)

feature_importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': perm_importance.importances_mean,
    'Std': perm_importance.importances_std
}).sort_values('Importance', ascending=False)

print("\nFeature Importance (Permutation):")
print(feature_importance_df.to_string(index=False))

# 繪製特徵重要性圖
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Importance for Pitch Speed Prediction', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(r'c:\git\openbiomechanics\baseball_pitching\feature_importance.png', dpi=300, bbox_inches='tight')
print("\nFeature importance plot saved to: feature_importance.png")

# 保存結果摘要
summary = {
    'model_type': 'Support Vector Regression (RBF kernel)',
    'best_params': grid_search.best_params_,
    'train_size': X_train.shape[0],
    'test_size': X_test.shape[0],
    'train_rmse': train_rmse,
    'train_mae': train_mae,
    'train_r2': train_r2,
    'test_rmse': test_rmse,
    'test_mae': test_mae,
    'test_r2': test_r2,
    'cv_rmse_mean': cv_rmse.mean(),
    'cv_rmse_std': cv_rmse.std(),
    'features': feature_columns
}

import json
with open(r'c:\git\openbiomechanics\baseball_pitching\svm_model_summary.json', 'w') as f:
    json.dump(summary, f, indent=4)

print("\n" + "="*60)
print("Model summary saved to: svm_model_summary.json")
print("="*60)
print("\nTraining completed successfully!")
