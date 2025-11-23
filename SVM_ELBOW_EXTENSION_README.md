# SVM 肘外展預測模型 (SVM Elbow Extension Prediction Model)

## 模型描述 (Model Description)

此模型使用支持向量機 (Support Vector Machine, SVM) 回歸來預測棒球投手的最大肘外展角度及其發生時間。

**輸入特徵 (Input Features):**
- 最大肩外旋角度 (max_shoulder_external_rotation) - 單位：度 (degrees)
- 最大肩外旋發生時間 (time_max_shoulder_external_rotation) - 單位：秒 (seconds)

**輸出目標 (Output Targets):**
- 最大肘外展角度 (max_elbow_extension) - 單位：度 (degrees)
- 最大肘外展發生時間 (time_max_elbow_extension) - 單位：秒 (seconds)

## 模型性能 (Model Performance)

### 整體指標 (Overall Metrics)

| 指標 | 訓練集 | 測試集 |
|------|--------|--------|
| MSE | 28.50 | 35.21 |
| MAE | 2.99 | 3.35 |
| R² | 0.350 | 0.184 |

### 各輸出的詳細指標 (Per-Output Metrics)

#### 最大肘外展角度 (max_elbow_extension)
- **訓練集**: MSE = 56.95, MAE = 5.80°, R² = 0.031
- **測試集**: MSE = 70.37, MAE = 6.49°, R² = -0.178

#### 最大肘外展發生時間 (time_max_elbow_extension)
- **訓練集**: MSE = 0.048, MAE = 0.188s, R² = 0.670
- **測試集**: MSE = 0.061, MAE = 0.215s, R² = 0.546

## 最佳超參數 (Best Hyperparameters)

- **C**: 100
- **epsilon**: 0.2
- **gamma**: 0.01
- **kernel**: RBF (Radial Basis Function)

## 數據集資訊 (Dataset Information)

- **總樣本數**: 411
- **訓練集大小**: 328 (80%)
- **測試集大小**: 83 (20%)
- **資料來源**: `max_angles_extracted.csv`

## 文件說明 (Files)

- `train_svm_elbow_extension.py` - 訓練腳本
- `svm_elbow_extension_model.pkl` - 訓練好的 SVM 模型
- `scaler_X_elbow_extension.pkl` - 輸入特徵標準化器
- `scaler_y_elbow_extension.pkl` - 輸出目標標準化器
- `svm_elbow_extension_model_summary.json` - 模型摘要（JSON格式）

## 使用方法 (Usage)

```python
import joblib
import numpy as np

# 載入模型和標準化器
model = joblib.load('svm_elbow_extension_model.pkl')
scaler_X = joblib.load('scaler_X_elbow_extension.pkl')
scaler_y = joblib.load('scaler_y_elbow_extension.pkl')

# 準備輸入數據 (肩外旋角度, 肩外旋時間)
X_new = np.array([[170.0, 1.3]])  # 例如：170度，1.3秒

# 標準化輸入
X_new_scaled = scaler_X.transform(X_new)

# 預測
y_pred_scaled = model.predict(X_new_scaled)

# 反標準化輸出
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# 結果
elbow_extension = y_pred[0, 0]  # 肘外展角度
elbow_time = y_pred[0, 1]        # 肘外展時間

print(f"預測的最大肘外展: {elbow_extension:.2f}°")
print(f"預測的發生時間: {elbow_time:.4f}s")
```

## 模型限制與建議 (Limitations and Recommendations)

1. **肘外展角度預測**: R²值較低（測試集為-0.178），表示模型對肘外展角度的預測能力有限。這可能是因為：
   - 肘外展角度受多個因素影響，僅用肩外旋資訊不足以準確預測
   - 數據中可能存在較大的個體差異

2. **時間預測**: 時間預測的R²值較好（測試集為0.546），說明肩外旋時間與肘外展時間有較強的相關性。

3. **改進建議**:
   - 增加更多輸入特徵（如髖肩分離、投球速度等）
   - 嘗試其他機器學習模型（如隨機森林、神經網絡）
   - 收集更多訓練數據

## 注意事項 (Notes)

- 此模型基於 OpenBiomechanics 項目的棒球投球數據訓練
- 預測結果僅供參考，實際應用時需考慮個體差異
- 模型使用 StandardScaler 進行特徵標準化，預測時必須使用相同的標準化器
