# SVM 髖肩分離預測模型 (SVM Hip-Shoulder Separation Prediction Model)

## 模型描述 (Model Description)

此模型使用支持向量機 (Support Vector Machine, SVM) 回歸來預測棒球投手的最大髖肩分離角度及其發生時間。

**輸入特徵 (Input Features):**
- 最大肩外旋角度 (max_shoulder_external_rotation) - 單位：度 (degrees)
- 最大肩外旋發生時間 (time_max_shoulder_external_rotation) - 單位：秒 (seconds)

**輸出目標 (Output Targets):**
- 最大髖肩分離角度 (max_hip_shoulder_separation) - 單位：度 (degrees)
- 最大髖肩分離發生時間 (time_max_hip_shoulder_separation) - 單位：秒 (seconds)

## 模型性能 (Model Performance)

### 整體指標 (Overall Metrics)

| 指標 | 訓練集 | 測試集 |
|------|--------|--------|
| MSE | 20.04 | 18.49 |
| MAE | 2.55 | 2.50 |
| R² | 0.551 | 0.520 |

### 各輸出的詳細指標 (Per-Output Metrics)

#### 最大髖肩分離角度 (max_hip_shoulder_separation)
- **訓練集**: MSE = 40.09, MAE = 5.08°, R² = 0.107
- **測試集**: MSE = 36.97, MAE = 4.97°, R² = 0.045

#### 最大髖肩分離發生時間 (time_max_hip_shoulder_separation)
- **訓練集**: MSE = 0.0005, MAE = 0.017s, R² = 0.994
- **測試集**: MSE = 0.0006, MAE = 0.018s, R² = 0.994

## 最佳超參數 (Best Hyperparameters)

- **C**: 10
- **epsilon**: 0.2
- **gamma**: 0.01
- **kernel**: RBF (Radial Basis Function)

## 數據集資訊 (Dataset Information)

- **總樣本數**: 411
- **訓練集大小**: 328 (80%)
- **測試集大小**: 83 (20%)
- **資料來源**: `max_angles_extracted.csv`

## 文件說明 (Files)

- `train_svm_hip_shoulder_separation.py` - 訓練腳本
- `svm_hip_shoulder_separation_model.pkl` - 訓練好的 SVM 模型
- `scaler_X_hip_shoulder_separation.pkl` - 輸入特徵標準化器
- `scaler_y_hip_shoulder_separation.pkl` - 輸出目標標準化器
- `svm_hip_shoulder_separation_model_summary.json` - 模型摘要（JSON格式）

## 使用方法 (Usage)

```python
import joblib
import numpy as np

# 載入模型和標準化器
model = joblib.load('svm_hip_shoulder_separation_model.pkl')
scaler_X = joblib.load('scaler_X_hip_shoulder_separation.pkl')
scaler_y = joblib.load('scaler_y_hip_shoulder_separation.pkl')

# 準備輸入數據 (肩外旋角度, 肩外旋時間)
X_new = np.array([[170.0, 1.3]])  # 例如：170度，1.3秒

# 標準化輸入
X_new_scaled = scaler_X.transform(X_new)

# 預測
y_pred_scaled = model.predict(X_new_scaled)

# 反標準化輸出
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# 結果
hip_shoulder_separation = y_pred[0, 0]  # 髖肩分離角度
separation_time = y_pred[0, 1]           # 髖肩分離時間

print(f"預測的最大髖肩分離: {hip_shoulder_separation:.2f}°")
print(f"預測的發生時間: {separation_time:.4f}s")
```

## 模型分析與建議 (Analysis and Recommendations)

### 模型優勢
1. **時間預測極佳**: 時間預測的R²值高達0.994，表示模型能非常準確地預測髖肩分離發生的時間點
2. **整體性能良好**: 整體測試集R²為0.520，說明模型有一定的預測能力
3. **時間相關性強**: 肩外旋時間與髖肩分離時間有很強的相關性，這符合生物力學原理

### 模型限制
1. **角度預測較弱**: 髖肩分離角度的R²值較低（0.045），說明僅用肩外旋資訊難以準確預測角度值
2. **個體差異**: 髖肩分離角度可能受到投手個人技術、體型等多種因素影響

### 改進建議
1. **增加特徵**: 加入更多生物力學參數（如身高、體重、投球速度等）
2. **嘗試其他模型**: 考慮使用隨機森林或神經網絡等模型
3. **特徵工程**: 創建交互特徵或多項式特徵

## 應用場景 (Use Cases)

- **投球動作分析**: 評估投手的動作協調性
- **傷害預防**: 監測異常的髖肩分離模式
- **訓練反饋**: 提供即時的動作時序反饋
- **運動表現優化**: 幫助投手優化投球動作

## 注意事項 (Notes)

- 此模型基於 OpenBiomechanics 項目的棒球投球數據訓練
- 預測結果僅供參考，實際應用時需考慮個體差異
- 時間預測非常可靠（R²=0.994），可用於動作時序分析
- 角度預測需謹慎使用，建議結合其他評估方法
