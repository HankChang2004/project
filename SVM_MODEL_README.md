# SVM Pitch Speed Prediction Model

## 概述

這個專案使用支持向量機 (Support Vector Machine, SVM) 來預測棒球投手的球速。模型使用從生物力學數據中提取的最大角度和時間特徵作為輸入。

## 文件說明

### 數據文件
- `max_angles_extracted.csv` - 從 `joint_angles.csv` 提取的特徵數據（411筆投球記錄）
  - 最大肩外旋角度和時間
  - 最大肘外展角度和時間
  - 最大肩髖分離角度和時間

### 腳本文件
- `extract_max_angles.py` - 從原始生物力學數據中提取特徵的腳本
- `train_svm_pitch_speed.py` - 訓練 SVM 模型的主要腳本
- `predict_pitch_speed.py` - 使用訓練好的模型進行預測的演示腳本

### 模型文件
- `svm_pitch_speed_model.pkl` - 訓練好的 SVM 模型
- `scaler.pkl` - 特徵標準化器（StandardScaler）
- `svm_model_summary.json` - 模型性能摘要

### 結果文件
- `svm_results.png` - 模型預測結果可視化
  - 訓練集和測試集的預測 vs 實際值散點圖
  - 殘差圖
- `feature_importance.png` - 特徵重要性排名

## 模型性能

### 訓練結果
- **訓練集 (328 樣本)**
  - RMSE: 2.06 mph
  - MAE: 1.47 mph
  - R²: 0.810

- **測試集 (83 樣本)**
  - RMSE: 2.97 mph
  - MAE: 2.35 mph
  - R²: 0.603

- **5-fold 交叉驗證**
  - RMSE: 3.48 ± 0.30 mph

### 最佳超參數
- Kernel: RBF (Radial Basis Function)
- C: 100
- Epsilon: 1
- Gamma: scale

## 特徵重要性

根據排列重要性分析，特徵對預測的重要程度排序：

1. `time_max_shoulder_external_rotation` (1.51) - 最大肩外旋時間
2. `time_max_elbow_extension` (1.06) - 最大肘外展時間
3. `time_max_hip_shoulder_separation` (0.82) - 最大肩髖分離時間
4. `max_elbow_extension` (0.44) - 最大肘外展角度
5. `max_hip_shoulder_separation` (0.43) - 最大肩髖分離角度
6. `max_shoulder_external_rotation` (0.37) - 最大肩外旋角度

**關鍵發現：** 時間特徵（動作發生的時間點）比角度特徵（動作的幅度）對球速預測更重要。

## 使用方法

### 1. 訓練新模型

```bash
python train_svm_pitch_speed.py
```

這將：
- 載入特徵數據和球速目標
- 執行網格搜索找到最佳超參數
- 訓練 SVM 模型
- 評估模型性能
- 保存模型和可視化結果

### 2. 使用訓練好的模型進行預測

```python
import joblib
import numpy as np

# 載入模型和標準化器
model = joblib.load('svm_pitch_speed_model.pkl')
scaler = joblib.load('scaler.pkl')

# 準備新的生物力學數據
new_data = np.array([[
    170.0,  # max_shoulder_external_rotation (度)
    1.3,    # time_max_shoulder_external_rotation (秒)
    25.0,   # max_elbow_extension (度)
    1.1,    # time_max_elbow_extension (秒)
    35.0,   # max_hip_shoulder_separation (度)
    1.2     # time_max_hip_shoulder_separation (秒)
]])

# 標準化特徵
new_data_scaled = scaler.transform(new_data)

# 預測球速
predicted_speed = model.predict(new_data_scaled)[0]
print(f"預測球速: {predicted_speed:.1f} mph")
```

### 3. 運行預測演示

```bash
python predict_pitch_speed.py
```

這將展示如何：
- 載入訓練好的模型
- 對隨機樣本進行預測
- 比較預測值和實際值
- 對新數據進行預測

## 依賴套件

```
pandas
numpy
scikit-learn
matplotlib
joblib
```

安裝方法：
```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

## 數據來源

數據來自 Open Biomechanics Project (OBP) 的棒球投球數據集：
- 原始數據位置: `data/full_sig/joint_angles/joint_angles.csv`
- 元數據位置: `data/metadata.csv`
- 411 筆投球記錄
- 球速範圍: 69.5 - 94.4 mph
- 平均球速: 84.7 ± 4.7 mph

## 注意事項

1. **模型限制**：此模型僅使用 6 個生物力學特徵，實際投球速度還受到許多其他因素影響（如體能、技術、天氣等）

2. **預測範圍**：模型在訓練數據範圍內（約 70-95 mph）表現最佳，對超出此範圍的預測可能不準確

3. **特徵工程**：時間特徵顯示出較高的重要性，未來可以考慮加入更多時序相關特徵

4. **數據平衡**：訓練數據主要來自大學和職業球員，在其他水平球員上的表現可能不同

## 未來改進方向

1. 加入更多生物力學特徵（如速度、力矩、能量流等）
2. 嘗試其他機器學習模型（Random Forest, Gradient Boosting, Neural Networks）
3. 考慮投手的身體測量數據（身高、體重、臂展等）
4. 使用時間序列特徵或深度學習方法處理完整的運動軌跡
5. 進行更細緻的特徵工程和選擇

## 作者

Created for biomechanics analysis of baseball pitching using the Open Biomechanics Project dataset.
