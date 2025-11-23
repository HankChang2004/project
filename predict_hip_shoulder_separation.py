"""
使用訓練好的SVM模型進行髖肩分離預測
Use trained SVM model to predict hip-shoulder separation
"""

import joblib
import numpy as np

# 載入模型和標準化器
print("載入模型...")
model = joblib.load('svm_hip_shoulder_separation_model.pkl')
scaler_X = joblib.load('scaler_X_hip_shoulder_separation.pkl')
scaler_y = joblib.load('scaler_y_hip_shoulder_separation.pkl')
print("模型載入完成！\n")

# 示例預測
examples = [
    [170.0, 1.3],   # 肩外旋170度，時間1.3秒
    [165.0, 1.5],   # 肩外旋165度，時間1.5秒
    [180.0, 1.2],   # 肩外旋180度，時間1.2秒
    [175.0, 1.4],   # 肩外旋175度，時間1.4秒
    [160.0, 1.6],   # 肩外旋160度，時間1.6秒
]

print("=" * 80)
print("SVM 髖肩分離預測模型 - 示例預測")
print("=" * 80)
print()

for i, (shoulder_rotation, shoulder_time) in enumerate(examples, 1):
    # 準備輸入數據
    X_new = np.array([[shoulder_rotation, shoulder_time]])
    
    # 標準化輸入
    X_new_scaled = scaler_X.transform(X_new)
    
    # 預測
    y_pred_scaled = model.predict(X_new_scaled)
    
    # 反標準化輸出
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # 提取結果
    hip_shoulder_separation = y_pred[0, 0]
    separation_time = y_pred[0, 1]
    
    print(f"示例 {i}:")
    print(f"  輸入 - 最大肩外旋: {shoulder_rotation:.1f}°, 時間: {shoulder_time:.2f}s")
    print(f"  預測 - 最大髖肩分離: {hip_shoulder_separation:.2f}°, 時間: {separation_time:.4f}s")
    print()

print("=" * 80)
print()

# 互動式預測
print("互動式預測（輸入 'q' 結束）")
print("-" * 80)

while True:
    try:
        user_input = input("\n請輸入最大肩外旋角度（度）（或 'q' 結束）: ").strip()
        if user_input.lower() == 'q':
            print("結束預測。")
            break
        
        shoulder_rotation = float(user_input)
        
        shoulder_time = float(input("請輸入最大肩外旋時間（秒）: ").strip())
        
        # 準備輸入數據
        X_new = np.array([[shoulder_rotation, shoulder_time]])
        
        # 標準化輸入
        X_new_scaled = scaler_X.transform(X_new)
        
        # 預測
        y_pred_scaled = model.predict(X_new_scaled)
        
        # 反標準化輸出
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        
        # 提取結果
        hip_shoulder_separation = y_pred[0, 0]
        separation_time = y_pred[0, 1]
        
        print(f"\n預測結果:")
        print(f"  最大髖肩分離: {hip_shoulder_separation:.2f}°")
        print(f"  發生時間: {separation_time:.4f}s")
        
    except ValueError:
        print("錯誤：請輸入有效的數字。")
    except KeyboardInterrupt:
        print("\n\n結束預測。")
        break
    except Exception as e:
        print(f"錯誤: {e}")
