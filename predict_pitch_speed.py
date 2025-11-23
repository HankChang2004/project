import pandas as pd
import numpy as np
import joblib

# 載入訓練好的模型和標準化器
print("Loading trained SVM model and scaler...")
model = joblib.load(r'c:\git\openbiomechanics\baseball_pitching\svm_pitch_speed_model.pkl')
scaler = joblib.load(r'c:\git\openbiomechanics\baseball_pitching\scaler.pkl')

print("Model loaded successfully!")
print(f"Model type: {type(model).__name__}")
print(f"Model parameters: {model.get_params()}")

# 載入測試數據
print("\n" + "="*60)
print("Loading test data...")
print("="*60)
angles_df = pd.read_csv(r'c:\git\openbiomechanics\baseball_pitching\max_angles_extracted.csv')
metadata_df = pd.read_csv(r'c:\git\openbiomechanics\baseball_pitching\data\metadata.csv')

# 合併數據
merged_df = pd.merge(angles_df, metadata_df[['session_pitch', 'pitch_speed_mph']], 
                     on='session_pitch', how='inner')

# 準備特徵
feature_columns = [
    'max_shoulder_external_rotation',
    'time_max_shoulder_external_rotation',
    'max_elbow_extension',
    'time_max_elbow_extension',
    'max_hip_shoulder_separation',
    'time_max_hip_shoulder_separation'
]

# 隨機選擇5個樣本進行預測演示
print("\nDemonstrating predictions on 5 random samples:")
print("="*60)
sample_indices = np.random.choice(merged_df.index, 5, replace=False)

for idx in sample_indices:
    sample = merged_df.loc[idx]
    session_pitch = sample['session_pitch']
    actual_speed = sample['pitch_speed_mph']
    
    # 提取特徵
    features = sample[feature_columns].values.reshape(1, -1)
    
    # 標準化
    features_scaled = scaler.transform(features)
    
    # 預測
    predicted_speed = model.predict(features_scaled)[0]
    
    # 計算誤差
    error = predicted_speed - actual_speed
    
    print(f"\nSession Pitch: {session_pitch}")
    print(f"  Max Shoulder External Rotation: {sample['max_shoulder_external_rotation']:.2f}°")
    print(f"  Max Elbow Extension: {sample['max_elbow_extension']:.2f}°")
    print(f"  Max Hip-Shoulder Separation: {sample['max_hip_shoulder_separation']:.2f}°")
    print(f"  Actual Speed: {actual_speed:.1f} mph")
    print(f"  Predicted Speed: {predicted_speed:.1f} mph")
    print(f"  Error: {error:.1f} mph ({abs(error)/actual_speed*100:.1f}%)")

# 演示如何對新數據進行預測
print("\n" + "="*60)
print("Example: Predicting pitch speed for new biomechanics data")
print("="*60)

# 創建一個假設的新數據點（使用平均值）
new_data = {
    'max_shoulder_external_rotation': 170.0,  # 度
    'time_max_shoulder_external_rotation': 1.3,  # 秒
    'max_elbow_extension': 25.0,  # 度
    'time_max_elbow_extension': 1.1,  # 秒
    'max_hip_shoulder_separation': 35.0,  # 度
    'time_max_hip_shoulder_separation': 1.2  # 秒
}

print("\nNew biomechanics data:")
for key, value in new_data.items():
    print(f"  {key}: {value}")

# 轉換為數組並進行預測
new_features = np.array([[
    new_data['max_shoulder_external_rotation'],
    new_data['time_max_shoulder_external_rotation'],
    new_data['max_elbow_extension'],
    new_data['time_max_elbow_extension'],
    new_data['max_hip_shoulder_separation'],
    new_data['time_max_hip_shoulder_separation']
]])

new_features_scaled = scaler.transform(new_features)
predicted_speed = model.predict(new_features_scaled)[0]

print(f"\nPredicted pitch speed: {predicted_speed:.1f} mph")

print("\n" + "="*60)
print("Prediction completed!")
print("="*60)
