import pandas as pd
import numpy as np

# Read the joint_angles.csv file
print("Reading joint_angles.csv...")
df = pd.read_csv(r'c:\git\openbiomechanics\baseball_pitching\data\full_sig\joint_angles\joint_angles.csv')

print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()[:20]}...")  # Show first 20 columns

# Based on README documentation:
# Shoulder: C3 ("_Z") = External (+)/Internal (-) Rotation
# Elbow: C1 ("_X") = Flexion (+)/Extension (-) 
# Torso-Pelvis: C3 ("_Z") = Hip-Shoulder Separation (+)/Closing (-)

# Column names in the CSV:
# 1. Shoulder external rotation (throwing arm) - "shoulder_angle_z"
# 2. Elbow extension (throwing arm) - "elbow_angle_x" (extension is negative, so we find minimum)
# 3. Torso-Pelvis separation - "torso_pelvis_angle_z"

print("\nColumn names confirmed:")
print(f"  shoulder_angle_z exists: {'shoulder_angle_z' in df.columns}")
print(f"  elbow_angle_x exists: {'elbow_angle_x' in df.columns}")
print(f"  torso_pelvis_angle_z exists: {'torso_pelvis_angle_z' in df.columns}")

# Initialize results list
results = []

# Group by session_pitch
print("\nProcessing each session_pitch...")
grouped = df.groupby('session_pitch')

for session_pitch, group in grouped:
    result = {'session_pitch': session_pitch}
    
    # 1. Find maximum shoulder external rotation (shoulder_angle_z, positive is external)
    if 'shoulder_angle_z' in df.columns:
        max_idx = group['shoulder_angle_z'].idxmax()
        result['max_shoulder_external_rotation'] = group.loc[max_idx, 'shoulder_angle_z']
        result['time_max_shoulder_external_rotation'] = group.loc[max_idx, 'time']
    
    # 2. Find maximum elbow extension (elbow_angle_x, extension is negative, so find minimum)
    if 'elbow_angle_x' in df.columns:
        min_idx = group['elbow_angle_x'].idxmin()
        result['max_elbow_extension'] = group.loc[min_idx, 'elbow_angle_x']
        result['time_max_elbow_extension'] = group.loc[min_idx, 'time']
    
    # 3. Find maximum hip-shoulder separation (torso_pelvis_angle_z, positive is separation)
    if 'torso_pelvis_angle_z' in df.columns:
        max_idx = group['torso_pelvis_angle_z'].idxmax()
        result['max_hip_shoulder_separation'] = group.loc[max_idx, 'torso_pelvis_angle_z']
        result['time_max_hip_shoulder_separation'] = group.loc[max_idx, 'time']
    
    results.append(result)

# Create results dataframe
results_df = pd.DataFrame(results)

print(f"\nProcessed {len(results_df)} unique session_pitches")
print("\nFirst few rows of results:")
print(results_df.head())

# Save to CSV
output_file = r'c:\git\openbiomechanics\baseball_pitching\max_angles_extracted.csv'
results_df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")
print(f"\nColumns in output: {results_df.columns.tolist()}")
