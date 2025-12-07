# enhance_features.py
import pandas as pd
import numpy as np
from datetime import datetime
import os

def enhance_streetlight_features():
    """
    Tambah features baru untuk membuat data lebih kompleks dan realistis
    """
    print("="*60)
    print("ENHANCE STREETLIGHT FEATURES")
    print("="*60)
    
    # Load cleaned data
    input_file = r"C:\Users\LENOVO\OneDrive\Documents\SIC\Belajar\streetlight\MachineLearningSL\DataCleaning\streetlight_data_cleaned.csv"
    if not os.path.exists(input_file):
        print(f"❌ File tidak ditemukan: {input_file}")
        return
    
    df = pd.read_csv(input_file)
    print(f"📥 Loaded {len(df)} rows from {input_file}")
    
    # 1. Parse timestamp untuk time features
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 2. Add basic time features
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # 3. Circular encoding untuk hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # 4. Light intensity categories
    df['light_category'] = pd.cut(df['light_intensity'],
                                 bins=[0, 30, 70, 100],
                                 labels=['Dark', 'Medium', 'Bright'])
    
    # 5. Rolling statistics
    df['intensity_rolling_mean'] = df['light_intensity'].rolling(3, min_periods=1).mean()
    df['intensity_rolling_std'] = df['light_intensity'].rolling(5, min_periods=1).std().fillna(0)
    
    # 6. Voltage patterns
    df['voltage_change'] = (df['voltage'] != df['voltage'].shift(1)).astype(int)
    df['lamp_on_duration'] = 0
    
    # Calculate lamp on duration
    duration = 0
    for i in range(len(df)):
        if df.iloc[i]['voltage'] > 0:
            duration += 1
        else:
            duration = 0
        df.at[i, 'lamp_on_duration'] = duration
    
    # 7. Interaction features
    df['hour_light_interaction'] = df['hour'] * df['light_intensity'] / 100
    df['energy_estimate'] = df['voltage'] * (100 - df['light_intensity']) / 10000
    
    # 8. Add small noise untuk realism (10% samples)
    np.random.seed(42)
    mask = np.random.random(len(df)) < 0.1
    df.loc[mask, 'light_intensity'] = df.loc[mask, 'light_intensity'] + np.random.normal(0, 5, mask.sum())
    df['light_intensity'] = df['light_intensity'].clip(0, 100)
    
    # 9. Save enhanced data
    output_file = "streetlight_enhanced.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Enhanced data saved: {output_file}")
    print(f"📊 Original features: 3")
    print(f"📊 Enhanced features: {len(df.columns)}")
    print(f"\n🎯 New features added:")
    new_features = [col for col in df.columns if col not in ['timestamp', 'light_intensity', 'voltage', 'class']]
    for i, feat in enumerate(new_features, 1):
        print(f"   {i:2d}. {feat}")
    
    return df

if __name__ == "__main__":
    enhance_streetlight_features()