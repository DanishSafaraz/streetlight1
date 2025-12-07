# create_proper_split.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def create_proper_split():
    """
    Create proper train/test split without data leakage
    """
    print("="*60)
    print("CREATE PROPER TRAIN/TEST SPLIT")
    print("="*60)
    
    # Load labeled data
    input_file = r"C:\Users\LENOVO\OneDrive\Documents\SIC\Belajar\streetlight\MachineLearningSL\streetlight_labeled.csv"
    if not os.path.exists(input_file):
        input_file = r"C:\Users\LENOVO\OneDrive\Documents\SIC\Belajar\streetlight\MachineLearningSL\streetlight_data_cleaned.csv"
        print(f"⚠ Using {input_file} (labeled not found)")
    
    df = pd.read_csv(input_file)
    print(f"📥 Loaded {len(df)} rows from {input_file}")
    print(f"Columns: {list(df.columns)}")
    
    # 1. CEK DAN FIX LABEL COLUMN
    print("\n🔍 1. Checking label column...")
    
    # Cari kolom label (bisa 'label', 'class', atau 'label_numeric')
    label_column = None
    for col in ['label', 'class', 'label_numeric']:
        if col in df.columns:
            label_column = col
            print(f"   Found label column: {label_column}")
            break
    
    if not label_column:
        # Buat label baru
        print("   ⚠ No label column found, creating new labels...")
        
        def create_label(row):
            intensity = row['light_intensity']
            voltage = row['voltage']
            
            if intensity >= 70 and voltage == 0:
                return 'Siang'
            elif intensity < 30 and voltage > 0:
                return 'Malam'
            else:
                return 'Senja'
        
        df['label'] = df.apply(create_label, axis=1)
        label_column = 'label'
        print("   ✅ Created 'label' column")
    
    # Tampilkan distribusi
    print(f"\n📊 Label distribution:")
    print(df[label_column].value_counts())
    
    # 2. PREPARE FEATURES (HANYA NUMERIC!)
    print("\n🔢 2. Preparing numeric features only...")
    
    # Kolom yang harus di-drop
    columns_to_drop = ['timestamp', 'date', label_column]
    # Hanya drop yang ada
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    # Features: semua kecuali yang di-drop
    X = df.drop(columns=columns_to_drop)
    
    # Pastikan semua features numeric
    print(f"   Features before cleaning: {list(X.columns)}")
    
    # Drop kolom string/kategorikal
    string_cols = X.select_dtypes(include=['object']).columns.tolist()
    if string_cols:
        print(f"   ⚠ Dropping string columns: {string_cols}")
        X = X.drop(columns=string_cols)
    
    # Tambah time features dari timestamp jika ada
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            X['hour'] = df['timestamp'].dt.hour
            X['is_night'] = ((df['timestamp'].dt.hour >= 18) | (df['timestamp'].dt.hour <= 6)).astype(int)
            print("   ✅ Added time features from timestamp")
        except:
            print("   ⚠ Could not parse timestamp")
    
    print(f"   Final features: {list(X.columns)}")
    
    # Target
    y = df[label_column]
    
    # 3. SPLIT DATA DENGAN STRATIFY
    print("\n✂️ 3. Creating train/test split (80/20) with stratification...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,  # Penting untuk maintain class distribution
        shuffle=True
    )
    
    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Test set:  {X_test.shape[0]} samples")
    
    # 4. CEK TIDAK ADA DUPLIKAT
    print("\n🔎 4. Checking for duplicates...")
    
    # Gabungkan untuk cek duplikat
    train_combined = pd.concat([X_train, y_train], axis=1)
    test_combined = pd.concat([X_test, y_test], axis=1)
    
    all_data = pd.concat([train_combined, test_combined])
    duplicates = all_data[all_data.duplicated()]
    
    if len(duplicates) > 0:
        print(f"   ⚠ Found {len(duplicates)} duplicates, removing...")
        # Remove duplicates dari test
        test_combined = test_combined.drop_duplicates()
        X_test = test_combined.drop(columns=[label_column])
        y_test = test_combined[label_column]
        print(f"   Test after removing duplicates: {X_test.shape[0]} samples")
    else:
        print("   ✅ No duplicates found")
    
    # 5. SAVE DATA
    print("\n💾 5. Saving train/test data...")
    
    # Save train
    train_df = pd.concat([X_train, y_train], axis=1)
    train_df.to_csv("train_data_proper.csv", index=False)
    print(f"   Train data: train_data_proper.csv ({len(train_df)} rows)")
    
    # Save test  
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv("test_data_proper.csv", index=False)
    print(f"   Test data:  test_data_proper.csv ({len(test_df)} rows)")
    
    # 6. VERIFICATION
    print("\n✅ 6. Verification:")
    print(f"   Train samples: {len(train_df)}")
    print(f"   Test samples:  {len(test_df)}")
    print(f"   Total:         {len(train_df) + len(test_df)}")
    print(f"   Original:      {len(df)}")
    
    print(f"\n📊 Train class distribution:")
    print(train_df[label_column].value_counts())
    
    print(f"\n📊 Test class distribution:")
    print(test_df[label_column].value_counts())
    
    # Cek persentase overlap
    print("\n🔍 Checking data overlap...")
    train_set = set([tuple(x) for x in train_df.values])
    test_set = set([tuple(x) for x in test_df.values])
    overlap = train_set.intersection(test_set)
    print(f"   Overlap samples: {len(overlap)}")
    
    if len(overlap) == 0:
        print("   ✅ PERFECT: No data leakage!")
    else:
        print(f"   ⚠ WARNING: {len(overlap)} samples in both sets")
    
    return train_df, test_df

if __name__ == "__main__":
    create_proper_split()