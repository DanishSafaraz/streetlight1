import pandas as pd
import numpy as np
import os

def label_streetlight_data():
    """
    STEP 3: Labeling data - Tambah/rename kolom label/target
    """
    print("="*60)
    print("STREETLIGHT DATA LABELING")
    print("="*60)
    
    # Input: cleaned data
    input_file = r"C:\Users\LENOVO\OneDrive\Documents\SIC\Belajar\streetlight\MachineLearningSL\streetlight_data_cleaned.csv"
    output_file = "streetlight_labeled.csv"
    
    if not os.path.exists(input_file):
        print(f"❌ File tidak ditemukan: {input_file}")
        print("   Jalankan cleandata.py terlebih dahulu!")
        return
    
    # Load cleaned data
    print(f"📥 Loading data: {input_file}")
    df = pd.read_csv(input_file)
    print(f"   Rows: {len(df)}, Columns: {list(df.columns)}")
    
    # CEK KOLOM YANG SUDAH ADA
    print("\n🔍 Checking existing columns...")
    
    # Jika sudah ada kolom 'class', rename ke 'label'
    if 'class' in df.columns:
        print("✅ Kolom 'class' ditemukan, rename ke 'label'")
        df = df.rename(columns={'class': 'label'})
        
        print("\n📊 Label distribution:")
        label_counts = df['label'].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {label}: {count} samples ({percentage:.1f}%)")
    
    # Jika sudah ada kolom 'label', langsung pakai
    elif 'label' in df.columns:
        print("✅ Kolom 'label' sudah ada")
        print("\n📊 Label distribution:")
        label_counts = df['label'].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {label}: {count} samples ({percentage:.1f}%)")
    
    # Jika tidak ada kolom class/label, buat baru
    else:
        print("\n🔖 Creating new labels based on business rules...")
        
        # BUAT LABEL BERDASARKAN LOGIKA BISNIS
        def create_label(row):
            """
            Rules untuk streetlight:
            1. Siang: intensity tinggi (≥70%) AND voltage = 0V (lampu mati)
            2. Malam: intensity rendah (<30%) AND voltage > 0V (lampu nyala)
            3. Senja: lainnya (30% ≤ intensity < 70%) OR kondisi tidak biasa
            """
            intensity = row['light_intensity']
            voltage = row['voltage']
            
            if intensity >= 70 and voltage == 0:
                return 'Siang'
            elif intensity < 30 and voltage > 0:
                return 'Malam'
            else:
                return 'Senja'
        
        # Apply labeling
        df['label'] = df.apply(create_label, axis=1)
        
        print("✅ Labels created!")
        print("\n📊 Label distribution:")
        label_counts = df['label'].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {label}: {count} samples ({percentage:.1f}%)")
    
    # Tambah kolom label_numeric untuk ML
    print("\n🔢 Creating numeric labels...")
    
    # Mapping label ke angka
    label_mapping = {'Siang': 0, 'Senja': 1, 'Malam': 2}
    df['label_numeric'] = df['label'].map(label_mapping)
    
    print("   Label mapping:")
    for label, num in label_mapping.items():
        count = len(df[df['label'] == label])
        print(f"   {label} → {num}: {count} samples")
    
    # Simpan data berlabel
    df.to_csv(output_file, index=False)
    print(f"\n💾 Labeled data saved: {output_file}")
    
    # Tampilkan sample (pastikan kolom yang ditampilkan ada)
    print("\n📋 Sample data (first 5 rows):")
    
    # Pilih kolom yang pasti ada
    available_cols = []
    for col in ['timestamp', 'light_intensity', 'voltage', 'label', 'label_numeric']:
        if col in df.columns:
            available_cols.append(col)
    
    if available_cols:
        print(df[available_cols].head())
    else:
        print("No suitable columns found for display")
    
    # Tampilkan info dataset
    print("\n📈 Dataset info:")
    print(f"   Total samples: {len(df)}")
    print(f"   Features: {len(df.columns)}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    return df

if __name__ == "__main__":
    label_streetlight_data()