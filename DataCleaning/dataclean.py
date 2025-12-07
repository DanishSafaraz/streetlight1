# cleandata.py
import pandas as pd
import numpy as np
import os
import sys

# File path
FILE_PATH = r"C:\Users\LENOVO\OneDrive\Documents\SIC\Belajar\streetlight\MachineLearningSL\streetlight_data.csv"

def clean_streetlight_data():
    """
    Simple data cleaning untuk dataset streetlight
    """
    print("="*60)
    print("STREETLIGHT DATA CLEANER")
    print("="*60)
    
    # Cek file exists
    if not os.path.exists(FILE_PATH):
        print(f"❌ ERROR: File tidak ditemukan!")
        print(f"   Path: {FILE_PATH}")
        print(f"   Pastikan file ada di lokasi tersebut")
        return
    
    print(f"📁 File: {FILE_PATH}")
    
    # 1. LOAD DATA
    print("\n📥 1. Loading data...")
    try:
        df = pd.read_csv(FILE_PATH)
        print(f"   ✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"   Columns: {', '.join(df.columns)}")
    except Exception as e:
        print(f"   ❌ Error loading CSV: {e}")
        return
    
    # 2. CHECK DATA TYPES
    print("\n🔍 2. Checking data types...")
    print(df.dtypes)
    
    # 3. CHECK MISSING VALUES
    print("\n🔍 3. Checking missing values...")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("   Missing values found:")
        for col, count in missing.items():
            if count > 0:
                print(f"   - {col}: {count} missing")
    else:
        print("   ✅ No missing values")
    
    # 4. CHECK DUPLICATES
    print("\n🔍 4. Checking duplicates...")
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"   Found {duplicates} duplicate rows")
        df = df.drop_duplicates()
        print(f"   ✅ Removed duplicates")
    else:
        print("   ✅ No duplicates")
    
    # 5. CHECK DATA RANGES
    print("\n🔍 5. Checking data ranges...")
    
    # Light intensity harus 0-100%
    if 'light_intensity' in df.columns:
        invalid_intensity = df[(df['light_intensity'] < 0) | (df['light_intensity'] > 100)]
        if len(invalid_intensity) > 0:
            print(f"   ⚠ Found {len(invalid_intensity)} invalid intensity values")
            print(f"   Min: {df['light_intensity'].min()}, Max: {df['light_intensity'].max()}")
            # Fix: clip to 0-100
            df['light_intensity'] = df['light_intensity'].clip(0, 100)
            print(f"   ✅ Fixed intensity range to 0-100%")
        else:
            print(f"   ✅ Intensity range OK: {df['light_intensity'].min():.1f}% to {df['light_intensity'].max():.1f}%")
    
    # Voltage harus 0 atau 220
    if 'voltage' in df.columns:
        unique_voltages = df['voltage'].unique()
        print(f"   Unique voltages: {unique_voltages}")
        
        # Fix voltage yang tidak 0 atau 220
        mask = ~df['voltage'].isin([0.0, 220.0])
        if mask.any():
            invalid_count = mask.sum()
            print(f"   ⚠ Found {invalid_count} invalid voltage values")
            
            # Round ke 0 atau 220 terdekat
            def fix_voltage(x):
                if abs(x - 0) < abs(x - 220):
                    return 0.0
                else:
                    return 220.0
            
            df.loc[mask, 'voltage'] = df.loc[mask, 'voltage'].apply(fix_voltage)
            print(f"   ✅ Fixed invalid voltages")
    
    # 6. ADD CLASS COLUMN
    print("\n🔍 6. Adding class labels...")
    
    def determine_class(intensity, voltage):
        """Tentukan kelas berdasarkan intensitas dan voltage"""
        if voltage > 0:  # Lampu menyala
            if intensity < 30:
                return 'Malam'
            elif intensity < 70:
                return 'Senja'
            else:
                return 'Siang'  # Anomali: lampu nyala tapi terang
        else:  # Lampu mati
            if intensity > 70:
                return 'Siang'
            elif intensity > 30:
                return 'Senja'
            else:
                return 'Malam'  # Anomali: lampu mati tapi gelap
    
    df['class'] = df.apply(lambda row: determine_class(row['light_intensity'], row['voltage']), axis=1)
    
    # Tampilkan distribusi kelas
    class_dist = df['class'].value_counts()
    print("   Class distribution:")
    for cls, count in class_dist.items():
        percentage = (count / len(df)) * 100
        print(f"   - {cls}: {count} ({percentage:.1f}%)")
    
    # 7. SAVE CLEANED DATA
    print("\n💾 7. Saving cleaned data...")
    
    # Buat nama file output
    base_name = os.path.splitext(FILE_PATH)[0]
    output_file = f"{base_name}_cleaned.csv"
    
    df.to_csv(output_file, index=False)
    print(f"   ✅ Saved to: {output_file}")
    print(f"   📊 Total rows: {len(df)}")
    
    # 8. SUMMARY
    print("\n" + "="*60)
    print("CLEANING SUMMARY")
    print("="*60)
    
    print(f"Input file: {FILE_PATH}")
    print(f"Output file: {output_file}")
    print(f"Original rows: {len(pd.read_csv(FILE_PATH))}")
    print(f"Cleaned rows: {len(df)}")
    
    if 'timestamp' in df.columns:
        # Coba parse timestamp
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        except:
            print("Time range: Could not parse")
    
    print("\n📈 Data statistics:")
    if 'light_intensity' in df.columns:
        print(f"Light intensity: {df['light_intensity'].min():.1f}% - {df['light_intensity'].max():.1f}%")
        print(f"Average intensity: {df['light_intensity'].mean():.1f}%")
    
    if 'voltage' in df.columns:
        print(f"\nVoltage distribution:")
        voltage_counts = df['voltage'].value_counts()
        for volt, count in voltage_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {volt}V: {count} rows ({percentage:.1f}%)")
    
    print(f"\n🎯 Class distribution:")
    for cls, count in class_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  {cls}: {count} rows ({percentage:.1f}%)")
    
    print("\n✅ Cleaning completed successfully!")
    print("="*60)

def main():
    """Main function"""
    # Check jika pandas terinstall
    try:
        import pandas as pd
    except ImportError:
        print("❌ ERROR: Pandas library not installed!")
        print("Install dengan: pip install pandas")
        sys.exit(1)
    
    # Run cleaning
    clean_streetlight_data()
    
    # Tanya user apakah mau lihat preview
    response = input("\n📋 Show data preview? (y/n): ").lower()
    if response == 'y':
        show_preview()

def show_preview():
    """Tampilkan preview data yang sudah dibersihkan"""
    base_name = os.path.splitext(FILE_PATH)[0]
    cleaned_file = f"{base_name}_cleaned.csv"
    
    if os.path.exists(cleaned_file):
        try:
            df = pd.read_csv(cleaned_file)
            print("\n" + "="*60)
            print("DATA PREVIEW (first 10 rows)")
            print("="*60)
            print(df.head(10).to_string())
            print("="*60)
            
            # Tampilkan info singkat
            print(f"\n📊 Dataset info:")
            print(f"Rows: {len(df)}")
            print(f"Columns: {len(df.columns)}")
            print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
        except Exception as e:
            print(f"Error showing preview: {e}")
    else:
        print(f"Cleaned file not found: {cleaned_file}")

if __name__ == "__main__":
    main()